"""
Model Wrapper
Combines the frozen DINOv2 backbone with the dynamic SFIN-U-Net decoder.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# Import our custom dynamic U-Net
from .unet import STEMEnhancementNet

class DINOSFIN_Architecture_NEW(nn.Module):
    def __init__(self, 
                 strategy="concat", 
                 base_channels=64, 
                 num_sfin=2, 
                 num_rhag=3, 
                 window_size=7,
                 dino_embed_dim=384,
                 fft_mode="fp32",
                 fft_pad_size=256,
                 use_precomputed_dino=False):
        """
        Args:
            strategy: 'concat' (Layered Depth prompting strategy)
            base_channels: Scales U-Net size (e.g., 32 for laptop, 64 for server)
            num_sfin: SFIN blocks per level
            num_rhag: RHAG blocks in the bottleneck
            window_size: 7 (Matches DINOv2 14-patch resolution at 224px)
        """
        super().__init__()
        self.strategy = strategy
        self.use_precomputed_dino = use_precomputed_dino
        self.dino_embed_dim = dino_embed_dim
        
        # 1. Initialize Foundational Backbone (DINOv2)
        # We use ViT-Small (21.4M params)
        self.encoder = None
        if not self.use_precomputed_dino:
            self.encoder = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
        
            # Freeze backbone parameters completely
            for param in self.encoder.parameters():
                param.requires_grad = False 
            
        # 2. Calculate Input Channels for Decoder
        # DINO gives 384. RGB gives 3. Prompt gives 1. Total = 388.
        in_c = dino_embed_dim + 3 + 1 
        
        # 3. Prompt Projection
        self.prompt_proj = nn.Linear(1, 1) 
        
        # 4. Initialize the Dynamic Decoder
        self.decoder = STEMEnhancementNet(
            in_channels=in_c, 
            out_channels=1,
            base_channels=base_channels,
            num_sfin=num_sfin,
            num_rhag=num_rhag,
            window_size=window_size,
            fft_mode=fft_mode,
            fft_pad_size=fft_pad_size,
        )

    def _extract_dino_features(self, x):
        if self.encoder is None:
            raise RuntimeError("DINO encoder is disabled. Provide precomputed features in forward().")

        with torch.no_grad():
            features = self.encoder.forward_features(x)["x_norm_patchtokens"]
        B, N, C = features.shape
        H = W = int(math.sqrt(N))
        dino_feats = features.permute(0, 2, 1).view(B, C, H, W)
        return F.interpolate(dino_feats, size=x.shape[2:], mode='bilinear', align_corners=False)

    def forward(self, x, target_layer=1, return_intermediate=False, use_checkpointing=False, precomputed_dino=None):
        # 1. Extract DINO Features
        if precomputed_dino is not None:
            dino_upsampled = precomputed_dino
            if dino_upsampled.shape[-2:] != x.shape[-2:]:
                dino_upsampled = F.interpolate(dino_upsampled, size=x.shape[2:], mode='bilinear', align_corners=False)
        else:
            dino_upsampled = self._extract_dino_features(x)

        B = x.shape[0]
        
        # 2. Process Layer Prompt
        # Expand prompt to match the dynamic Batch Size (B) safely on the correct GPU
        prompt = torch.tensor([[float(target_layer)]], device=x.device).expand(B, 1)
        
        # Project and expand spatially to match image dimensions
        p_vec = self.prompt_proj(prompt).view(B, 1, 1, 1).expand(-1, -1, x.shape[2], x.shape[3])
        
        # 3. Early Fusion Concatenation
        fused_input = torch.cat([x, dino_upsampled, p_vec], dim=1) 
        
        # 4. Decoder Pass
        if return_intermediate:
            # Multi-Stage Loss Routing
            out_dict = self.decoder(fused_input, return_intermediate=True, use_checkpointing=use_checkpointing)
            return {
                'bottleneck': F.softplus(out_dict['bottleneck']),
                'decoder': F.softplus(out_dict['decoder']),
                'final': F.softplus(out_dict['final'])
            }
            
        # Standard Prediction Routing
        return F.softplus(self.decoder(fused_input, return_intermediate=False, use_checkpointing=use_checkpointing))
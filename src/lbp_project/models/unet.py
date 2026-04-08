"""
U-Net Architecture
Constructs the SFIN-U-Net using dynamic parameters for local/server scaling.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

# Import the foundational blocks we built in components.py
from .components import SFINBlock, RHAG, AttentionGate
from .conditioning import AdaLNZero2d

class STEMEnhancementNet(nn.Module):
    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        base_channels=64,
        num_sfin=2,
        num_rhag=3,
        window_size=7,
        fft_mode="fp32",
        fft_pad_size=256,
        adaln_zero_enabled=False,
        adaln_condition_dim=128,
    ):
        """
        Args:
            in_channels: 388 (DINO 384 + RGB 3 + Prompt 1)
            out_channels: 1 (Depth Map)
            base_channels: 32 for Local (Fast), 64 for Server (Accurate)
            num_sfin: 1 for Local, 2 for Server
            num_rhag: 2 for Local, 3 for Server
            window_size: 7 (Required for DINOv2 224px alignment)
        """
        super().__init__()
        self.window_size = window_size
        self.adaln_zero_enabled = bool(adaln_zero_enabled)
        
        # Dynamic channel scaling based on the injected base_channels
        self.channels = [base_channels, base_channels * 2, base_channels * 4, base_channels * 8]
        
        self.input_conv = nn.Conv2d(in_channels, self.channels[0], 3, 1, 1)
        
        # --- ENCODER ---
        self.encoder1 = SFINBlock(self.channels[0], self.channels[0], num_sfin, fft_mode=fft_mode, fft_pad_size=fft_pad_size)
        self.down1 = nn.Conv2d(self.channels[0], self.channels[1], 3, stride=2, padding=1)
        
        self.encoder2 = SFINBlock(self.channels[1], self.channels[1], num_sfin, fft_mode=fft_mode, fft_pad_size=fft_pad_size)
        self.down2 = nn.Conv2d(self.channels[1], self.channels[2], 3, stride=2, padding=1)
        
        self.encoder3 = SFINBlock(self.channels[2], self.channels[2], num_sfin, fft_mode=fft_mode, fft_pad_size=fft_pad_size)
        self.down3 = nn.Conv2d(self.channels[2], self.channels[3], 3, stride=2, padding=1)
        
        self.encoder4 = SFINBlock(self.channels[3], self.channels[3], num_sfin, fft_mode=fft_mode, fft_pad_size=fft_pad_size)
        self.down4 = nn.Conv2d(self.channels[3], self.channels[3], 3, stride=2, padding=1)
        
        # --- BOTTLENECK ---
        self.bottleneck = RHAG(dim=self.channels[3], num_heads=8, num_blocks=num_rhag, window_size=window_size)
        
        # --- DECODER ---
        self.up1 = nn.ConvTranspose2d(self.channels[3], self.channels[3], 4, stride=2, padding=1)
        self.ag1 = AttentionGate(F_g=self.channels[3], F_l=self.channels[3], F_int=self.channels[2])
        self.decoder1 = SFINBlock(self.channels[3] * 2, self.channels[3], num_sfin, fft_mode=fft_mode, fft_pad_size=fft_pad_size)
        
        self.up2 = nn.ConvTranspose2d(self.channels[3], self.channels[2], 4, stride=2, padding=1)
        self.ag2 = AttentionGate(F_g=self.channels[2], F_l=self.channels[2], F_int=self.channels[1])
        self.decoder2 = SFINBlock(self.channels[2] * 2, self.channels[2], num_sfin, fft_mode=fft_mode, fft_pad_size=fft_pad_size)
        
        self.up3 = nn.ConvTranspose2d(self.channels[2], self.channels[1], 4, stride=2, padding=1)
        self.ag3 = AttentionGate(F_g=self.channels[1], F_l=self.channels[1], F_int=self.channels[0])
        self.decoder3 = SFINBlock(self.channels[1] * 2, self.channels[1], num_sfin, fft_mode=fft_mode, fft_pad_size=fft_pad_size)
        
        self.up4 = nn.ConvTranspose2d(self.channels[1], self.channels[0], 4, stride=2, padding=1)
        self.ag4 = AttentionGate(F_g=self.channels[0], F_l=self.channels[0], F_int=self.channels[0] // 2)
        self.decoder4 = SFINBlock(self.channels[0] * 2, self.channels[0], num_sfin, fft_mode=fft_mode, fft_pad_size=fft_pad_size)
        
        # --- OUTPUT PROJECTIONS ---
        self.output_conv = nn.Conv2d(self.channels[0], out_channels, 3, 1, 1)
        self.bottleneck_proj = nn.Conv2d(self.channels[3], out_channels, 1, 1, 0)
        self.decoder_proj = nn.Conv2d(self.channels[0], out_channels, 1, 1, 0)

        self.adaln_blocks = None
        if self.adaln_zero_enabled:
            cond_dim = int(adaln_condition_dim)
            if cond_dim < 1:
                raise ValueError(f"adaln_condition_dim must be >= 1, got {cond_dim}")
            self.adaln_blocks = nn.ModuleDict(
                {
                    "input": AdaLNZero2d(self.channels[0], cond_dim),
                    "enc1": AdaLNZero2d(self.channels[0], cond_dim),
                    "enc2": AdaLNZero2d(self.channels[1], cond_dim),
                    "enc3": AdaLNZero2d(self.channels[2], cond_dim),
                    "enc4": AdaLNZero2d(self.channels[3], cond_dim),
                    "bottleneck": AdaLNZero2d(self.channels[3], cond_dim),
                    "dec1": AdaLNZero2d(self.channels[3], cond_dim),
                    "dec2": AdaLNZero2d(self.channels[2], cond_dim),
                    "dec3": AdaLNZero2d(self.channels[1], cond_dim),
                    "dec4": AdaLNZero2d(self.channels[0], cond_dim),
                }
            )

    def _apply_adaln(self, key, x, conditioning):
        if not self.adaln_zero_enabled:
            return x
        if self.adaln_blocks is None:
            raise RuntimeError("Internal error: AdaLN-Zero enabled but blocks are missing")
        if conditioning is None:
            raise ValueError("conditioning tensor is required when adaln_zero_enabled=true")
        return self.adaln_blocks[key](x, conditioning)

    def _align_to(self, src, ref):
        if src.shape[-2:] == ref.shape[-2:]:
            return src
        return F.interpolate(src, size=ref.shape[-2:], mode="bilinear", align_corners=False)
        
    def forward(self, x, return_intermediate=False, use_checkpointing=False, conditioning=None):
        """
        use_checkpointing: Set to True to save VRAM at the cost of ~20% slower compute.
        """
        # Helper function for Gradient Checkpointing
        def run_block(block, *inputs):
            return block(*inputs)

        x = self.input_conv(x)
        x = self._apply_adaln("input", x, conditioning)
        
        # Encoder Pass (With Optional Checkpointing for Heavy SFIN Blocks)
        if use_checkpointing:
            enc1 = checkpoint(run_block, self.encoder1, x, use_reentrant=False)
            enc1 = self._apply_adaln("enc1", enc1, conditioning)
            enc2 = checkpoint(run_block, self.encoder2, self.down1(enc1), use_reentrant=False)
            enc2 = self._apply_adaln("enc2", enc2, conditioning)
            enc3 = checkpoint(run_block, self.encoder3, self.down2(enc2), use_reentrant=False)
            enc3 = self._apply_adaln("enc3", enc3, conditioning)
            enc4 = checkpoint(run_block, self.encoder4, self.down3(enc3), use_reentrant=False)
            enc4 = self._apply_adaln("enc4", enc4, conditioning)
            
            bottleneck_feat = checkpoint(run_block, self.bottleneck, self.down4(enc4), use_reentrant=False)
            bottleneck_feat = self._apply_adaln("bottleneck", bottleneck_feat, conditioning)
        else:
            enc1 = self.encoder1(x)
            enc1 = self._apply_adaln("enc1", enc1, conditioning)
            enc2 = self.encoder2(self.down1(enc1))
            enc2 = self._apply_adaln("enc2", enc2, conditioning)
            enc3 = self.encoder3(self.down2(enc2))
            enc3 = self._apply_adaln("enc3", enc3, conditioning)
            enc4 = self.encoder4(self.down3(enc3))
            enc4 = self._apply_adaln("enc4", enc4, conditioning)
            
            bottleneck_feat = self.bottleneck(self.down4(enc4))
            bottleneck_feat = self._apply_adaln("bottleneck", bottleneck_feat, conditioning)
            
        # Decoder Pass
        up1_feat = self._align_to(self.up1(bottleneck_feat), enc4)
        x = self.decoder1(torch.cat([up1_feat, self.ag1(up1_feat, enc4)], dim=1))
        x = self._apply_adaln("dec1", x, conditioning)
        
        up2_feat = self._align_to(self.up2(x), enc3)
        x = self.decoder2(torch.cat([up2_feat, self.ag2(up2_feat, enc3)], dim=1))
        x = self._apply_adaln("dec2", x, conditioning)
        
        up3_feat = self._align_to(self.up3(x), enc2)
        x = self.decoder3(torch.cat([up3_feat, self.ag3(up3_feat, enc2)], dim=1))
        x = self._apply_adaln("dec3", x, conditioning)
        
        up4_feat = self._align_to(self.up4(x), enc1)
        decoder_feat = self.decoder4(torch.cat([up4_feat, self.ag4(up4_feat, enc1)], dim=1))
        decoder_feat = self._apply_adaln("dec4", decoder_feat, conditioning)
        
        output = self.output_conv(decoder_feat)
        
        if return_intermediate:
            return {
                'bottleneck': self.bottleneck_proj(bottleneck_feat),
                'decoder': self.decoder_proj(decoder_feat),
                'final': output,
                'decoder_features': decoder_feat,
            }
        return output
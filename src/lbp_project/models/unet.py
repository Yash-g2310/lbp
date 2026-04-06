"""
U-Net Architecture
Constructs the SFIN-U-Net using dynamic parameters for local/server scaling.
"""

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

# Import the foundational blocks we built in components.py
from .components import SFINBlock, RHAG, AttentionGate

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
        
    def forward(self, x, return_intermediate=False, use_checkpointing=False):
        """
        use_checkpointing: Set to True to save VRAM at the cost of ~20% slower compute.
        """
        # Helper function for Gradient Checkpointing
        def run_block(block, *inputs):
            return block(*inputs)

        x = self.input_conv(x)
        if x.shape[-2] % self.window_size != 0 or x.shape[-1] % self.window_size != 0:
            raise ValueError(
                f"Input spatial size ({x.shape[-2]}x{x.shape[-1]}) must be divisible by window_size={self.window_size}."
            )
        
        # Encoder Pass (With Optional Checkpointing for Heavy SFIN Blocks)
        if use_checkpointing:
            enc1 = checkpoint(run_block, self.encoder1, x, use_reentrant=False)
            enc2 = checkpoint(run_block, self.encoder2, self.down1(enc1), use_reentrant=False)
            enc3 = checkpoint(run_block, self.encoder3, self.down2(enc2), use_reentrant=False)
            enc4 = checkpoint(run_block, self.encoder4, self.down3(enc3), use_reentrant=False)
            
            bottleneck_feat = checkpoint(run_block, self.bottleneck, self.down4(enc4), use_reentrant=False)
        else:
            enc1 = self.encoder1(x)
            enc2 = self.encoder2(self.down1(enc1))
            enc3 = self.encoder3(self.down2(enc2))
            enc4 = self.encoder4(self.down3(enc3))
            
            bottleneck_feat = self.bottleneck(self.down4(enc4))
            
        # Decoder Pass
        up1_feat = self.up1(bottleneck_feat)
        x = self.decoder1(torch.cat([up1_feat, self.ag1(up1_feat, enc4)], dim=1))
        
        up2_feat = self.up2(x)
        x = self.decoder2(torch.cat([up2_feat, self.ag2(up2_feat, enc3)], dim=1))
        
        up3_feat = self.up3(x)
        x = self.decoder3(torch.cat([up3_feat, self.ag3(up3_feat, enc2)], dim=1))
        
        up4_feat = self.up4(x)
        decoder_feat = self.decoder4(torch.cat([up4_feat, self.ag4(up4_feat, enc1)], dim=1))
        
        output = self.output_conv(decoder_feat)
        
        if return_intermediate:
            return {
                'bottleneck': self.bottleneck_proj(bottleneck_feat),
                'decoder': self.decoder_proj(decoder_feat),
                'final': output
            }
        return output
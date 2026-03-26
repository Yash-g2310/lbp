"""
Core Neural Network Components
Contains the building blocks for the SFIN-U-Net architecture.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast
from contextlib import nullcontext
import warnings

_dynamo_disable = torch._dynamo.disable if hasattr(torch, "_dynamo") else (lambda fn: fn)

# ==========================================
# 1. HYBRID ATTENTION BLOCKS (RHAG)
# ==========================================
class ChannelAttention(nn.Module):
    def __init__(self, num_feat, squeeze_factor=16):
        super().__init__()
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(num_feat, num_feat // squeeze_factor, 1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_feat // squeeze_factor, num_feat, 1, padding=0),
            nn.Sigmoid()
        )
    def forward(self, x):
        return x * self.attention(x)

class CAB(nn.Module):
    def __init__(self, num_feat, compress_ratio=3, squeeze_factor=30):
        super().__init__()
        self.cab = nn.Sequential(
            nn.Conv2d(num_feat, num_feat // compress_ratio, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(num_feat // compress_ratio, num_feat, 3, 1, 1),
            ChannelAttention(num_feat, squeeze_factor)
        )
    def forward(self, x):
        return self.cab(x)

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features or in_features)
        self.drop = nn.Dropout(drop)
    def forward(self, x):
        return self.drop(self.fc2(self.drop(self.act(self.fc1(x)))))

def window_partition(x, window_size):
    B, H, W, C = x.shape
    if H % window_size != 0 or W % window_size != 0:
        raise ValueError(f"Window size {window_size} must divide HxW ({H}x{W}).")
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)

def window_reverse(windows, window_size, H, W):
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)

class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        
    def forward(self, x):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
        attn = F.softmax((q @ k.transpose(-2, -1)), dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        return self.proj(x)

class HAB(nn.Module):
    def __init__(self, dim, num_heads, window_size=8, mlp_ratio=4.):
        super().__init__()
        self.window_size = window_size
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, window_size=(window_size, window_size), num_heads=num_heads)
        self.conv_block = CAB(num_feat=dim)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio))
        
    def forward(self, x):
        B, C, H, W = x.shape
        shortcut = x
        x = self.norm1(x.permute(0, 2, 3, 1).contiguous())
        x_windows = window_partition(x, self.window_size).view(-1, self.window_size * self.window_size, C)
        attn_windows = self.attn(x_windows).view(-1, self.window_size, self.window_size, C)
        x = window_reverse(attn_windows, self.window_size, H, W).permute(0, 3, 1, 2).contiguous()
        x = shortcut + x
        x = x + self.conv_block(x)
        shortcut = x
        x = self.mlp(self.norm2(x.permute(0, 2, 3, 1).contiguous().view(B, H * W, C)))
        return shortcut + x.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()

class RHAG(nn.Module):
    def __init__(self, dim, num_heads=8, num_blocks=3, window_size=8, mlp_ratio=4.):
        super().__init__()
        self.blocks = nn.ModuleList([HAB(dim, num_heads, window_size, mlp_ratio) for _ in range(num_blocks)])
        self.conv = nn.Conv2d(dim, dim, 3, 1, 1)
        
    def forward(self, x):
        shortcut = x
        for block in self.blocks: 
            x = block(x)
        return self.conv(x) + shortcut

# ==========================================
# 2. SFIN BLOCK (Spatial-Frequency Interactive Network)
# ==========================================
class FourierUnit(nn.Module):
    def __init__(self, in_channels, out_channels, fft_mode="fp32", fft_pad_size=256):
        super().__init__()
        self.conv_layer = nn.Conv2d(in_channels * 2 + 2, out_channels * 2, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels * 2)
        self.relu = nn.ReLU(inplace=True)
        self.fft_mode = fft_mode
        self.fft_pad_size = fft_pad_size

    def _frequency_conv(self, x_fft, h, w):
        batch = x_fft.shape[0]
        # view_as_real avoids an extra allocation versus stacking real/imag tensors.
        ffted = torch.view_as_real(x_fft).permute(0, 1, 4, 2, 3).contiguous()
        ffted = ffted.view((batch, -1,) + ffted.size()[3:])

        cv = torch.linspace(0, 1, h, dtype=torch.float32, device=ffted.device)[None, None, :, None].expand(batch, 1, h, w)
        ch = torch.linspace(0, 1, w, dtype=torch.float32, device=ffted.device)[None, None, None, :].expand(batch, 1, h, w)
        ffted = self.relu(self.bn(self.conv_layer(torch.cat((cv, ch, ffted), dim=1))))
        ffted = ffted.view((batch, -1, 2,) + ffted.size()[2:]).permute(0, 1, 3, 4, 2).contiguous()
        return torch.complex(ffted[..., 0], ffted[..., 1])

    @_dynamo_disable
    def forward(self, x):
        orig_dtype = x.dtype
        # If input contains non-finite values in reduced precision (bf16/fp16),
        # attempt a recovery by casting to float32 and continue. Otherwise
        # raise with diagnostic counts.
        if not torch.isfinite(x).all():
            if x.is_floating_point() and x.dtype in (torch.bfloat16, torch.float16):
                x32 = x.to(torch.float32)
                if torch.isfinite(x32).all():
                    warnings.warn(
                        "FourierUnit: non-finite input detected in reduced precision; casting input to float32 for stability",
                        RuntimeWarning,
                    )
                    x = x32
                else:
                    finite_count = int(torch.isfinite(x32).sum().item())
                    total = int(x32.numel())
                    nan_count = int(torch.isnan(x32).sum().item())
                    inf_count = int(torch.isinf(x32).sum().item())
                    raise RuntimeError(
                        "FourierUnit received non-finite input even after casting to float32. "
                        f"mode={self.fft_mode} input_dtype={orig_dtype} finite={finite_count}/{total} nan={nan_count} inf={inf_count} shape={tuple(x32.shape)}"
                    )
            else:
                finite_count = int(torch.isfinite(x).sum().item())
                total = int(x.numel())
                nan_count = int(torch.isnan(x).sum().item())
                inf_count = int(torch.isinf(x).sum().item())
                raise RuntimeError(
                    "FourierUnit received non-finite input. "
                    f"mode={self.fft_mode} input_dtype={orig_dtype} finite={finite_count}/{total} nan={nan_count} inf={inf_count} shape={tuple(x.shape)}"
                )

        def _run_fp32_fft(x_in: torch.Tensor, spatial_shape: tuple[int, int]) -> torch.Tensor:
            x_float = x_in.to(torch.float32)
            fp32_ctx = autocast('cuda', enabled=False) if x_float.is_cuda else nullcontext()
            with fp32_ctx:
                ffted_local = torch.fft.rfftn(x_float, dim=(-2, -1), norm='ortho')
                h_local, w_local = ffted_local.shape[-2:]
                ffted_local = self._frequency_conv(ffted_local, h_local, w_local)
                out_local = torch.fft.irfftn(ffted_local, s=spatial_shape, dim=(-2, -1), norm='ortho')
            return out_local

        if self.fft_mode == "pad_fp16":
            h_in, w_in = x.shape[-2:]

            # Pad to the next power-of-two up to fft_pad_size so deep features
            # (e.g. 112x112) do not require oversized reflection padding.
            target_h = min(self.fft_pad_size, max(h_in, 1 << (max(1, h_in) - 1).bit_length()))
            target_w = min(self.fft_pad_size, max(w_in, 1 << (max(1, w_in) - 1).bit_length()))
            pad_h = max(0, target_h - h_in)
            pad_w = max(0, target_w - w_in)
            if pad_h > 0 or pad_w > 0:
                pad_mode = "reflect" if (pad_h < h_in and pad_w < w_in and h_in > 1 and w_in > 1) else "replicate"
                x_fft_in = F.pad(x, (0, pad_w, 0, pad_h), mode=pad_mode)
            else:
                x_fft_in = x

            # Keep FFT compute on supported reduced precision under global AMP.
            # rfftn does not accept bf16 on some CUDA stacks.
            fft_in = x_fft_in.to(torch.float16) if x_fft_in.is_cuda and x_fft_in.dtype == torch.bfloat16 else x_fft_in
            fft_ctx = autocast('cuda', enabled=False) if fft_in.is_cuda else nullcontext()
            with fft_ctx:
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore",
                        message="ComplexHalf support is experimental",
                        category=UserWarning,
                    )
                    try:
                        ffted = torch.fft.rfftn(fft_in, dim=(-2, -1), norm='ortho')
                    except RuntimeError as exc:
                        if "Unsupported dtype" not in str(exc):
                            raise
                        ffted = torch.fft.rfftn(fft_in.to(torch.float32), dim=(-2, -1), norm='ortho')
                h_fft, w_fft = ffted.shape[-2:]
                # Keep pad_fp16 in reduced precision where possible to control memory footprint.
                ffted = self._frequency_conv(ffted, h_fft, w_fft)
                output = torch.fft.irfftn(ffted, s=fft_in.shape[-2:], dim=(-2, -1), norm='ortho')

            if not torch.isfinite(output).all():
                # Recover from unstable reduced-precision frequency path.
                warnings.warn(
                    "FourierUnit pad_fp16 produced non-finite output; retrying FFT path in fp32",
                    RuntimeWarning,
                )
                output = _run_fp32_fft(x_fft_in, x_fft_in.shape[-2:])

            output = output[..., :h_in, :w_in]
        else:
            # Fallback path for numerical stability across hardware.
            output = _run_fp32_fft(x, x.shape[-2:])

        if not torch.isfinite(output).all():
            finite_count = int(torch.isfinite(output).sum().item())
            total = int(output.numel())
            raise RuntimeError(
                "FourierUnit produced non-finite output. "
                f"mode={self.fft_mode} input_dtype={orig_dtype} output_dtype={output.dtype} "
                f"finite={finite_count}/{total} shape={tuple(output.shape)}"
            )

        return output.to(orig_dtype)

class SpectralTransform(nn.Module):
    def __init__(self, channels, fft_mode="fp32", fft_pad_size=256):
        super().__init__()
        self.conv1 = nn.Conv2d(channels // 2, channels // 2, 3, 1, 1)
        self.fu = FourierUnit(channels // 2, channels // 2, fft_mode=fft_mode, fft_pad_size=fft_pad_size)
        self.conv2 = nn.Conv2d(channels, channels // 2, 3, 1, 1)
    def forward(self, x):
        # Quick parameter finiteness checks to catch corrupted weights/biases.
        w = self.conv1.weight
        if not torch.isfinite(w).all():
            finite_count = int(torch.isfinite(w).sum().item())
            total = int(w.numel())
            nan_count = int(torch.isnan(w).sum().item())
            inf_count = int(torch.isinf(w).sum().item())
            raise RuntimeError(
                "SpectralTransform.conv1 weight contains non-finite values. "
                f"finite={finite_count}/{total} nan={nan_count} inf={inf_count} shape={tuple(w.shape)} dtype={w.dtype}"
            )
        if self.conv1.bias is not None:
            b = self.conv1.bias
            if not torch.isfinite(b).all():
                finite_count = int(torch.isfinite(b).sum().item())
                total = int(b.numel())
                nan_count = int(torch.isnan(b).sum().item())
                inf_count = int(torch.isinf(b).sum().item())
                raise RuntimeError(
                    "SpectralTransform.conv1 bias contains non-finite values. "
                    f"finite={finite_count}/{total} nan={nan_count} inf={inf_count} shape={tuple(b.shape)} dtype={b.dtype}"
                )

        # Protect against upstream NaNs by checking activations immediately after conv1.
        x1 = self.conv1(x)
        if not torch.isfinite(x1).all():
            finite_count = int(torch.isfinite(x1).sum().item())
            total = int(x1.numel())
            nan_count = int(torch.isnan(x1).sum().item())
            inf_count = int(torch.isinf(x1).sum().item())
            raise RuntimeError(
                "SpectralTransform.conv1 produced non-finite activations. "
                f"finite={finite_count}/{total} nan={nan_count} inf={inf_count} shape={tuple(x1.shape)} dtype={x1.dtype}"
            )
        return self.conv2(torch.cat([x, self.fu(x1)], dim=1))

class FFC(nn.Module):
    def __init__(self, channels, fft_mode="fp32", fft_pad_size=256):
        super().__init__()
        self.convl2l = nn.Conv2d(channels // 2, channels // 2, 3, 1, 1)
        self.convl2g = nn.Conv2d(channels // 2, channels // 2, 3, 1, 1)
        self.convg2l = nn.Conv2d(channels // 2, channels // 2, 3, 1, 1)
        self.convg2g = SpectralTransform(channels, fft_mode=fft_mode, fft_pad_size=fft_pad_size)
    def forward(self, x):
        x_l, x_g = x if isinstance(x, tuple) else (x, torch.zeros_like(x))
        return self.convl2l(x_l) + self.convg2l(x_g), self.convl2g(x_l) + self.convg2g(x_g)

class SFIB(nn.Module):
    def __init__(self, channels, fft_mode="fp32", fft_pad_size=256):
        super().__init__()
        self.ffc = FFC(channels, fft_mode=fft_mode, fft_pad_size=fft_pad_size)
        self.bn_l, self.bn_g = nn.BatchNorm2d(channels // 2), nn.BatchNorm2d(channels // 2)
        self.act_l, self.act_g = nn.ReLU(inplace=True), nn.ReLU(inplace=True)
    def forward(self, x):
        x_l, x_g = self.ffc(x)
        return self.act_l(self.bn_l(x_l)), self.act_g(self.bn_g(x_g))

class SFINResBlock(nn.Module):
    def __init__(self, channels, fft_mode="fp32", fft_pad_size=256):
        super().__init__()
        self.conv1, self.conv2 = SFIB(channels, fft_mode=fft_mode, fft_pad_size=fft_pad_size), SFIB(channels, fft_mode=fft_mode, fft_pad_size=fft_pad_size)
    def forward(self, x):
        c = x.shape[1]
        x_l, x_g = torch.split(x, (c // 2, c - c // 2), dim=1)
        id_l, id_g = x_l, x_g
        x_l, x_g = self.conv1((x_l, x_g))
        x_l, x_g = self.conv2((x_l, x_g))
        return torch.cat((id_l + x_l, id_g + x_g), dim=1)

class SFINBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_blocks=2, fft_mode="fp32", fft_pad_size=256):
        super().__init__()
        self.input_conv = nn.Conv2d(in_channels, out_channels, 3, 1, 1)
        self.blocks = nn.Sequential(
            *[
                SFINResBlock(out_channels, fft_mode=fft_mode, fft_pad_size=fft_pad_size)
                for _ in range(num_blocks)
            ]
        )
        self.output_conv = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
    def forward(self, x):
        x = self.input_conv(x)
        return self.output_conv(self.blocks(x) + x)

# ==========================================
# 3. ATTENTION GATE 
# ==========================================
class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(nn.Conv2d(F_g, F_int, 1, bias=True), nn.BatchNorm2d(F_int))
        self.W_x = nn.Sequential(nn.Conv2d(F_l, F_int, 1, bias=True), nn.BatchNorm2d(F_int))
        self.psi = nn.Sequential(nn.Conv2d(F_int, 1, 1, bias=True), nn.BatchNorm2d(1), nn.Sigmoid())
    def forward(self, g, x):
        if g.shape[2:] != x.shape[2:]:
            g = F.interpolate(g, size=x.shape[2:], mode='bilinear', align_corners=False)
        return x * self.psi(F.relu(self.W_g(g) + self.W_x(x), inplace=True))
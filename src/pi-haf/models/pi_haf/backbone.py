from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import FlowRHAG, FlowSFINBlock
from .conditioning import CrossAttention2D, LRConditionEncoder
from .config import PIHAFConfig
from .embeddings import SinusoidalTimeEmbedding
from .utils import _validate_nchw


class PIHAFBackbone(nn.Module):
    """PI-HAF architecture for rectified-flow velocity prediction.

    Contract:
    - x_t: [B, in_channels, 150, 150] (HR-space ODE state)
    - t: [B] or [B,1]
    - lr_img: [B, lr_channels, 75, 75]
    - output: [B, out_channels, 150, 150]
    """

    def __init__(
        self,
        config: Optional[Union[PIHAFConfig, Dict[str, Any]]] = None,
        **kwargs: Any,
    ):
        super().__init__()

        if config is None:
            cfg = PIHAFConfig(**kwargs)
        elif isinstance(config, dict):
            merged = dict(config)
            if isinstance(merged.get("input_size"), list):
                merged["input_size"] = tuple(merged["input_size"])
            if isinstance(merged.get("cross_attention_stages"), list):
                merged["cross_attention_stages"] = tuple(merged["cross_attention_stages"])
            merged.update(kwargs)
            cfg = PIHAFConfig(**merged)
        elif isinstance(config, PIHAFConfig):
            if kwargs:
                raise ValueError(
                    "When passing PIHAFConfig, kwargs must be empty to avoid ambiguity"
                )
            cfg = config
        else:
            raise TypeError(
                f"config must be None, dict, or PIHAFConfig. Got {type(config).__name__}"
            )

        cfg.validate()
        self.config = cfg

        c = self.config.hidden_channels

        self.time_embed = SinusoidalTimeEmbedding(embed_dim=self.config.time_embed_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(self.config.time_embed_dim, self.config.time_mlp_dim),
            nn.SiLU(),
            nn.Linear(self.config.time_mlp_dim, self.config.time_mlp_dim),
        )

        # HR-space state adapter with LR anchor:
        # [B,(in_channels+lr_channels),150,150] -> [B,(in_channels+lr_channels)*4,75,75].
        self.state_unshuffle = nn.PixelUnshuffle(downscale_factor=2)
        self.input_proj = nn.Conv2d(
            (self.config.in_channels + self.config.lr_channels) * 4,
            c,
            kernel_size=3,
            padding=1,
        )

        self.lr_encoder = LRConditionEncoder(
            in_channels=self.config.lr_channels,
            hidden_channels=c,
            num_feature_levels=self.config.total_stages,
        )

        self.encoder_sfin = nn.ModuleList(
            [
                FlowSFINBlock(
                    channels=c,
                    time_embed_dim=self.config.time_mlp_dim,
                    ffn_expansion=self.config.sfin_ffn_expansion,
                    dropout=self.config.dropout,
                    use_gradient_checkpointing=self.config.use_gradient_checkpointing,
                )
                for _ in range(self.config.num_encoder_stages)
            ]
        )
        self.encoder_rhag = nn.ModuleList(
            [
                FlowRHAG(
                    channels=c,
                    time_embed_dim=self.config.time_mlp_dim,
                    num_heads=self.config.num_heads,
                    num_blocks=1,
                    window_size=self.config.window_size,
                    mlp_ratio=self.config.rhag_mlp_ratio,
                    dropout=self.config.dropout,
                    pad_mode=self.config.pad_mode,
                    strict_76_mode=self.config.strict_76_mode,
                    use_gradient_checkpointing=self.config.use_gradient_checkpointing,
                )
                for _ in range(self.config.num_encoder_stages)
            ]
        )

        self.bottleneck_sfin = nn.ModuleList(
            [
                FlowSFINBlock(
                    channels=c,
                    time_embed_dim=self.config.time_mlp_dim,
                    ffn_expansion=self.config.sfin_ffn_expansion,
                    dropout=self.config.dropout,
                    use_gradient_checkpointing=self.config.use_gradient_checkpointing,
                )
                for _ in range(self.config.num_bottleneck_sfin_blocks)
            ]
        )
        self.bottleneck_rhag = FlowRHAG(
            channels=c,
            time_embed_dim=self.config.time_mlp_dim,
            num_heads=self.config.num_heads,
            num_blocks=self.config.num_bottleneck_rhag_blocks,
            window_size=self.config.window_size,
            mlp_ratio=self.config.rhag_mlp_ratio,
            dropout=self.config.dropout,
            pad_mode=self.config.pad_mode,
            strict_76_mode=self.config.strict_76_mode,
            use_gradient_checkpointing=self.config.use_gradient_checkpointing,
        )

        self.decoder_sfin = nn.ModuleList(
            [
                FlowSFINBlock(
                    channels=c,
                    time_embed_dim=self.config.time_mlp_dim,
                    ffn_expansion=self.config.sfin_ffn_expansion,
                    dropout=self.config.dropout,
                    use_gradient_checkpointing=self.config.use_gradient_checkpointing,
                )
                for _ in range(self.config.num_decoder_stages)
            ]
        )
        self.decoder_rhag = nn.ModuleList(
            [
                FlowRHAG(
                    channels=c,
                    time_embed_dim=self.config.time_mlp_dim,
                    num_heads=self.config.num_heads,
                    num_blocks=1,
                    window_size=self.config.window_size,
                    mlp_ratio=self.config.rhag_mlp_ratio,
                    dropout=self.config.dropout,
                    pad_mode=self.config.pad_mode,
                    strict_76_mode=self.config.strict_76_mode,
                    use_gradient_checkpointing=self.config.use_gradient_checkpointing,
                )
                for _ in range(self.config.num_decoder_stages)
            ]
        )

        self.skip_gates = nn.ParameterList(
            [nn.Parameter(torch.tensor(1.0)) for _ in range(self.config.num_decoder_stages)]
        )

        stage_set = set(self.config.cross_attention_stages)
        self.cross_attention = nn.ModuleDict(
            {
                str(stage_idx): CrossAttention2D(
                    channels=c,
                    num_heads=self.config.num_heads,
                    dropout=self.config.dropout,
                    use_gradient_checkpointing=self.config.checkpoint_cross_attention,
                )
                for stage_idx in sorted(stage_set)
            }
        )

        self.pre_shuffle = nn.Conv2d(
            c,
            self.config.output_head_channels * 4,
            kernel_size=3,
            padding=1,
        )
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor=2)
        self.output_conv = nn.Conv2d(
            self.config.output_head_channels,
            self.config.out_channels,
            kernel_size=3,
            padding=1,
        )

    def _encode_time(self, t: torch.Tensor, batch_size: int, device: torch.device) -> torch.Tensor:
        if t.ndim == 0:
            t = t.expand(batch_size)
        elif t.ndim == 2 and t.shape[1] == 1:
            t = t.squeeze(1)

        if t.ndim != 1:
            raise ValueError(f"t must have shape [B], [B,1], or scalar, got {tuple(t.shape)}")
        if t.shape[0] != batch_size:
            raise ValueError(f"t batch mismatch: expected B={batch_size}, got {t.shape[0]}")

        t = t.to(device=device, dtype=torch.float32)
        return self.time_mlp(self.time_embed(t))

    def _apply_cross_attention(
        self,
        stage_idx: int,
        x: torch.Tensor,
        lr_features: Sequence[torch.Tensor],
    ) -> torch.Tensor:
        key = str(stage_idx)
        if key not in self.cross_attention:
            return x
        if stage_idx < 0 or stage_idx >= len(lr_features):
            raise ValueError(
                f"stage_idx {stage_idx} out of range for lr_features length {len(lr_features)}"
            )
        return self.cross_attention[key](x, lr_features[stage_idx])

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, lr_img: torch.Tensor) -> torch.Tensor:
        _validate_nchw(
            "x_t",
            x_t,
            expected_channels=self.config.in_channels,
            expected_hw=self.config.state_size,
        )
        _validate_nchw(
            "lr_img",
            lr_img,
            expected_channels=self.config.lr_channels,
            expected_hw=self.config.input_size,
        )
        if x_t.shape[0] != lr_img.shape[0]:
            raise ValueError(
                f"Batch mismatch between x_t and lr_img: {x_t.shape[0]} vs {lr_img.shape[0]}"
            )

        batch_size = x_t.shape[0]
        t_emb = self._encode_time(t=t, batch_size=batch_size, device=x_t.device)

        lr_upscaled = F.interpolate(
            lr_img,
            size=self.config.state_size,
            mode="bilinear",
            align_corners=False,
        )
        anchored_state = torch.cat([x_t, lr_upscaled], dim=1)
        unshuffled = self.state_unshuffle(anchored_state)
        _validate_nchw(
            "unshuffled",
            unshuffled,
            expected_channels=(self.config.in_channels + self.config.lr_channels) * 4,
            expected_hw=self.config.input_size,
        )

        x = self.input_proj(unshuffled)
        lr_features = self.lr_encoder(lr_img)
        if len(lr_features) != self.config.total_stages:
            raise RuntimeError(
                "LRConditionEncoder produced incorrect stage count. "
                f"expected={self.config.total_stages}, got={len(lr_features)}"
            )

        skips: List[torch.Tensor] = []

        for idx in range(self.config.num_encoder_stages):
            x = self.encoder_sfin[idx](x, t_emb)
            x = self.encoder_rhag[idx](x, t_emb)
            x = self._apply_cross_attention(stage_idx=idx, x=x, lr_features=lr_features)
            skips.append(x)

        for block in self.bottleneck_sfin:
            x = block(x, t_emb)
        x = self.bottleneck_rhag(x, t_emb)
        x = self._apply_cross_attention(
            stage_idx=self.config.num_encoder_stages,
            x=x,
            lr_features=lr_features,
        )

        for idx in range(self.config.num_decoder_stages):
            skip = skips[-(idx + 1)]
            x = x + torch.tanh(self.skip_gates[idx]) * skip
            x = self.decoder_sfin[idx](x, t_emb)
            x = self.decoder_rhag[idx](x, t_emb)
            stage_idx = self.config.num_encoder_stages + 1 + idx
            x = self._apply_cross_attention(stage_idx=stage_idx, x=x, lr_features=lr_features)

        x = self.pre_shuffle(x)
        _validate_nchw(
            "pre-shuffle x",
            x,
            expected_channels=self.config.output_head_channels * 4,
            expected_hw=self.config.input_size,
        )
        x = self.pixel_shuffle(x)
        x = self.output_conv(x)

        if x.shape[-2:] != self.config.state_size:
            raise RuntimeError(
                "Unexpected PI-HAF output spatial shape. "
                f"expected={self.config.state_size}, got={tuple(x.shape[-2:])}"
            )
        if x.shape[1] != self.config.out_channels:
            raise RuntimeError(
                "Unexpected PI-HAF output channel count. "
                f"expected={self.config.out_channels}, got={x.shape[1]}"
            )
        return x

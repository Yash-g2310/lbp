from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class PIHAFConfig:
    """Typed config schema for PI-HAF model instantiation."""

    in_channels: int = 1
    lr_channels: int = 1
    out_channels: int = 1
    input_size: Tuple[int, int] = (75, 75)

    hidden_channels: int = 96
    output_head_channels: int = 64

    time_embed_dim: int = 256
    time_mlp_dim: int = 256

    num_encoder_stages: int = 2
    num_bottleneck_sfin_blocks: int = 1
    num_bottleneck_rhag_blocks: int = 2
    num_decoder_stages: int = 2

    sfin_ffn_expansion: float = 2.0
    rhag_mlp_ratio: float = 2.0
    num_heads: int = 4
    window_size: int = 4
    dropout: float = 0.0

    cross_attention_stages: Tuple[int, ...] = field(default_factory=lambda: (1, 2, 3))
    pad_mode: str = "reflect"
    strict_76_mode: bool = True

    use_gradient_checkpointing: bool = True
    checkpoint_cross_attention: bool = False

    @property
    def total_stages(self) -> int:
        return self.num_encoder_stages + 1 + self.num_decoder_stages

    @property
    def state_size(self) -> Tuple[int, int]:
        return (self.input_size[0] * 2, self.input_size[1] * 2)

    def validate(self) -> None:
        if self.in_channels <= 0:
            raise ValueError(f"in_channels must be > 0, got {self.in_channels}")
        if self.lr_channels <= 0:
            raise ValueError(f"lr_channels must be > 0, got {self.lr_channels}")
        if self.out_channels <= 0:
            raise ValueError(f"out_channels must be > 0, got {self.out_channels}")

        if len(self.input_size) != 2:
            raise ValueError(f"input_size must be (H,W), got {self.input_size}")
        if any(dim <= 0 for dim in self.input_size):
            raise ValueError(f"input_size values must be > 0, got {self.input_size}")
        if self.input_size != (75, 75):
            raise ValueError(
                f"Current PI-HAF phase expects LR input_size=(75,75). got input_size={self.input_size}"
            )

        if self.hidden_channels <= 0 or self.hidden_channels % 2 != 0:
            raise ValueError(
                "hidden_channels must be positive and even for SFIN local/global split. "
                f"got {self.hidden_channels}"
            )
        if self.output_head_channels <= 0:
            raise ValueError(
                f"output_head_channels must be > 0, got {self.output_head_channels}"
            )

        if self.time_embed_dim <= 0 or self.time_mlp_dim <= 0:
            raise ValueError(
                f"time_embed_dim and time_mlp_dim must be > 0, got {self.time_embed_dim}, {self.time_mlp_dim}"
            )

        if self.num_encoder_stages <= 0:
            raise ValueError(
                f"num_encoder_stages must be > 0, got {self.num_encoder_stages}"
            )
        if self.num_bottleneck_sfin_blocks <= 0:
            raise ValueError(
                "num_bottleneck_sfin_blocks must be > 0, "
                f"got {self.num_bottleneck_sfin_blocks}"
            )
        if self.num_bottleneck_rhag_blocks <= 0:
            raise ValueError(
                "num_bottleneck_rhag_blocks must be > 0, "
                f"got {self.num_bottleneck_rhag_blocks}"
            )
        if self.num_decoder_stages <= 0:
            raise ValueError(
                f"num_decoder_stages must be > 0, got {self.num_decoder_stages}"
            )

        if self.num_heads <= 0:
            raise ValueError(f"num_heads must be > 0, got {self.num_heads}")
        if self.hidden_channels % self.num_heads != 0:
            raise ValueError(
                "hidden_channels must be divisible by num_heads. "
                f"got hidden_channels={self.hidden_channels}, num_heads={self.num_heads}"
            )

        if self.window_size <= 0:
            raise ValueError(f"window_size must be > 0, got {self.window_size}")
        if self.strict_76_mode and (76 % self.window_size) != 0:
            raise ValueError(
                "strict_76_mode=True requires window_size dividing 76 exactly. "
                f"got window_size={self.window_size}"
            )

        if self.pad_mode not in {"constant", "reflect", "replicate", "circular"}:
            raise ValueError(
                f"pad_mode must be one of constant|reflect|replicate|circular, got {self.pad_mode}"
            )

        if self.sfin_ffn_expansion <= 0.0:
            raise ValueError(
                f"sfin_ffn_expansion must be > 0, got {self.sfin_ffn_expansion}"
            )
        if self.rhag_mlp_ratio <= 0.0:
            raise ValueError(f"rhag_mlp_ratio must be > 0, got {self.rhag_mlp_ratio}")
        if not (0.0 <= self.dropout < 1.0):
            raise ValueError(f"dropout must be in [0,1), got {self.dropout}")

        for stage in self.cross_attention_stages:
            if stage < 0 or stage >= self.total_stages:
                raise ValueError(
                    f"cross_attention_stages contains invalid stage={stage}. "
                    f"Valid range is [0, {self.total_stages - 1}]"
                )

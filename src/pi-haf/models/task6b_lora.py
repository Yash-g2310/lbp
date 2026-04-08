from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .pi_haf.attention import FlowHAB
from .pi_haf.backbone import PIHAFBackbone
from .pi_haf.conditioning import CrossAttention2D


def _validate_rank3_tensor(name: str, tensor: torch.Tensor) -> None:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be torch.Tensor, got {type(tensor)}")
    if tensor.ndim != 3:
        raise ValueError(f"{name} must be rank-3 [B,L,E], got shape {tuple(tensor.shape)}")
    if not torch.is_floating_point(tensor):
        raise TypeError(f"{name} must be floating-point, got dtype {tensor.dtype}")
    if not torch.isfinite(tensor).all():
        raise ValueError(f"{name} contains NaN or Inf values")


def _validate_attention_shapes(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    embed_dim: int,
) -> Tuple[int, int, int]:
    _validate_rank3_tensor("query", query)
    _validate_rank3_tensor("key", key)
    _validate_rank3_tensor("value", value)

    batch_size, query_len, query_embed = query.shape
    key_batch, key_len, key_embed = key.shape
    value_batch, value_len, value_embed = value.shape

    if key_batch != batch_size or value_batch != batch_size:
        raise ValueError(
            "query/key/value batch mismatch: "
            f"query={batch_size}, key={key_batch}, value={value_batch}"
        )
    if key_len != value_len:
        raise ValueError(f"key/value sequence mismatch: key={key_len}, value={value_len}")
    if query_embed != embed_dim or key_embed != embed_dim or value_embed != embed_dim:
        raise ValueError(
            "query/key/value embed mismatch with wrapped attention embed_dim. "
            f"expected={embed_dim}, got query={query_embed}, key={key_embed}, value={value_embed}"
        )

    return batch_size, query_len, key_len


def _validate_finite_parameter(name: str, parameter: torch.Tensor) -> None:
    if not torch.isfinite(parameter).all():
        raise ValueError(f"Parameter {name} contains NaN or Inf values")


def _make_lora_parameters(
    rank: int,
    embed_dim: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[nn.Parameter, nn.Parameter]:
    lora_a = nn.Parameter(torch.empty((rank, embed_dim), device=device, dtype=dtype))
    lora_b = nn.Parameter(torch.empty((embed_dim, rank), device=device, dtype=dtype))

    # Stability rule: A random, B zeros so initial delta is exactly 0.
    nn.init.kaiming_uniform_(lora_a, a=math.sqrt(5.0))
    nn.init.zeros_(lora_b)

    _validate_finite_parameter("lora_a", lora_a)
    _validate_finite_parameter("lora_b", lora_b)
    return lora_a, lora_b


class LoRAMultiheadAttention(nn.Module):
    """Drop-in MultiheadAttention wrapper with LoRA on Query/Value projections.

    The wrapped base attention remains frozen. Low-rank adapters are injected only
    into Q and V projections with scaling alpha / rank.
    """

    def __init__(
        self,
        base_attn: nn.MultiheadAttention,
        rank: int = 16,
        alpha: int = 32,
        lora_dropout: float = 0.05,
        train_lora: bool = False,
    ):
        super().__init__()

        if not isinstance(base_attn, nn.MultiheadAttention):
            raise TypeError(f"base_attn must be nn.MultiheadAttention, got {type(base_attn)}")
        if not base_attn.batch_first:
            raise ValueError("LoRAMultiheadAttention requires base_attn.batch_first=True")
        if not base_attn._qkv_same_embed_dim:
            raise ValueError("LoRAMultiheadAttention currently supports qkv_same_embed_dim=True only")
        if base_attn.in_proj_weight is None:
            raise ValueError("base_attn.in_proj_weight must exist for QKV slicing")
        if rank <= 0:
            raise ValueError(f"rank must be > 0, got {rank}")
        if alpha <= 0:
            raise ValueError(f"alpha must be > 0, got {alpha}")
        if not (0.0 <= lora_dropout < 1.0):
            raise ValueError(f"lora_dropout must be in [0,1), got {lora_dropout}")

        self.base_attn = base_attn
        self.embed_dim = int(base_attn.embed_dim)
        self.num_heads = int(base_attn.num_heads)
        self.dropout = float(base_attn.dropout)
        self.batch_first = True

        if self.embed_dim % self.num_heads != 0:
            raise ValueError(
                "embed_dim must be divisible by num_heads. "
                f"got embed_dim={self.embed_dim}, num_heads={self.num_heads}"
            )

        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / float(self.rank)
        self.lora_dropout = nn.Dropout(p=float(lora_dropout))

        base_weight = self.base_attn.in_proj_weight
        assert base_weight is not None
        self.proj_device = base_weight.device
        self.proj_dtype = base_weight.dtype

        self.lora_q_a, self.lora_q_b = _make_lora_parameters(
            rank=self.rank,
            embed_dim=self.embed_dim,
            device=self.proj_device,
            dtype=self.proj_dtype,
        )
        self.lora_v_a, self.lora_v_b = _make_lora_parameters(
            rank=self.rank,
            embed_dim=self.embed_dim,
            device=self.proj_device,
            dtype=self.proj_dtype,
        )

        self.set_lora_trainable(train_lora)

        for parameter in self.base_attn.parameters():
            parameter.requires_grad = False

    def set_lora_trainable(self, enabled: bool) -> None:
        flag = bool(enabled)
        self.lora_q_a.requires_grad = flag
        self.lora_q_b.requires_grad = flag
        self.lora_v_a.requires_grad = flag
        self.lora_v_b.requires_grad = flag

    def _reshape_to_heads(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, _ = x.shape
        head_dim = self.embed_dim // self.num_heads
        x = x.view(bsz, seq_len, self.num_heads, head_dim)
        return x.transpose(1, 2).contiguous()

    def _reshape_from_heads(self, x: torch.Tensor) -> torch.Tensor:
        bsz, _, seq_len, head_dim = x.shape
        x = x.transpose(1, 2).contiguous()
        return x.view(bsz, seq_len, self.num_heads * head_dim)

    def _project_with_lora(
        self,
        x: torch.Tensor,
        base_weight: torch.Tensor,
        base_bias: Optional[torch.Tensor],
        lora_a: torch.Tensor,
        lora_b: torch.Tensor,
    ) -> torch.Tensor:
        base_out = F.linear(x, base_weight, base_bias)
        lora_hidden = F.linear(self.lora_dropout(x), lora_a)
        lora_delta = F.linear(lora_hidden, lora_b)
        return base_out + (self.scaling * lora_delta)

    def _build_attn_bias(
        self,
        batch_size: int,
        query_len: int,
        key_len: int,
        dtype: torch.dtype,
        device: torch.device,
        key_padding_mask: Optional[torch.Tensor],
        attn_mask: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        neg_inf = torch.tensor(torch.finfo(dtype).min, dtype=dtype, device=device)
        bias: Optional[torch.Tensor] = None

        if attn_mask is not None:
            if not isinstance(attn_mask, torch.Tensor):
                raise TypeError(f"attn_mask must be torch.Tensor, got {type(attn_mask)}")

            if attn_mask.ndim == 2:
                if attn_mask.shape != (query_len, key_len):
                    raise ValueError(
                        "2D attn_mask must have shape [L,S], "
                        f"got {tuple(attn_mask.shape)} for L={query_len}, S={key_len}"
                    )
                expanded = attn_mask.unsqueeze(0).unsqueeze(0).expand(
                    batch_size,
                    self.num_heads,
                    query_len,
                    key_len,
                )
            elif attn_mask.ndim == 3:
                if attn_mask.shape[1:] != (query_len, key_len):
                    raise ValueError(
                        "3D attn_mask must have trailing shape [L,S], "
                        f"got {tuple(attn_mask.shape)}"
                    )
                if attn_mask.shape[0] == batch_size * self.num_heads:
                    expanded = attn_mask.view(batch_size, self.num_heads, query_len, key_len)
                elif attn_mask.shape[0] == batch_size:
                    expanded = attn_mask.unsqueeze(1).expand(
                        batch_size,
                        self.num_heads,
                        query_len,
                        key_len,
                    )
                else:
                    raise ValueError(
                        "3D attn_mask first dimension must be B or B*H, "
                        f"got {attn_mask.shape[0]} with B={batch_size}, H={self.num_heads}"
                    )
            else:
                raise ValueError(f"attn_mask must be 2D or 3D, got ndim={attn_mask.ndim}")

            expanded = expanded.to(device=device)
            if expanded.dtype == torch.bool:
                float_mask = torch.zeros_like(expanded, dtype=dtype, device=device)
                float_mask = float_mask.masked_fill(expanded, neg_inf)
                bias = float_mask
            elif torch.is_floating_point(expanded):
                bias = expanded.to(dtype=dtype)
            else:
                raise TypeError(f"attn_mask must be bool or floating tensor, got {expanded.dtype}")

        if key_padding_mask is not None:
            if not isinstance(key_padding_mask, torch.Tensor):
                raise TypeError(
                    f"key_padding_mask must be torch.Tensor, got {type(key_padding_mask)}"
                )
            if key_padding_mask.ndim != 2:
                raise ValueError(
                    f"key_padding_mask must be rank-2 [B,S], got shape {tuple(key_padding_mask.shape)}"
                )
            if key_padding_mask.shape != (batch_size, key_len):
                raise ValueError(
                    "key_padding_mask shape mismatch, "
                    f"expected {(batch_size, key_len)}, got {tuple(key_padding_mask.shape)}"
                )
            if key_padding_mask.dtype != torch.bool:
                raise TypeError(
                    f"key_padding_mask must be bool tensor per MHA contract, got {key_padding_mask.dtype}"
                )

            pad_mask = key_padding_mask.view(batch_size, 1, 1, key_len).expand(
                batch_size,
                self.num_heads,
                query_len,
                key_len,
            )
            pad_bias = torch.zeros((batch_size, self.num_heads, query_len, key_len), dtype=dtype, device=device)
            pad_bias = pad_bias.masked_fill(pad_mask.to(device=device), neg_inf)
            bias = pad_bias if bias is None else (bias + pad_bias)

        return bias

    def _compute_attention_weights(
        self,
        q_heads: torch.Tensor,
        k_heads: torch.Tensor,
        attn_bias: Optional[torch.Tensor],
        is_causal: bool,
        average_attn_weights: bool,
    ) -> torch.Tensor:
        head_dim = self.embed_dim // self.num_heads
        scale = 1.0 / math.sqrt(float(head_dim))

        scores = torch.matmul(q_heads, k_heads.transpose(-2, -1)) * scale
        if attn_bias is not None:
            scores = scores + attn_bias

        if is_causal:
            query_len = q_heads.shape[-2]
            key_len = k_heads.shape[-2]
            causal_mask = torch.ones((query_len, key_len), dtype=torch.bool, device=q_heads.device).triu(1)
            neg_inf = torch.tensor(torch.finfo(scores.dtype).min, dtype=scores.dtype, device=scores.device)
            scores = scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), neg_inf)

        attn_probs = torch.softmax(scores.float(), dim=-1).to(dtype=scores.dtype)
        if average_attn_weights:
            return attn_probs.mean(dim=1)
        return attn_probs

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        need_weights: bool = True,
        attn_mask: Optional[torch.Tensor] = None,
        average_attn_weights: bool = True,
        is_causal: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        base_weight = self.base_attn.in_proj_weight
        if base_weight is None:
            raise RuntimeError("Wrapped base_attn lost in_proj_weight; cannot run LoRA attention")

        batch_size, query_len, key_len = _validate_attention_shapes(
            query=query,
            key=key,
            value=value,
            embed_dim=self.embed_dim,
        )

        in_proj_bias = self.base_attn.in_proj_bias

        w_q = base_weight[: self.embed_dim, :]
        w_k = base_weight[self.embed_dim : 2 * self.embed_dim, :]
        w_v = base_weight[2 * self.embed_dim :, :]

        b_q: Optional[torch.Tensor] = None
        b_k: Optional[torch.Tensor] = None
        b_v: Optional[torch.Tensor] = None
        if in_proj_bias is not None:
            b_q = in_proj_bias[: self.embed_dim]
            b_k = in_proj_bias[self.embed_dim : 2 * self.embed_dim]
            b_v = in_proj_bias[2 * self.embed_dim :]

        q_proj = self._project_with_lora(query, w_q, b_q, self.lora_q_a, self.lora_q_b)
        k_proj = F.linear(key, w_k, b_k)
        v_proj = self._project_with_lora(value, w_v, b_v, self.lora_v_a, self.lora_v_b)

        _validate_rank3_tensor("q_proj", q_proj)
        _validate_rank3_tensor("k_proj", k_proj)
        _validate_rank3_tensor("v_proj", v_proj)

        q_heads = self._reshape_to_heads(q_proj)
        k_heads = self._reshape_to_heads(k_proj)
        v_heads = self._reshape_to_heads(v_proj)

        attn_bias = self._build_attn_bias(
            batch_size=batch_size,
            query_len=query_len,
            key_len=key_len,
            dtype=q_heads.dtype,
            device=q_heads.device,
            key_padding_mask=key_padding_mask,
            attn_mask=attn_mask,
        )

        dropout_p = self.dropout if self.training else 0.0
        attn_out_heads = F.scaled_dot_product_attention(
            q_heads,
            k_heads,
            v_heads,
            attn_mask=attn_bias,
            dropout_p=dropout_p,
            is_causal=is_causal,
        )

        attn_out = self._reshape_from_heads(attn_out_heads)
        out_proj = self.base_attn.out_proj
        attn_output = F.linear(attn_out, out_proj.weight, out_proj.bias)

        if not torch.isfinite(attn_output).all():
            raise ValueError("LoRAMultiheadAttention produced NaN or Inf output")

        if need_weights:
            attn_weights = self._compute_attention_weights(
                q_heads=q_heads,
                k_heads=k_heads,
                attn_bias=attn_bias,
                is_causal=is_causal,
                average_attn_weights=average_attn_weights,
            )
            return attn_output, attn_weights

        return attn_output, None


def set_lora_trainable(model: nn.Module, enabled: bool) -> None:
    """Enable or disable LoRA adapter training across a model tree."""
    for module in model.modules():
        if isinstance(module, LoRAMultiheadAttention):
            module.set_lora_trainable(enabled)


def prepare_task6b_model(
    model: PIHAFBackbone,
    rank: int = 16,
    alpha: int = 32,
    lora_dropout: float = 0.05,
) -> PIHAFBackbone:
    """Freeze PI-HAF, inject LoRA wrappers into attention, unfreeze output head.

    Stage-1 policy: output head trainable, LoRA injected but frozen.
    """
    if not isinstance(model, PIHAFBackbone):
        raise TypeError(f"prepare_task6b_model expects PIHAFBackbone, got {type(model)}")

    if not hasattr(model, "output_conv"):
        raise ValueError("PIHAFBackbone is missing required output_conv")
    if not isinstance(model.output_conv, nn.Conv2d):
        raise ValueError(f"output_conv must be nn.Conv2d, got {type(model.output_conv)}")
    if model.output_conv.weight is None:
        raise ValueError("output_conv.weight is missing")

    for parameter in model.parameters():
        parameter.requires_grad = False

    replaced_count = 0
    for module in model.modules():
        if isinstance(module, (FlowHAB, CrossAttention2D)):
            if not hasattr(module, "attn"):
                raise ValueError(f"Target module {module.__class__.__name__} has no 'attn' attribute")

            attn_module = module.attn
            if isinstance(attn_module, LoRAMultiheadAttention):
                continue
            if not isinstance(attn_module, nn.MultiheadAttention):
                raise TypeError(
                    f"Expected nn.MultiheadAttention in {module.__class__.__name__}.attn, got {type(attn_module)}"
                )

            wrapped_attn = LoRAMultiheadAttention(
                base_attn=attn_module,
                rank=rank,
                alpha=alpha,
                lora_dropout=lora_dropout,
                train_lora=False,
            )
            setattr(module, "attn", wrapped_attn)
            replaced_count += 1

    if replaced_count == 0:
        raise RuntimeError("No attention modules were replaced with LoRA wrappers")

    model.output_conv.weight.requires_grad = True
    if model.output_conv.bias is not None:
        model.output_conv.bias.requires_grad = True

    return model


def print_trainable_parameters(model: nn.Module) -> None:
    """Print clear frozen/trainable parameter statistics."""
    total_params = 0
    trainable_params = 0

    for parameter in model.parameters():
        count = int(parameter.numel())
        total_params += count
        if parameter.requires_grad:
            trainable_params += count

    if total_params <= 0:
        raise ValueError("Model has zero parameters; cannot summarize trainability")

    frozen_params = total_params - trainable_params
    trainable_pct = 100.0 * (float(trainable_params) / float(total_params))
    frozen_pct = 100.0 * (float(frozen_params) / float(total_params))

    print(f"Total parameters:      {total_params:,}")
    print(f"Trainable parameters:  {trainable_params:,} ({trainable_pct:.2f}%)")
    print(f"Frozen parameters:     {frozen_params:,} ({frozen_pct:.2f}%)")


__all__ = [
    "LoRAMultiheadAttention",
    "prepare_task6b_model",
    "print_trainable_parameters",
    "set_lora_trainable",
]

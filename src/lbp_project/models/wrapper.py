"""
Model Wrapper
Combines a frozen DINO backbone with the dynamic SFIN-U-Net decoder.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Import our custom dynamic U-Net
from .backbone_loader import FrozenBackbone, load_frozen_backbone
from .backbone_policy import (
    BackboneSpec,
    collect_backbone_candidates,
    is_backbone_fallback_approved,
    resolve_backbone_spec,
    should_stop_on_primary_backbone_failure,
)
from .conditioning import SinusoidalTimeEmbedding
from .unet import STEMEnhancementNet

class DINOSFIN_Architecture_NEW(nn.Module):
    def __init__(self, 
                 strategy="concat", 
                 base_channels=64, 
                 num_sfin=2, 
                 num_rhag=3, 
                 window_size=7,
                 dino_embed_dim=768,
                 backbone_repo="timm",
                 backbone_model="timm/convnext_small.dinov3_lvd1689m",
                 backbone_backend="",
                 backbone_fallback_models=None,
                 backbone_stop_on_failure=True,
                 backbone_fallback_approved=False,
                 max_layer_id=8,
                 enable_velocity_head=False,
                 velocity_hidden_channels=64,
                 fft_mode="fp32",
                 fft_pad_size=256,
                 adaln_zero_enabled=False,
                 adaln_layer_embed_dim=64,
                 adaln_time_embed_dim=64,
                 adaln_condition_dim=128,
                 adaln_timestep_default=1.0,
                 use_precomputed_dino=False):
        """
        Args:
            strategy: 'concat' (Layered Depth prompting strategy)
            base_channels: Scales U-Net size (e.g., 32 for laptop, 64 for server)
            num_sfin: SFIN blocks per level
            num_rhag: RHAG blocks in the bottleneck
            window_size: 7 (works for standard 224px ViT feature grids)
        """
        super().__init__()
        self.strategy = strategy
        self.use_precomputed_dino = use_precomputed_dino
        self.dino_embed_dim = int(dino_embed_dim)
        self.backbone_repo = str(backbone_repo)
        self.backbone_model = str(backbone_model)
        self.backbone_backend = str(backbone_backend)
        self.max_layer_id = int(max_layer_id)
        self.enable_velocity_head = bool(enable_velocity_head)
        self.adaln_zero_enabled = bool(adaln_zero_enabled)
        self.adaln_timestep_default = float(adaln_timestep_default)
        if self.max_layer_id < 1:
            raise ValueError(f"max_layer_id must be >= 1, got {self.max_layer_id}")
        if self.adaln_timestep_default < 0.0 or self.adaln_timestep_default > 1.0:
            raise ValueError(
                f"adaln_timestep_default must be in [0,1], got {self.adaln_timestep_default}"
            )

        arch_cfg = {
            "backbone_repo": self.backbone_repo,
            "backbone_model": self.backbone_model,
            "backbone_backend": self.backbone_backend,
            "backbone_fallback_models": backbone_fallback_models or [],
            "backbone_stop_on_failure": bool(backbone_stop_on_failure),
            "backbone_fallback_approved": bool(backbone_fallback_approved),
            "dino_embed_dim": self.dino_embed_dim,
        }
        self.backbone_spec: BackboneSpec = resolve_backbone_spec(arch_cfg)
        self.backbone_descriptor = self.backbone_spec.descriptor
        
        # 1. Initialize frozen foundational backbone (DINO family).
        self.encoder: FrozenBackbone | None = None
        if not self.use_precomputed_dino:
            candidates = collect_backbone_candidates(arch_cfg)
            stop_on_primary_failure = should_stop_on_primary_backbone_failure(arch_cfg)
            fallback_approved = is_backbone_fallback_approved(arch_cfg)
            if stop_on_primary_failure and candidates:
                candidates = candidates[:1]
            fallback_blocked = False
            if (not stop_on_primary_failure) and len(candidates) > 1 and (not fallback_approved):
                # Keep strict explicit-approval behavior: no silent fallback attempts.
                candidates = candidates[:1]
                fallback_blocked = True

            load_errors: list[str] = []
            for candidate in candidates:
                try:
                    self.encoder = load_frozen_backbone(candidate)
                    self.backbone_spec = candidate
                    self.backbone_descriptor = candidate.descriptor
                    break
                except Exception as exc:
                    load_errors.append(f"{candidate.descriptor}: {exc}")

            if self.encoder is None:
                details = "\n".join(f"- {msg}" for msg in load_errors) if load_errors else "- no candidates attempted"
                fallback_note = (
                    "\nFallback candidates are configured but blocked because "
                    "architecture.backbone_fallback_approved=false. "
                    "Set architecture.backbone_fallback_approved=true for an explicit one-off fallback attempt."
                    if fallback_blocked
                    else ""
                )
                raise RuntimeError(
                    "Backbone initialization failed. "
                    f"Primary descriptor: {self.backbone_spec.descriptor}.\n"
                    f"Attempted loaders:\n{details}\n"
                    "To keep strict stop-on-failure behavior, leave architecture.backbone_stop_on_failure=true. "
                    "To allow explicit fallback attempts, set architecture.backbone_stop_on_failure=false "
                    "and provide architecture.backbone_fallback_models."
                    f"{fallback_note}"
                )
            
        # 2. Calculate Input Channels for Decoder
        # DINO gives C. RGB gives 3. Prompt gives 1. Total = C + 4.
        in_c = self.dino_embed_dim + 3 + 1 
        
        # 3. Prompt Projection
        # We keep a scalar prompt map but learn it from categorical layer IDs.
        self.layer_embed = nn.Embedding(self.max_layer_id + 1, 1)
        nn.init.zeros_(self.layer_embed.weight)

        self.adaln_layer_embed: nn.Embedding | None = None
        self.adaln_time_embed: SinusoidalTimeEmbedding | None = None
        self.adaln_condition_proj: nn.Module | None = None
        self.adaln_condition_dim = int(adaln_condition_dim)
        if self.adaln_zero_enabled:
            layer_dim = int(adaln_layer_embed_dim)
            time_dim = int(adaln_time_embed_dim)
            if layer_dim < 1:
                raise ValueError(f"adaln_layer_embed_dim must be >= 1, got {layer_dim}")
            if time_dim < 1:
                raise ValueError(f"adaln_time_embed_dim must be >= 1, got {time_dim}")
            if self.adaln_condition_dim < 1:
                raise ValueError(
                    f"adaln_condition_dim must be >= 1, got {self.adaln_condition_dim}"
                )

            self.adaln_layer_embed = nn.Embedding(self.max_layer_id + 1, layer_dim)
            nn.init.normal_(self.adaln_layer_embed.weight, mean=0.0, std=0.02)
            self.adaln_time_embed = SinusoidalTimeEmbedding(embed_dim=time_dim)
            self.adaln_condition_proj = nn.Sequential(
                nn.Linear(layer_dim + time_dim, self.adaln_condition_dim),
                nn.SiLU(),
                nn.Linear(self.adaln_condition_dim, self.adaln_condition_dim),
            )
        
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
            adaln_zero_enabled=self.adaln_zero_enabled,
            adaln_condition_dim=self.adaln_condition_dim,
        )

        self.velocity_head: nn.Module | None = None
        if self.enable_velocity_head:
            hidden_c = int(velocity_hidden_channels)
            if hidden_c < 1:
                raise ValueError(
                    "velocity_hidden_channels must be >= 1 when velocity head is enabled, "
                    f"got {hidden_c}"
                )
            decoder_feat_c = int(self.decoder.channels[0])
            self.velocity_head = nn.Sequential(
                nn.Conv2d(decoder_feat_c + 2, hidden_c, kernel_size=3, stride=1, padding=1),
                nn.GELU(),
                nn.Conv2d(hidden_c, 1, kernel_size=1, stride=1, padding=0),
            )

    def _extract_dino_features(self, x):
        if self.encoder is None:
            raise RuntimeError("DINO encoder is disabled. Provide precomputed features in forward().")

        with torch.no_grad():
            dino_feats = self.encoder.forward_features(x)
        if dino_feats.ndim != 4:
            raise RuntimeError(
                "Backbone adapter must return [B,C,H,W], got shape={} for {}".format(
                    tuple(dino_feats.shape),
                    self.backbone_descriptor,
                )
            )

        B, C, H, W = dino_feats.shape
        if C != self.dino_embed_dim:
            raise RuntimeError(
                "Backbone feature dim mismatch. "
                f"Expected {self.dino_embed_dim}, got {C} from {self.backbone_descriptor}"
            )
        return F.interpolate(dino_feats, size=x.shape[2:], mode='bilinear', align_corners=False)

    def _normalize_target_layer(self, target_layer, batch_size: int, device: torch.device) -> torch.Tensor:
        if torch.is_tensor(target_layer):
            layers = target_layer.to(device=device)
        else:
            layers = torch.as_tensor(target_layer, device=device)

        if layers.ndim == 0:
            layers = layers.view(1).expand(batch_size)
        elif layers.ndim == 1 and layers.shape[0] == 1:
            layers = layers.expand(batch_size)
        elif layers.ndim == 2 and layers.shape[1] == 1 and layers.shape[0] == batch_size:
            layers = layers.squeeze(1)
        elif layers.ndim != 1:
            raise ValueError(
                "target_layer must be scalar, [B], or [B,1]. "
                f"Got shape={tuple(layers.shape)}"
            )

        if layers.shape[0] != batch_size:
            raise ValueError(
                f"target_layer batch mismatch: expected {batch_size}, got {layers.shape[0]}"
            )

        layers = layers.to(dtype=torch.float32)
        if not torch.isfinite(layers).all():
            raise ValueError("target_layer contains non-finite values")

        rounded = layers.round()
        if not torch.allclose(layers, rounded):
            raise ValueError(
                "target_layer must contain integer IDs only. "
                f"Received values: {layers.detach().cpu().tolist()}"
            )
        layers = rounded.to(dtype=torch.long)

        if int(layers.min().item()) < 1 or int(layers.max().item()) > self.max_layer_id:
            raise ValueError(
                "target_layer out of supported range. "
                f"Expected IDs in [1, {self.max_layer_id}], got "
                f"min={int(layers.min().item())}, max={int(layers.max().item())}"
            )
        return layers

    def _normalize_flow_timestep(
        self,
        flow_t,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if torch.is_tensor(flow_t):
            t = flow_t.to(device=device)
        else:
            t = torch.as_tensor(flow_t, device=device)

        if t.ndim == 0:
            t = t.view(1).expand(batch_size)
        if t.ndim == 2 and t.shape[1] == 1:
            t = t.squeeze(1)
        if t.ndim != 1 or t.shape[0] != batch_size:
            raise ValueError(f"flow_t must have shape [B] or [B,1], got {tuple(t.shape)}")

        t = t.to(dtype=dtype)
        if not torch.isfinite(t).all():
            raise ValueError("flow_t contains non-finite values")
        if ((t < 0.0) | (t > 1.0)).any():
            raise ValueError("flow_t must be in [0, 1]")
        return t

    def _resolve_adaln_condition(
        self,
        layer_ids: torch.Tensor,
        flow_t,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor | None:
        if not self.adaln_zero_enabled:
            return None
        if self.adaln_layer_embed is None or self.adaln_time_embed is None or self.adaln_condition_proj is None:
            raise RuntimeError("AdaLN-Zero is enabled but conditioning modules are missing")

        if flow_t is None:
            t_vec = torch.full(
                (batch_size,),
                fill_value=self.adaln_timestep_default,
                device=device,
                dtype=dtype,
            )
        else:
            t_vec = self._normalize_flow_timestep(
                flow_t,
                batch_size=batch_size,
                device=device,
                dtype=dtype,
            )

        layer_vec = self.adaln_layer_embed(layer_ids)
        time_vec = self.adaln_time_embed(t_vec)
        cond = self.adaln_condition_proj(torch.cat([layer_vec, time_vec.to(layer_vec.dtype)], dim=1))
        return cond.to(dtype=dtype)

    def forward(
        self,
        x,
        target_layer=1,
        return_intermediate=False,
        use_checkpointing=False,
        precomputed_dino=None,
        flow_noisy_depth=None,
        flow_t=None,
        return_velocity=False,
    ):
        # 1. Extract DINO Features
        if precomputed_dino is not None:
            dino_upsampled = precomputed_dino
            if dino_upsampled.shape[-2:] != x.shape[-2:]:
                dino_upsampled = F.interpolate(dino_upsampled, size=x.shape[2:], mode='bilinear', align_corners=False)
        else:
            dino_upsampled = self._extract_dino_features(x)

        if dino_upsampled.shape[1] != self.dino_embed_dim:
            raise RuntimeError(
                "Precomputed DINO feature dim mismatch. "
                f"Expected {self.dino_embed_dim}, got {dino_upsampled.shape[1]}"
            )

        B = x.shape[0]
        
        # 2. Process Layer Prompt
        layer_ids = self._normalize_target_layer(target_layer, batch_size=B, device=x.device)
        p_vec = self.layer_embed(layer_ids).view(B, 1, 1, 1).expand(-1, -1, x.shape[2], x.shape[3])
        
        # 3. Early Fusion Concatenation
        fused_input = torch.cat([x, dino_upsampled, p_vec], dim=1) 
        adaln_condition = self._resolve_adaln_condition(
            layer_ids,
            flow_t=flow_t,
            batch_size=B,
            device=x.device,
            dtype=fused_input.dtype,
        )

        if return_velocity and (not self.enable_velocity_head or self.velocity_head is None):
            raise RuntimeError(
                "Velocity prediction was requested but architecture.enable_velocity_head=false. "
                "Enable the velocity head in config when using flow training."
            )

        need_intermediate = bool(return_intermediate or return_velocity)

        # 4. Decoder Pass
        if need_intermediate:
            out_dict = self.decoder(
                fused_input,
                return_intermediate=True,
                use_checkpointing=use_checkpointing,
                conditioning=adaln_condition,
            )
            if "decoder_features" not in out_dict:
                raise RuntimeError("Decoder did not return 'decoder_features' required for velocity head")

            velocity_pred = None
            if return_velocity:
                if flow_noisy_depth is None or flow_t is None:
                    raise ValueError("flow_noisy_depth and flow_t are required when return_velocity=true")
                if not torch.is_tensor(flow_noisy_depth):
                    flow_noisy_depth = torch.as_tensor(flow_noisy_depth, device=x.device)
                if flow_noisy_depth.ndim != 4 or flow_noisy_depth.shape[0] != B:
                    raise ValueError(
                        "flow_noisy_depth must have shape [B,1,H,W] (or broadcastable spatially), "
                        f"got {tuple(flow_noisy_depth.shape)}"
                    )
                if flow_noisy_depth.shape[1] != 1:
                    raise ValueError(
                        "flow_noisy_depth must have channel size 1, "
                        f"got {flow_noisy_depth.shape[1]}"
                    )

                decoder_features = out_dict["decoder_features"]
                flow_depth = flow_noisy_depth.to(device=x.device, dtype=decoder_features.dtype)
                if flow_depth.shape[-2:] != x.shape[-2:]:
                    flow_depth = F.interpolate(
                        flow_depth,
                        size=x.shape[-2:],
                        mode="bilinear",
                        align_corners=False,
                    )

                t_vec = self._normalize_flow_timestep(
                    flow_t,
                    batch_size=B,
                    device=x.device,
                    dtype=decoder_features.dtype,
                )
                t_map = t_vec.view(B, 1, 1, 1).expand(-1, -1, x.shape[2], x.shape[3])

                velocity_in = torch.cat([decoder_features, flow_depth, t_map], dim=1)
                velocity_pred = self.velocity_head(velocity_in)
                if not torch.isfinite(velocity_pred).all():
                    raise RuntimeError("Velocity head produced non-finite outputs")

            if return_intermediate:
                payload = {
                    "bottleneck": F.softplus(out_dict["bottleneck"]),
                    "decoder": F.softplus(out_dict["decoder"]),
                    "final": F.softplus(out_dict["final"]),
                }
                if velocity_pred is not None:
                    payload["velocity"] = velocity_pred
                return payload

            # Velocity-only path for flow objective calls.
            if velocity_pred is None:
                raise RuntimeError("Internal error: return_velocity path missing velocity prediction")
            return velocity_pred

        # Standard Prediction Routing
        return F.softplus(
            self.decoder(
                fused_input,
                return_intermediate=False,
                use_checkpointing=use_checkpointing,
                conditioning=adaln_condition,
            )
        )
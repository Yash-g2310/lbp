"""Config schema and semantic validation helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, Tuple, Type

from lbp_project.models.backbone_policy import collect_backbone_candidates


def _path_label(source: str | Path | None) -> str:
    if source is None:
        return "<config>"
    return str(source)


def _deep_get(payload: Dict[str, Any], dotted_key: str) -> Any:
    cur: Any = payload
    for part in dotted_key.split("."):
        if not isinstance(cur, dict) or part not in cur:
            raise KeyError(dotted_key)
        cur = cur[part]
    return cur


def _validate_type(value: Any, expected: Type[Any] | Tuple[Type[Any], ...]) -> bool:
    expected_types = expected if isinstance(expected, tuple) else (expected,)

    if int in expected_types and isinstance(value, bool):
        return False
    if float in expected_types and isinstance(value, bool):
        return False

    for tp in expected_types:
        if tp is float and isinstance(value, (int, float)) and not isinstance(value, bool):
            return True
        if isinstance(value, tp):
            return True
    return False


def _require_keys(cfg: Dict[str, Any], requirements: Iterable[Tuple[str, Type[Any] | Tuple[Type[Any], ...]]], source: str | Path | None) -> None:
    label = _path_label(source)
    for key, expected in requirements:
        try:
            value = _deep_get(cfg, key)
        except KeyError as exc:
            raise ValueError(f"{label}: missing required config key '{key}'") from exc

        if not _validate_type(value, expected):
            exp = expected if isinstance(expected, tuple) else (expected,)
            exp_name = "/".join(t.__name__ for t in exp)
            raise ValueError(
                f"{label}: key '{key}' expected type {exp_name}, got {type(value).__name__}"
            )


def _ensure_range(value: float, key: str, low: float, high: float, source: str | Path | None) -> None:
    if value < low or value > high:
        raise ValueError(f"{_path_label(source)}: '{key}' must be in [{low}, {high}], got {value}")


def _ensure_positive(value: float, key: str, source: str | Path | None) -> None:
    if value <= 0:
        raise ValueError(f"{_path_label(source)}: '{key}' must be > 0, got {value}")


def validate_config_dict(cfg: Dict[str, Any], source: str | Path | None = None) -> Dict[str, Any]:
    if not isinstance(cfg, dict):
        raise ValueError(f"{_path_label(source)}: config root must be a mapping")

    required_sections = ("experiment", "hardware", "data", "architecture", "training", "logging")
    for section in required_sections:
        if section not in cfg or not isinstance(cfg[section], dict):
            raise ValueError(f"{_path_label(source)}: missing required section '{section}'")

    _require_keys(
        cfg,
        (
            ("experiment.name", str),
            ("experiment.seed", int),
            ("hardware.device", str),
            ("hardware.num_workers", int),
            ("data.train_dataset_name", str),
            ("data.train_split", str),
            ("data.val_dataset_name", str),
            ("data.val_split", str),
            ("data.cache_dir", str),
            ("data.batch_size", int),
            ("data.val_batch_size", int),
            ("data.use_precomputed_dino", bool),
            ("architecture.strategy", str),
            ("architecture.base_channels", int),
            ("architecture.num_sfin", int),
            ("architecture.num_rhag", int),
            ("architecture.window_size", int),
            ("architecture.fft_mode", str),
            ("architecture.fft_pad_size", int),
            ("architecture.dino_embed_dim", int),
            ("training.epochs", int),
            ("training.accum_steps", int),
            ("training.learning_rate", (int, float)),
            ("training.weight_decay", (int, float)),
            ("training.grad_clip_norm", (int, float)),
            ("training.scheduler.name", str),
            ("training.checkpoint.dir", str),
            ("training.checkpoint.latest_name", str),
            ("training.checkpoint.best_name", str),
            ("logging.use_wandb", bool),
            ("logging.log_to_terminal", bool),
        ),
        source,
    )

    device = str(cfg["hardware"]["device"]).lower()
    if device not in {"cpu", "cuda", "mps"}:
        raise ValueError(f"{_path_label(source)}: hardware.device must be one of cpu/cuda/mps, got '{device}'")

    _ensure_positive(float(cfg["training"]["epochs"]), "training.epochs", source)
    _ensure_positive(float(cfg["training"]["accum_steps"]), "training.accum_steps", source)
    _ensure_positive(float(cfg["data"]["batch_size"]), "data.batch_size", source)
    _ensure_positive(float(cfg["data"]["val_batch_size"]), "data.val_batch_size", source)
    _ensure_positive(float(cfg["training"]["learning_rate"]), "training.learning_rate", source)
    _ensure_positive(float(cfg["training"]["grad_clip_norm"]), "training.grad_clip_norm", source)
    _ensure_positive(float(cfg["architecture"]["base_channels"]), "architecture.base_channels", source)
    _ensure_positive(float(cfg["architecture"]["num_sfin"]), "architecture.num_sfin", source)
    _ensure_positive(float(cfg["architecture"]["num_rhag"]), "architecture.num_rhag", source)
    _ensure_positive(float(cfg["architecture"]["window_size"]), "architecture.window_size", source)
    _ensure_positive(float(cfg["architecture"]["dino_embed_dim"]), "architecture.dino_embed_dim", source)

    if "enable_velocity_head" in cfg["architecture"] and not isinstance(
        cfg["architecture"]["enable_velocity_head"], bool
    ):
        raise ValueError(
            f"{_path_label(source)}: architecture.enable_velocity_head must be a boolean"
        )
    if "velocity_hidden_channels" in cfg["architecture"]:
        hidden = int(cfg["architecture"]["velocity_hidden_channels"])
        if hidden < 1:
            raise ValueError(
                f"{_path_label(source)}: architecture.velocity_hidden_channels must be >= 1"
            )
    if "backbone_fallback_approved" in cfg["architecture"] and not isinstance(
        cfg["architecture"].get("backbone_fallback_approved"), bool
    ):
        raise ValueError(
            f"{_path_label(source)}: architecture.backbone_fallback_approved must be a boolean"
        )

    adaln_cfg = cfg["architecture"].get("adaln_zero", {})
    if adaln_cfg:
        if not isinstance(adaln_cfg, dict):
            raise ValueError(f"{_path_label(source)}: architecture.adaln_zero must be a mapping")
        if "enabled" in adaln_cfg and not isinstance(adaln_cfg.get("enabled"), bool):
            raise ValueError(f"{_path_label(source)}: architecture.adaln_zero.enabled must be a boolean")

        for key in ("layer_embed_dim", "time_embed_dim", "condition_dim"):
            if key in adaln_cfg and int(adaln_cfg.get(key, 0)) < 1:
                raise ValueError(
                    f"{_path_label(source)}: architecture.adaln_zero.{key} must be >= 1"
                )

        if "timestep_default" in adaln_cfg:
            _ensure_range(
                float(adaln_cfg.get("timestep_default", 1.0)),
                "architecture.adaln_zero.timestep_default",
                0.0,
                1.0,
                source,
            )

    num_workers = int(cfg["hardware"]["num_workers"])
    if num_workers < 0:
        raise ValueError(f"{_path_label(source)}: hardware.num_workers must be >= 0")

    if "target_gpu_class" in cfg["hardware"] and not isinstance(
        cfg["hardware"].get("target_gpu_class"), str
    ):
        raise ValueError(
            f"{_path_label(source)}: hardware.target_gpu_class must be a string"
        )

    if "min_vram_gb" in cfg["hardware"]:
        min_vram_gb = float(cfg["hardware"].get("min_vram_gb", 0.0))
        if min_vram_gb < 0.0:
            raise ValueError(
                f"{_path_label(source)}: hardware.min_vram_gb must be >= 0"
            )

    if "fallback_cache_dir" in cfg["data"] and not isinstance(
        cfg["data"].get("fallback_cache_dir"), str
    ):
        raise ValueError(
            f"{_path_label(source)}: data.fallback_cache_dir must be a string when provided"
        )

    if "allow_hf_downloads" in cfg["data"] and not isinstance(
        cfg["data"].get("allow_hf_downloads"), bool
    ):
        raise ValueError(
            f"{_path_label(source)}: data.allow_hf_downloads must be a boolean when provided"
        )

    if "allow_partial_local_shards" in cfg["data"] and not isinstance(
        cfg["data"].get("allow_partial_local_shards"), bool
    ):
        raise ValueError(
            f"{_path_label(source)}: data.allow_partial_local_shards must be a boolean when provided"
        )

    if "repair_hf_cache_once" in cfg["data"] and not isinstance(
        cfg["data"].get("repair_hf_cache_once"), bool
    ):
        raise ValueError(
            f"{_path_label(source)}: data.repair_hf_cache_once must be a boolean when provided"
        )

    if "partial_local_shards_min_per_split" in cfg["data"]:
        min_shards = int(cfg["data"].get("partial_local_shards_min_per_split", 0))
        if min_shards < 1:
            raise ValueError(
                f"{_path_label(source)}: data.partial_local_shards_min_per_split must be >= 1"
            )

    if cfg["data"]["use_precomputed_dino"]:
        idx_path = str(cfg["data"].get("precomputed_index_path", "")).strip()
        if not idx_path:
            raise ValueError(
                f"{_path_label(source)}: data.precomputed_index_path must be set when data.use_precomputed_dino=true"
            )

    max_layers = int(cfg["data"].get("max_layers", 0))
    max_layer_id = int(cfg["architecture"].get("max_layer_id", max(1, max_layers) if max_layers > 0 else 8))
    if max_layer_id < 1:
        raise ValueError(f"{_path_label(source)}: architecture.max_layer_id must be >= 1")
    if max_layers > 0 and max_layer_id < max_layers:
        raise ValueError(
            f"{_path_label(source)}: architecture.max_layer_id ({max_layer_id}) must be >= data.max_layers ({max_layers})"
        )

    curriculum_cfg = cfg["training"].get("curriculum", {})
    if curriculum_cfg:
        midpoint = float(curriculum_cfg.get("midpoint_fraction", 0.5))
        _ensure_range(midpoint, "training.curriculum.midpoint_fraction", 0.0, 1.0, source)
        _ensure_positive(float(curriculum_cfg.get("decoder_weight", 0.5)), "training.curriculum.decoder_weight", source)
        _ensure_positive(
            float(curriculum_cfg.get("bottleneck_weight", 0.25)),
            "training.curriculum.bottleneck_weight",
            source,
        )
        min_decay = float(curriculum_cfg.get("min_decay", 0.1))
        _ensure_positive(min_decay, "training.curriculum.min_decay", source)

    resume_cfg = cfg["training"].get("resume", {})
    if resume_cfg:
        if not isinstance(resume_cfg, dict):
            raise ValueError(f"{_path_label(source)}: training.resume must be a mapping when provided")
        if "enabled" in resume_cfg and not isinstance(resume_cfg.get("enabled"), bool):
            raise ValueError(f"{_path_label(source)}: training.resume.enabled must be a boolean")
        if "require_config_sha256_match" in resume_cfg and not isinstance(
            resume_cfg.get("require_config_sha256_match"), bool
        ):
            raise ValueError(
                f"{_path_label(source)}: training.resume.require_config_sha256_match must be a boolean"
            )
        if "fail_on_config_mismatch" in resume_cfg and not isinstance(
            resume_cfg.get("fail_on_config_mismatch"), bool
        ):
            raise ValueError(
                f"{_path_label(source)}: training.resume.fail_on_config_mismatch must be a boolean"
            )

    staged_cfg = cfg["training"].get("staged_losses", {})
    if staged_cfg.get("enabled", False):
        mode = str(staged_cfg.get("mode", cfg["training"].get("loss_mode", "legacy"))).strip().lower()
        if mode not in {"legacy", "silog", "silog_legacy", "flow", "flow_staged", "rectified_flow"}:
            raise ValueError(
                f"{_path_label(source)}: training.staged_losses.mode unsupported value '{mode}'"
            )

        stage_a = float(staged_cfg.get("stage_a_fraction", 0.3))
        stage_b = float(staged_cfg.get("stage_b_fraction", 0.7))
        _ensure_range(stage_a, "training.staged_losses.stage_a_fraction", 0.0, 1.0, source)
        _ensure_range(stage_b, "training.staged_losses.stage_b_fraction", 0.0, 1.0, source)
        if stage_b < stage_a:
            raise ValueError(
                f"{_path_label(source)}: training.staged_losses.stage_b_fraction must be >= stage_a_fraction"
            )
        if float(staged_cfg.get("ordinal_weight", 0.0)) < 0:
            raise ValueError(f"{_path_label(source)}: training.staged_losses.ordinal_weight must be >= 0")
        if float(staged_cfg.get("smoothness_weight", 0.0)) < 0:
            raise ValueError(f"{_path_label(source)}: training.staged_losses.smoothness_weight must be >= 0")

        if mode in {"flow", "flow_staged", "rectified_flow"}:
            if not bool(adaln_cfg.get("enabled", False)):
                raise ValueError(
                    f"{_path_label(source)}: flow-staged mode requires architecture.adaln_zero.enabled=true"
                )

            for key in ("flow_weight", "ssi_weight", "wavelet_weight", "ordinal_weight"):
                if float(staged_cfg.get(key, 0.0)) < 0.0:
                    raise ValueError(
                        f"{_path_label(source)}: training.staged_losses.{key} must be >= 0 in flow mode"
                    )

            flow_t_low = float(staged_cfg.get("flow_t_low", 0.0))
            flow_t_high = float(staged_cfg.get("flow_t_high", 1.0))
            _ensure_range(flow_t_low, "training.staged_losses.flow_t_low", 0.0, 1.0, source)
            _ensure_range(flow_t_high, "training.staged_losses.flow_t_high", 0.0, 1.0, source)
            if flow_t_high < flow_t_low:
                raise ValueError(
                    f"{_path_label(source)}: training.staged_losses.flow_t_high must be >= flow_t_low"
                )

            wavelet_levels = int(staged_cfg.get("wavelet_levels", 2))
            if wavelet_levels < 1:
                raise ValueError(
                    f"{_path_label(source)}: training.staged_losses.wavelet_levels must be >= 1"
                )

            allowed_families = {"sym4", "bior3.5"}
            family = str(staged_cfg.get("wavelet_family", "sym4")).strip().lower()
            fallback_family = str(staged_cfg.get("wavelet_fallback_family", "bior3.5")).strip().lower()
            if family not in allowed_families:
                raise ValueError(
                    f"{_path_label(source)}: training.staged_losses.wavelet_family must be one of {sorted(allowed_families)}"
                )
            if fallback_family not in allowed_families:
                raise ValueError(
                    f"{_path_label(source)}: training.staged_losses.wavelet_fallback_family must be one of {sorted(allowed_families)}"
                )

    ablation_cfg = cfg["training"].get("ablation", {})
    if ablation_cfg:
        if not isinstance(ablation_cfg, dict):
            raise ValueError(f"{_path_label(source)}: training.ablation must be a mapping when provided")

        if "enabled" in ablation_cfg and not isinstance(ablation_cfg.get("enabled"), bool):
            raise ValueError(f"{_path_label(source)}: training.ablation.enabled must be a boolean")
        if "allow_scaffold_only" in ablation_cfg and not isinstance(
            ablation_cfg.get("allow_scaffold_only"), bool
        ):
            raise ValueError(
                f"{_path_label(source)}: training.ablation.allow_scaffold_only must be a boolean"
            )

        variant = str(ablation_cfg.get("variant", "baseline")).strip().lower()
        allowed_variants = {
            "baseline",
            "frequency_only",
            "wavelet_only",
            "hybrid_wavelet_frequency",
        }
        if variant not in allowed_variants:
            raise ValueError(
                f"{_path_label(source)}: training.ablation.variant must be one of {sorted(allowed_variants)}"
            )

        if "notes" in ablation_cfg and not isinstance(ablation_cfg.get("notes"), str):
            raise ValueError(f"{_path_label(source)}: training.ablation.notes must be a string when provided")

    for gate_key in ("stage_a_gate", "stage_b_gate"):
        stage_gate_cfg = cfg.get("evaluation", {}).get(gate_key, {})
        if not stage_gate_cfg:
            continue
        if not isinstance(stage_gate_cfg, dict):
            raise ValueError(
                f"{_path_label(source)}: evaluation.{gate_key} must be a mapping when provided"
            )

        if "enabled" in stage_gate_cfg and not isinstance(stage_gate_cfg.get("enabled"), bool):
            raise ValueError(f"{_path_label(source)}: evaluation.{gate_key}.enabled must be a boolean")
        if "require_non_decreasing_pairs_trend" in stage_gate_cfg and not isinstance(
            stage_gate_cfg.get("require_non_decreasing_pairs_trend"), bool
        ):
            raise ValueError(
                f"{_path_label(source)}: evaluation.{gate_key}.require_non_decreasing_pairs_trend must be a boolean"
            )
        if "min_points" in stage_gate_cfg and int(stage_gate_cfg.get("min_points", 0)) < 1:
            raise ValueError(f"{_path_label(source)}: evaluation.{gate_key}.min_points must be >= 1")
        if "min_pairs_acc" in stage_gate_cfg:
            _ensure_range(
                float(stage_gate_cfg.get("min_pairs_acc", 5.0)),
                f"evaluation.{gate_key}.min_pairs_acc",
                0.0,
                100.0,
                source,
            )
        if "metric_key" in stage_gate_cfg:
            metric_key = str(stage_gate_cfg.get("metric_key", "")).strip().lower()
            allowed_metric_keys = {"pairs_acc", "trips_acc", "quads_acc", "all_acc"}
            if metric_key not in allowed_metric_keys:
                raise ValueError(
                    f"{_path_label(source)}: evaluation.{gate_key}.metric_key must be one of "
                    f"{sorted(allowed_metric_keys)}, got '{metric_key}'"
                )
        if "min_metric_total" in stage_gate_cfg and float(stage_gate_cfg.get("min_metric_total", 0.0)) < 0.0:
            raise ValueError(
                f"{_path_label(source)}: evaluation.{gate_key}.min_metric_total must be >= 0"
            )
        if "max_missing_layer_tuples" in stage_gate_cfg and float(
            stage_gate_cfg.get("max_missing_layer_tuples", 0.0)
        ) < 0.0:
            raise ValueError(
                f"{_path_label(source)}: evaluation.{gate_key}.max_missing_layer_tuples must be >= 0"
            )
        if "max_missing_layer_ratio" in stage_gate_cfg:
            _ensure_range(
                float(stage_gate_cfg.get("max_missing_layer_ratio", 0.0)),
                f"evaluation.{gate_key}.max_missing_layer_ratio",
                0.0,
                1.0,
                source,
            )
        if "expected_config_sha256" in stage_gate_cfg and not isinstance(
            stage_gate_cfg.get("expected_config_sha256"), str
        ):
            raise ValueError(
                f"{_path_label(source)}: evaluation.{gate_key}.expected_config_sha256 must be a string"
            )
        if "min_report_timestamp_utc" in stage_gate_cfg and not isinstance(
            stage_gate_cfg.get("min_report_timestamp_utc"), str
        ):
            raise ValueError(
                f"{_path_label(source)}: evaluation.{gate_key}.min_report_timestamp_utc must be a string"
            )
        if "require_config_sha256_match" in stage_gate_cfg and not isinstance(
            stage_gate_cfg.get("require_config_sha256_match"), bool
        ):
            raise ValueError(
                f"{_path_label(source)}: evaluation.{gate_key}.require_config_sha256_match must be a boolean"
            )
        if "require_report_timestamp" in stage_gate_cfg and not isinstance(
            stage_gate_cfg.get("require_report_timestamp"), bool
        ):
            raise ValueError(
                f"{_path_label(source)}: evaluation.{gate_key}.require_report_timestamp must be a boolean"
            )
        if "allow_fallback_report" in stage_gate_cfg and not isinstance(
            stage_gate_cfg.get("allow_fallback_report"), bool
        ):
            raise ValueError(
                f"{_path_label(source)}: evaluation.{gate_key}.allow_fallback_report must be a boolean"
            )
        if "report_dir" in stage_gate_cfg and not str(stage_gate_cfg.get("report_dir", "")).strip():
            raise ValueError(
                f"{_path_label(source)}: evaluation.{gate_key}.report_dir must be non-empty when provided"
            )

    stage_b_runtime_cfg = cfg.get("evaluation", {}).get("stage_b_runtime", {})
    if stage_b_runtime_cfg:
        if not isinstance(stage_b_runtime_cfg, dict):
            raise ValueError(
                f"{_path_label(source)}: evaluation.stage_b_runtime must be a mapping when provided"
            )
        if "enabled" in stage_b_runtime_cfg and not isinstance(stage_b_runtime_cfg.get("enabled"), bool):
            raise ValueError(
                f"{_path_label(source)}: evaluation.stage_b_runtime.enabled must be a boolean"
            )
        if "require_terminal_full_real_eval" in stage_b_runtime_cfg and not isinstance(
            stage_b_runtime_cfg.get("require_terminal_full_real_eval"), bool
        ):
            raise ValueError(
                f"{_path_label(source)}: evaluation.stage_b_runtime.require_terminal_full_real_eval must be a boolean"
            )
        if "hard_fail_on_terminal_eval_failure" in stage_b_runtime_cfg and not isinstance(
            stage_b_runtime_cfg.get("hard_fail_on_terminal_eval_failure"), bool
        ):
            raise ValueError(
                f"{_path_label(source)}: evaluation.stage_b_runtime.hard_fail_on_terminal_eval_failure must be a boolean"
            )

        if "max_epochs" in stage_b_runtime_cfg and int(stage_b_runtime_cfg.get("max_epochs", 0)) < 1:
            raise ValueError(
                f"{_path_label(source)}: evaluation.stage_b_runtime.max_epochs must be >= 1"
            )
        if "max_runtime_hours" in stage_b_runtime_cfg and float(
            stage_b_runtime_cfg.get("max_runtime_hours", 0.0)
        ) <= 0.0:
            raise ValueError(
                f"{_path_label(source)}: evaluation.stage_b_runtime.max_runtime_hours must be > 0"
            )

    if "fixture_checkpoint" in cfg.get("evaluation", {}) and not isinstance(
        cfg.get("evaluation", {}).get("fixture_checkpoint"), str
    ):
        raise ValueError(
            f"{_path_label(source)}: evaluation.fixture_checkpoint must be a string when provided"
        )

    # Validate backbone policy and fallback schema.
    collect_backbone_candidates(cfg["architecture"])

    return cfg

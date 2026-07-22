"""Attention-weight heatmap audit for the controlled Q+O proxy adapter.

The tool is intentionally independent from training.  It captures selected
eager-attention tensors through forward hooks and compares four reversible
adapter modes without modifying or republishing adapter artifacts.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
from typing import Any, Iterator, Mapping, Sequence

import numpy as np
import yaml

from anchor_mvp.training import qwen_lora_diagnostic as qdiag
from anchor_mvp.training import qwen_synthetic_scaffold_diagnostic as synth
from anchor_mvp.training.config import ConfigError, _expand_env


CONFIG_VERSION = "anchor.qwen-attention-weight-hook-config.v1"
SUMMARY_VERSION = "anchor.qwen-attention-weight-hook-summary.v1"
CONFIG_PATH = "configs/research/qwen_attention_weight_hook_v1.yaml"
IMPLEMENTATION_PATH = "src/anchor_mvp/research/qwen_attention_weight_hook.py"
SELECTED_LAYERS = (0, 13, 27)
MODES = ("adapter_off", "q_only_component", "o_only_component", "full")
DIFFERENCE_PANEL = "full_minus_q_only_component"
_SHA_RE = __import__("re").compile(r"^[0-9a-f]{64}$")
_MAX_METADATA_BYTES = 2_000_000
_ADAPTER_FILES = frozenset(
    {
        "adapter_config.json",
        "adapter_model.safetensors",
        "diagnostic_receipt.json",
        "diagnostic_receipt.json.sha256",
    }
)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            separators=(",", ": "),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _root() -> Path:
    return qdiag._project_root_from_module()


def _strict_json(data: bytes, label: str) -> Mapping[str, Any]:
    def pairs_hook(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ConfigError(f"{label} contains duplicate key: {key}")
            result[key] = value
        return result

    def reject_constant(value: str) -> object:
        raise ConfigError(f"{label} contains non-finite number: {value}")

    try:
        value = json.loads(
            data.decode("utf-8"),
            object_pairs_hook=pairs_hook,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ConfigError(f"{label} is not strict UTF-8 JSON") from exc
    if not isinstance(value, Mapping):
        raise ConfigError(f"{label} must contain an object")
    return value


def _require_sha(value: object, label: str) -> str:
    if not isinstance(value, str) or _SHA_RE.fullmatch(value) is None:
        raise ConfigError(f"{label} must be lowercase SHA-256")
    return value


def _canonical_config(path: str | Path) -> Path:
    canonical = _root() / CONFIG_PATH
    requested = Path(path)
    candidate = requested if requested.is_absolute() else _root() / requested
    if os.path.normcase(str(candidate.resolve())) != os.path.normcase(
        str(canonical.resolve())
    ):
        raise ConfigError(f"config must remain exactly {CONFIG_PATH}")
    return qdiag._assert_physical_path(
        canonical, require_file=True, label="attention audit config"
    )


def load_config(path: str | Path = CONFIG_PATH) -> dict[str, Any]:
    config_path = _canonical_config(path)
    raw = config_path.read_bytes()
    if len(raw) > _MAX_METADATA_BYTES or b"\r" in raw:
        raise ConfigError("attention audit config must be small LF-only UTF-8 YAML")
    try:
        value = yaml.safe_load(raw.decode("utf-8"))
    except (UnicodeDecodeError, yaml.YAMLError) as exc:
        raise ConfigError("attention audit config is invalid") from exc
    if not isinstance(value, dict):
        raise ConfigError("attention audit config must contain a mapping")
    config = _expand_env(value)
    config["_config_path"] = str(config_path)
    validate_config(config)
    return config


def validate_config(config: Mapping[str, Any]) -> None:
    if set(config) != {
        "schema_version",
        "claim_scope",
        "model",
        "adapter",
        "probe",
        "capture",
        "output",
        "claims",
        "audit",
        "_config_path",
    }:
        raise ConfigError("attention audit config fields drifted")
    if (
        config.get("schema_version") != CONFIG_VERSION
        or config.get("claim_scope")
        != "diagnostic_attention_proxy_only_no_quality_or_formal_authority"
    ):
        raise ConfigError("attention audit identity/claim scope drifted")
    model = config.get("model")
    if not isinstance(model, Mapping) or (
        model.get("id") != qdiag.EXPECTED_MODEL_ID
        or model.get("local_files_only") is not True
        or model.get("allow_network") is not False
        or model.get("trust_remote_code") is not False
        or model.get("expected_source_revision") != qdiag.EXPECTED_SOURCE_REVISION
        or model.get("expected_source_repo") != qdiag.EXPECTED_SOURCE_REPO
    ):
        raise ConfigError("attention audit model contract drifted")
    for key in (
        "expected_config_json_sha256",
        "expected_model_safetensors_sha256",
        "expected_tokenizer_json_sha256",
        "expected_tokenizer_config_sha256",
    ):
        _require_sha(model.get(key), f"model.{key}")
    adapter = config.get("adapter")
    if not isinstance(adapter, Mapping) or set(adapter) != {
        "path",
        "profile",
        "expected_adapter_config_sha256",
        "expected_adapter_model_sha256",
        "expected_diagnostic_receipt_sha256",
    }:
        raise ConfigError("attention audit adapter fields drifted")
    if adapter.get("profile") != "q_plus_o" or adapter.get("path") != (
        "artifacts/diagnostics/"
        "qwen2_5_1_5b_synthetic_scaffold_budget_matched_q_plus_o_step80"
    ):
        raise ConfigError("attention audit requires the controlled Q+O adapter")
    for key in (
        "expected_adapter_config_sha256",
        "expected_adapter_model_sha256",
        "expected_diagnostic_receipt_sha256",
    ):
        _require_sha(adapter.get(key), f"adapter.{key}")
    probe = config.get("probe")
    if not isinstance(probe, Mapping) or set(probe) != {
        "source",
        "prompt",
        "target",
        "max_sequence_length",
        "expected_batch_size",
    }:
        raise ConfigError("attention audit probe fields drifted")
    if (
        probe.get("source") != "inline_synthetic_no_dataset"
        or not isinstance(probe.get("prompt"), str)
        or not probe["prompt"]
        or not isinstance(probe.get("target"), str)
        or not probe["target"]
        or probe.get("max_sequence_length") != 512
        or probe.get("expected_batch_size") != 1
    ):
        raise ConfigError("attention audit probe contract drifted")
    capture = config.get("capture")
    expected_capture = {
        "attention_implementation": "eager",
        "selected_layers": list(SELECTED_LAYERS),
        "expected_decoder_layers": 28,
        "aggregation": "head_mean_float32_cpu",
        "modes": list(MODES),
        "difference_panel": DIFFERENCE_PANEL,
        "component_semantics": {
            "q_only_component": "o_proj_scaling_zero",
            "o_only_component": "q_proj_scaling_zero",
        },
        "use_cache": False,
        "output_attentions": True,
    }
    if capture != expected_capture:
        raise ConfigError("attention capture contract drifted")
    if config.get("output") != {
        "directory": "artifacts/diagnostics/qwen_attention_weight_hook_qpluso_v1",
        "summary_filename": "summary.json",
        "summary_sidecar_filename": "summary.json.sha256",
        "heatmap_template": "layer_{layer:02d}_attention.png",
        "image_dpi": 150,
    }:
        raise ConfigError("attention output contract drifted")
    if config.get("claims") != {
        "diagnostic_only": True,
        "training_authorized": False,
        "formal": False,
        "quality_validated": False,
        "causal_effect_proven": False,
        "attention_equals_explanation": False,
    }:
        raise ConfigError("attention claims drifted")
    if config.get("audit") != {
        "network_requests": 0,
        "heldout_reads": 0,
        "protected_body_reads": 0,
        "dataset_reads": 0,
        "single_probe": True,
        "prompt_or_target_text_in_output": False,
        "raw_token_ids_in_output": False,
        "atomic_no_replace": True,
    }:
        raise ConfigError("attention audit boundary drifted")


def _adapter_path(config: Mapping[str, Any]) -> Path:
    path = (_root() / str(config["adapter"]["path"])).resolve()
    diagnostics = (_root() / "artifacts" / "diagnostics").resolve()
    if path == diagnostics or diagnostics not in path.parents:
        raise ConfigError("adapter path escaped artifacts/diagnostics")
    return path


def authenticate_adapter(config: Mapping[str, Any]) -> dict[str, str]:
    path = _adapter_path(config)
    synth._assert_exact_regular_files(path, _ADAPTER_FILES, label="Q+O adapter")
    expected = {
        "adapter_config.json": config["adapter"]["expected_adapter_config_sha256"],
        "adapter_model.safetensors": config["adapter"]["expected_adapter_model_sha256"],
        "diagnostic_receipt.json": config["adapter"][
            "expected_diagnostic_receipt_sha256"
        ],
    }
    observed = {name: _sha256((path / name).read_bytes()) for name in sorted(expected)}
    if observed != expected:
        raise ConfigError("Q+O adapter physical SHA-256 identity drifted")
    sidecar = (path / "diagnostic_receipt.json.sha256").read_bytes()
    receipt_sha = expected["diagnostic_receipt.json"]
    if sidecar != f"{receipt_sha}  diagnostic_receipt.json\n".encode("ascii"):
        raise ConfigError("Q+O adapter receipt sidecar is malformed")
    receipt = _strict_json(
        (path / "diagnostic_receipt.json").read_bytes(), "Q+O adapter receipt"
    )
    claims = receipt.get("claims")
    if (
        receipt.get("profile") != "q_plus_o"
        or receipt.get("status") != "passed_controlled_proxy_only"
        or not isinstance(claims, Mapping)
        or claims.get("diagnostic_only") is not True
        or claims.get("formal") is not False
        or claims.get("training_authorized") is not False
    ):
        raise ConfigError("Q+O adapter receipt crossed the diagnostic boundary")
    return observed


class AttentionWeightHook:
    """Capture selected self-attention weights as head-mean float32 matrices."""

    def __init__(self, selected_layers: Sequence[int] = SELECTED_LAYERS) -> None:
        self.selected_layers = tuple(int(value) for value in selected_layers)
        if not self.selected_layers or len(set(self.selected_layers)) != len(
            self.selected_layers
        ):
            raise ValueError("selected layers must be unique and non-empty")
        self._handles: list[Any] = []
        self._captured: dict[int, np.ndarray] = {}

    @staticmethod
    def decoder_layers(model: Any) -> Sequence[Any]:
        candidates = (
            ("base_model", "model", "model", "layers"),
            ("model", "model", "layers"),
            ("model", "layers"),
        )
        for chain in candidates:
            value = model
            try:
                for attribute in chain:
                    value = getattr(value, attribute)
            except AttributeError:
                continue
            if hasattr(value, "__len__") and hasattr(value, "__getitem__"):
                return value
        raise RuntimeError("cannot locate decoder layers for attention hooks")

    def _forward_hook(self, layer: int):
        def capture(_module: Any, _args: Any, output: Any) -> None:
            weights = (
                output[1]
                if isinstance(output, (tuple, list)) and len(output) > 1
                else None
            )
            if weights is None or not hasattr(weights, "ndim") or weights.ndim != 4:
                raise RuntimeError(
                    f"layer {layer} eager attention returned no 4D attention weights"
                )
            if int(weights.shape[0]) != 1:
                raise RuntimeError("attention audit requires batch size one")
            matrix = weights.detach().float().mean(dim=1)[0].cpu().contiguous().numpy()
            if matrix.ndim != 2 or not np.isfinite(matrix).all():
                raise RuntimeError("captured attention matrix is invalid")
            if layer in self._captured:
                raise RuntimeError(f"layer {layer} attention captured more than once")
            self._captured[layer] = np.asarray(matrix, dtype=np.float32)

        return capture

    def install(self, model: Any, *, expected_decoder_layers: int = 28) -> None:
        if self._handles:
            raise RuntimeError("attention hooks are already installed")
        layers = self.decoder_layers(model)
        if len(layers) != expected_decoder_layers:
            raise RuntimeError("decoder layer count drifted")
        if any(layer < 0 or layer >= len(layers) for layer in self.selected_layers):
            raise RuntimeError("selected attention layer is out of range")
        self._handles = [
            layers[layer].self_attn.register_forward_hook(self._forward_hook(layer))
            for layer in self.selected_layers
        ]

    def clear(self) -> None:
        self._captured.clear()

    def snapshot(self) -> dict[int, np.ndarray]:
        if set(self._captured) != set(self.selected_layers):
            raise RuntimeError("selected attention layers were not all captured")
        return {layer: value.copy() for layer, value in self._captured.items()}

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self.clear()

    def __enter__(self) -> "AttentionWeightHook":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


def _lora_scaling_modules(model: Any) -> dict[str, list[Any]]:
    result: dict[str, list[Any]] = {"q_proj": [], "o_proj": []}
    for name, module in model.named_modules():
        target = name.rsplit(".", 1)[-1]
        scaling = getattr(module, "scaling", None)
        if target in result and isinstance(scaling, dict) and scaling:
            result[target].append(module)
    if not result["q_proj"] or not result["o_proj"]:
        raise RuntimeError("Q+O adapter scaling modules are incomplete")
    return result


@contextmanager
def component_scaling(model: Any, enabled: frozenset[str]) -> Iterator[dict[str, int]]:
    if not enabled <= {"q_proj", "o_proj"}:
        raise ValueError("component scaling accepts only q_proj/o_proj")
    modules = _lora_scaling_modules(model)
    originals: list[tuple[dict[str, Any], str, Any]] = []
    changed = {"q_proj": 0, "o_proj": 0}
    try:
        for target, values in modules.items():
            for module in values:
                for adapter_name, scaling in module.scaling.items():
                    originals.append((module.scaling, adapter_name, scaling))
                    if target not in enabled:
                        module.scaling[adapter_name] = 0.0
                        changed[target] += 1
        yield changed
    finally:
        for mapping, adapter_name, scaling in originals:
            mapping[adapter_name] = scaling


def _attention_sha256(matrix: np.ndarray) -> str:
    value = np.ascontiguousarray(matrix, dtype="<f4")
    header = json.dumps(list(value.shape), separators=(",", ":")).encode("ascii")
    return _sha256(header + b"\0" + value.tobytes(order="C"))


def _attention_metrics(matrix: np.ndarray, target_start: int) -> dict[str, Any]:
    value = np.asarray(matrix, dtype=np.float64)
    if value.ndim != 2 or value.shape[0] != value.shape[1]:
        raise ValueError("attention matrix must be square")
    if target_start <= 0 or target_start >= value.shape[0]:
        raise ValueError("target boundary must lie inside the attention matrix")
    rows = np.clip(value[target_start:, :], 0.0, None)
    totals = rows.sum(axis=1, keepdims=True)
    if np.any(totals <= 0):
        raise ValueError("target attention row has zero mass")
    probabilities = rows / totals
    log_probabilities = np.zeros_like(probabilities)
    positive = probabilities > 0
    log_probabilities[positive] = np.log(probabilities[positive])
    entropy = -np.sum(probabilities * log_probabilities, axis=1)
    nonzero = np.count_nonzero(probabilities > 0, axis=1)
    denominator = np.log(np.maximum(nonzero, 2))
    return {
        "attention_sha256": _attention_sha256(value.astype(np.float32)),
        "shape": [int(item) for item in value.shape],
        "target_query_prompt_mass_mean": float(
            probabilities[:, :target_start].sum(axis=1).mean()
        ),
        "target_query_entropy_mean": float(entropy.mean()),
        "target_query_normalized_entropy_mean": float((entropy / denominator).mean()),
    }


def _difference_metrics(left: np.ndarray, right: np.ndarray) -> dict[str, Any]:
    difference = np.asarray(left, dtype=np.float32) - np.asarray(
        right, dtype=np.float32
    )
    if difference.shape != left.shape or not np.isfinite(difference).all():
        raise ValueError("attention difference is invalid")
    return {
        "difference_sha256": _attention_sha256(difference),
        "mean_absolute": float(np.mean(np.abs(difference))),
        "root_mean_square": float(np.sqrt(np.mean(np.square(difference)))),
        "maximum_absolute": float(np.max(np.abs(difference))),
    }


def build_summary(
    config: Mapping[str, Any],
    matrices: Mapping[str, Mapping[int, np.ndarray]],
    *,
    prompt_tokens: int,
    target_tokens: int,
    full_token_ids_sha256: str,
    adapter_hashes: Mapping[str, str],
    runtime: Mapping[str, Any],
) -> dict[str, Any]:
    if set(matrices) != set(MODES):
        raise ValueError("attention modes are incomplete")
    total_tokens = prompt_tokens + target_tokens
    layers: dict[str, Any] = {}
    for layer in SELECTED_LAYERS:
        by_mode = {mode: matrices[mode][layer] for mode in MODES}
        if any(
            value.shape != (total_tokens, total_tokens) for value in by_mode.values()
        ):
            raise ValueError("attention shape disagrees with token boundary")
        layers[str(layer)] = {
            "modes": {
                mode: _attention_metrics(value, prompt_tokens)
                for mode, value in by_mode.items()
            },
            "mode_differences_from_full": {
                mode: _difference_metrics(by_mode["full"], by_mode[mode])
                for mode in MODES
                if mode != "full"
            },
            DIFFERENCE_PANEL: _difference_metrics(
                by_mode["full"], by_mode["q_only_component"]
            ),
        }
    config_path = Path(str(config["_config_path"]))
    implementation_path = _root() / IMPLEMENTATION_PATH
    return {
        "schema_version": SUMMARY_VERSION,
        "status": "completed_attention_proxy_only",
        "identity": {
            "config_path": CONFIG_PATH,
            "config_sha256": _sha256(config_path.read_bytes()),
            "implementation_path": IMPLEMENTATION_PATH,
            "implementation_sha256": _sha256(implementation_path.read_bytes()),
            "adapter_artifact_sha256": dict(adapter_hashes),
        },
        "probe": {
            "source": "inline_synthetic_no_dataset",
            "prompt_sha256": _sha256(config["probe"]["prompt"].encode("utf-8")),
            "target_sha256": _sha256(config["probe"]["target"].encode("utf-8")),
            "full_token_ids_sha256": _require_sha(
                full_token_ids_sha256, "full token-ID digest"
            ),
            "full_token_ids_digest_algorithm": (
                "sha256_signed_int64_big_endian_concat_v1"
            ),
            "prompt_tokens": prompt_tokens,
            "target_tokens": target_tokens,
            "total_tokens": total_tokens,
            "target_boundary_zero_based": prompt_tokens,
            "raw_token_ids_emitted": False,
            "prompt_or_target_text_emitted": False,
        },
        "capture": {
            "attention_implementation": "eager",
            "selected_layers": list(SELECTED_LAYERS),
            "aggregation": "head_mean_float32_cpu",
            "modes": list(MODES),
            "difference_panel": DIFFERENCE_PANEL,
            "layers": layers,
        },
        "runtime": dict(runtime),
        "claims": dict(config["claims"]),
        "audit": {
            **dict(config["audit"]),
            "model_loads": 1,
            "adapter_loads": 1,
            "tokenizer_loads": 1,
            "gpu_requests": 1,
        },
    }


def _output_path(config: Mapping[str, Any]) -> Path:
    path = (_root() / str(config["output"]["directory"])).resolve()
    allowed = (_root() / "artifacts" / "diagnostics").resolve()
    if path == allowed or allowed not in path.parents:
        raise ConfigError("attention output escaped artifacts/diagnostics")
    if os.path.lexists(path):
        raise ConfigError(f"attention output already exists: {path}")
    return path


def render_layer_heatmap(
    matrices: Mapping[str, np.ndarray],
    *,
    layer: int,
    target_start: int,
    output: Path,
    dpi: int,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    difference = matrices["full"] - matrices["q_only_component"]
    panels = [*(matrices[mode] for mode in MODES), difference]
    titles = [*MODES, DIFFERENCE_PANEL]
    figure, axes = plt.subplots(1, 5, figsize=(20, 4), constrained_layout=True)
    for index, (axis, value, title) in enumerate(zip(axes, panels, titles)):
        if index == 4:
            limit = max(float(np.max(np.abs(value))), 1e-8)
            image = axis.imshow(
                value, cmap="coolwarm", vmin=-limit, vmax=limit, aspect="auto"
            )
        else:
            image = axis.imshow(value, cmap="viridis", vmin=0.0, aspect="auto")
        axis.axvline(target_start - 0.5, color="white", linewidth=0.8)
        axis.axhline(target_start - 0.5, color="white", linewidth=0.8)
        axis.set_title(title)
        axis.set_xlabel("key token position")
        axis.set_ylabel("query token position")
        figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    figure.suptitle(f"layer {layer} head-mean attention")
    figure.savefig(output, dpi=dpi, metadata={"Title": f"layer {layer} attention"})
    plt.close(figure)


def _validate_output(
    path: Path, config: Mapping[str, Any], expected_summary_sha256: str
) -> None:
    expected = {
        str(config["output"]["summary_filename"]),
        str(config["output"]["summary_sidecar_filename"]),
        *(
            str(config["output"]["heatmap_template"]).format(layer=layer)
            for layer in SELECTED_LAYERS
        ),
    }
    synth._assert_exact_regular_files(
        path, frozenset(expected), label="attention output"
    )
    summary_name = str(config["output"]["summary_filename"])
    sidecar_name = str(config["output"]["summary_sidecar_filename"])
    summary = (path / summary_name).read_bytes()
    sidecar = (path / sidecar_name).read_bytes()
    if _sha256(summary) != expected_summary_sha256 or sidecar != (
        f"{expected_summary_sha256}  {summary_name}\n".encode("ascii")
    ):
        raise RuntimeError("attention summary authentication failed")
    value = _strict_json(summary, "attention summary")
    if value.get("schema_version") != SUMMARY_VERSION or value.get("status") != (
        "completed_attention_proxy_only"
    ):
        raise RuntimeError("attention summary identity drifted")


def publish_audit(
    config: Mapping[str, Any],
    summary: Mapping[str, Any],
    matrices: Mapping[str, Mapping[int, np.ndarray]],
) -> tuple[Path, str]:
    output = _output_path(config)
    with qdiag._adapter_staging_directory(output) as staging:
        heatmaps: dict[str, str] = {}
        for layer in SELECTED_LAYERS:
            name = str(config["output"]["heatmap_template"]).format(layer=layer)
            render_layer_heatmap(
                {mode: matrices[mode][layer] for mode in MODES},
                layer=layer,
                target_start=int(summary["probe"]["target_boundary_zero_based"]),
                output=staging / name,
                dpi=int(config["output"]["image_dpi"]),
            )
            heatmaps[name] = _sha256((staging / name).read_bytes())
        final_summary = dict(summary)
        final_summary["heatmaps"] = heatmaps
        summary_bytes = _canonical_json_bytes(final_summary)
        summary_sha = _sha256(summary_bytes)
        summary_name = str(config["output"]["summary_filename"])
        sidecar_name = str(config["output"]["summary_sidecar_filename"])
        (staging / summary_name).write_bytes(summary_bytes)
        (staging / sidecar_name).write_bytes(
            f"{summary_sha}  {summary_name}\n".encode("ascii")
        )
        _validate_output(staging, config, summary_sha)
        qdiag._rename_directory_noreplace(staging, output)
    try:
        _validate_output(output, config, summary_sha)
    except Exception:
        if os.path.lexists(output) and not qdiag._is_reparse_or_symlink(output):
            shutil.rmtree(output)
        raise
    return output / str(config["output"]["summary_filename"]), summary_sha


def _qdiag_config(config: Mapping[str, Any], output: Path) -> dict[str, Any]:
    return {
        "schema_version": qdiag.SCHEMA_VERSION,
        "paths": {"project_root": "../.."},
        "model": dict(config["model"]),
        "lora": {
            "rank": 4,
            "alpha": 8,
            "dropout": 0.0,
            "bias": "none",
            "target_modules": ["q_proj"],
        },
        "training": {
            "max_steps": 1,
            "batch_size": 1,
            "gradient_accumulation_steps": 1,
            "sequence_length": 128,
            "learning_rate": 0.0001,
            "seed": 1337,
        },
        "dataset": {
            "kind": "inline_toy_plumbing_v1",
            "formal_inputs_allowed": False,
            "heldout_allowed": False,
        },
        "output": {"adapter_dir": output.relative_to(_root()).as_posix()},
        "_config_path": config["_config_path"],
    }


def _token_ids(value: object) -> list[int]:
    return synth._token_ids(value)


def _tokenize_probe(tokenizer: Any, config: Mapping[str, Any]) -> tuple[list[int], int]:
    prompt_messages = [{"role": "user", "content": config["probe"]["prompt"]}]
    full_messages = [
        *prompt_messages,
        {"role": "assistant", "content": config["probe"]["target"]},
    ]
    prompt_ids = _token_ids(
        tokenizer.apply_chat_template(
            prompt_messages, tokenize=True, add_generation_prompt=True
        )
    )
    full_ids = _token_ids(
        tokenizer.apply_chat_template(
            full_messages, tokenize=True, add_generation_prompt=False
        )
    )
    if full_ids[: len(prompt_ids)] != prompt_ids:
        raise RuntimeError("probe target is not a prompt-prefix continuation")
    if len(full_ids) > int(config["probe"]["max_sequence_length"]):
        raise RuntimeError("single attention probe exceeds sequence cap")
    if not 0 < len(prompt_ids) < len(full_ids):
        raise RuntimeError("single attention probe has no target tokens")
    return full_ids, len(prompt_ids)


def _capture_mode(
    model: Any, hook: AttentionWeightHook, batch: Mapping[str, Any], torch: Any
) -> dict[int, np.ndarray]:
    hook.clear()
    with torch.inference_mode():
        output = model(
            **batch,
            use_cache=False,
            output_attentions=True,
            return_dict=True,
        )
    captured = hook.snapshot()
    del output
    return captured


def execute(config: Mapping[str, Any]) -> Mapping[str, Any]:
    """Run the single-probe audit; never trains or modifies adapter weights."""

    adapter_hashes = authenticate_adapter(config)
    output = _output_path(config)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("attention audit requires CUDA")
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    base_config = _qdiag_config(config, output)
    source_model = qdiag._resolved_local_model(base_config)
    with qdiag._authenticated_model_snapshot(base_config, output.parent) as (
        model_path,
        snapshot_identity,
    ):
        tokenizer = AutoTokenizer.from_pretrained(
            model_path, local_files_only=True, trust_remote_code=False
        )
        token_ids, target_start = _tokenize_probe(tokenizer, config)
        batch = {
            "input_ids": torch.tensor([token_ids], dtype=torch.long, device="cuda"),
            "attention_mask": torch.ones(
                (1, len(token_ids)), dtype=torch.long, device="cuda"
            ),
        }
        base = AutoModelForCausalLM.from_pretrained(
            model_path,
            local_files_only=True,
            trust_remote_code=False,
            torch_dtype=torch.bfloat16,
            attn_implementation="eager",
            low_cpu_mem_usage=True,
        ).to("cuda")
        model = PeftModel.from_pretrained(
            base,
            _adapter_path(config),
            is_trainable=False,
            local_files_only=True,
        )
        model.eval()
        hook = AttentionWeightHook(SELECTED_LAYERS)
        hook.install(model, expected_decoder_layers=28)
        matrices: dict[str, dict[int, np.ndarray]] = {}
        try:
            with model.disable_adapter():
                matrices["adapter_off"] = _capture_mode(model, hook, batch, torch)
            with component_scaling(model, frozenset({"q_proj"})):
                matrices["q_only_component"] = _capture_mode(model, hook, batch, torch)
            with component_scaling(model, frozenset({"o_proj"})):
                matrices["o_only_component"] = _capture_mode(model, hook, batch, torch)
            with component_scaling(model, frozenset({"q_proj", "o_proj"})):
                matrices["full"] = _capture_mode(model, hook, batch, torch)
        finally:
            hook.close()
        full_token_ids_sha = synth._signed_int64_sequence_sha256(token_ids)
        summary = build_summary(
            config,
            matrices,
            prompt_tokens=target_start,
            target_tokens=len(token_ids) - target_start,
            full_token_ids_sha256=full_token_ids_sha,
            adapter_hashes=adapter_hashes,
            runtime={
                "device": torch.cuda.get_device_name(0),
                "dtype": "bfloat16",
                "tf32": True,
                "model_snapshot_identity_sha256": _sha256(
                    json.dumps(
                        snapshot_identity, sort_keys=True, separators=(",", ":")
                    ).encode("utf-8")
                ),
                "source_model_path_sha256": _sha256(str(source_model).encode("utf-8")),
            },
        )
        if authenticate_adapter(config) != adapter_hashes:
            raise RuntimeError("Q+O adapter changed during attention capture")
        summary_path, summary_sha = publish_audit(config, summary, matrices)
        del model, base, tokenizer, batch
        gc.collect()
        torch.cuda.empty_cache()
        return {
            "summary": summary_path.relative_to(_root()).as_posix(),
            "summary_sha256": summary_sha,
            "status": "completed_attention_proxy_only",
        }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Q+O attention-weight heatmap audit")
    parser.add_argument("--config", default=CONFIG_PATH)
    parser.add_argument("--execute", action="store_true", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = execute(load_config(args.config))
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Strictly isolated Qwen2.5 1.5B one-step LoRA diagnostic.

The entry point deliberately does not share the formal A--F configuration or
datasets.  Imports of torch/transformers/PEFT happen only after ``--execute``
and only after a local Hugging Face directory has passed the static gates.
"""

from __future__ import annotations

import argparse
import ctypes
import gc
import hashlib
import json
import math
import os
import re
import shutil
import stat
import subprocess
import sys
import uuid
from contextlib import contextmanager
from pathlib import Path, PurePosixPath
from typing import Any, Iterator, Mapping, Sequence

from .config import ConfigError, _expand_env, _read_mapping
from .manifest import config_fingerprint, sha256_file, write_json


SCHEMA_VERSION = "anchor.qwen25-1.5b-lora-one-step-diagnostic.v1"
EXPECTED_MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
EXPECTED_SOURCE_REVISION = "3c3787b7c81927cc64ad45dc32ff1c9ce2a5de34"
EXPECTED_SOURCE_REPO = "https://www.modelscope.cn/Qwen/Qwen2.5-1.5B-Instruct.git"
_REPO_ROOT = Path(__file__).resolve().parents[3]
_GGUF_MARKERS = (".gguf", "-gguf", "q4_k", "q5_k", "q8_0")
_LORA_TENSOR_RE = re.compile(
    r"(?:^|\.)model\.layers\.(\d+)\.self_attn\.q_proj\."
    r"lora_([AB])(?:\.[^.]+)?\.weight$"
)
_ADAPTER_FILES = ("adapter_config.json", "adapter_model.safetensors")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_EXPECTED_ARTIFACTS = {
    "expected_config_json_sha256": "config.json",
    "expected_model_safetensors_sha256": "model.safetensors",
    "expected_tokenizer_json_sha256": "tokenizer.json",
    "expected_tokenizer_config_sha256": "tokenizer_config.json",
}
_TOP_LEVEL_KEYS = {
    "schema_version",
    "paths",
    "model",
    "lora",
    "training",
    "dataset",
    "output",
    "_config_path",
}


def _mapping(config: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = config.get(key)
    if not isinstance(value, Mapping):
        raise ConfigError(f"{key} must be a mapping")
    return value


def _exact_int(value: Any, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"{path} must be an integer")
    return value


def _reject_unknown_keys(
    value: Mapping[str, Any], allowed: set[str], path: str
) -> None:
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ConfigError(f"{path} contains unknown fields: {', '.join(unknown)}")


def validate_qwen_diagnostic_config(config: Mapping[str, Any]) -> None:
    _reject_unknown_keys(config, _TOP_LEVEL_KEYS, "config")
    if config.get("schema_version") != SCHEMA_VERSION:
        raise ConfigError(f"schema_version must be {SCHEMA_VERSION!r}")

    paths = _mapping(config, "paths")
    _reject_unknown_keys(paths, {"project_root"}, "paths")
    if paths.get("project_root") != "../..":
        raise ConfigError("paths.project_root must remain exactly '../..'")
    model = _mapping(config, "model")
    _reject_unknown_keys(
        model,
        {
            "id",
            "local_path",
            "local_files_only",
            "allow_network",
            "trust_remote_code",
            "expected_source_revision",
            "expected_source_repo",
            *_EXPECTED_ARTIFACTS,
        },
        "model",
    )
    model_id = model.get("id")
    local_path = model.get("local_path")
    if model_id != EXPECTED_MODEL_ID:
        raise ConfigError(f"model.id must be exactly {EXPECTED_MODEL_ID!r}")
    if not isinstance(local_path, str) or not local_path.strip():
        raise ConfigError("model.local_path must be a non-empty local HF directory")
    serialized_identity = f"{model_id} {local_path}".casefold()
    if any(marker in serialized_identity for marker in _GGUF_MARKERS):
        raise ConfigError(
            "GGUF is inference-only and is rejected by this PEFT diagnostic; "
            "provide an unpacked local Hugging Face checkpoint directory"
        )
    if model.get("local_files_only") is not True:
        raise ConfigError("model.local_files_only must be true")
    if model.get("allow_network") is not False:
        raise ConfigError("model.allow_network must be false")
    if model.get("trust_remote_code") is not False:
        raise ConfigError("model.trust_remote_code must be false")
    if model.get("expected_source_revision") != EXPECTED_SOURCE_REVISION:
        raise ConfigError(
            f"model.expected_source_revision must be {EXPECTED_SOURCE_REVISION!r}"
        )
    if model.get("expected_source_repo") != EXPECTED_SOURCE_REPO:
        raise ConfigError(
            f"model.expected_source_repo must be {EXPECTED_SOURCE_REPO!r}"
        )
    for key in _EXPECTED_ARTIFACTS:
        value = model.get(key, "")
        if not isinstance(value, str):
            raise ConfigError(f"model.{key} must be a lowercase SHA-256 or empty")
        if value and _SHA256_RE.fullmatch(value) is None:
            raise ConfigError(f"model.{key} must be a lowercase SHA-256 or empty")

    lora = _mapping(config, "lora")
    _reject_unknown_keys(
        lora, {"rank", "alpha", "dropout", "bias", "target_modules"}, "lora"
    )
    rank = _exact_int(lora.get("rank"), "lora.rank")
    if rank not in (4, 8):
        raise ConfigError("lora.rank must be 4 or 8")
    if lora.get("target_modules") != ["q_proj"]:
        raise ConfigError("lora.target_modules must be exactly ['q_proj']")
    if lora.get("bias") != "none":
        raise ConfigError("lora.bias must be 'none'")
    if _exact_int(lora.get("alpha"), "lora.alpha") != rank * 2:
        raise ConfigError("lora.alpha must equal 2 * lora.rank")
    dropout = lora.get("dropout")
    if not isinstance(dropout, (int, float)) or float(dropout) != 0.0:
        raise ConfigError("lora.dropout must be 0 for the mechanical diagnostic")

    training = _mapping(config, "training")
    _reject_unknown_keys(
        training,
        {
            "max_steps",
            "batch_size",
            "gradient_accumulation_steps",
            "sequence_length",
            "learning_rate",
            "seed",
        },
        "training",
    )
    if _exact_int(training.get("max_steps"), "training.max_steps") != 1:
        raise ConfigError("training.max_steps must be 1")
    if _exact_int(training.get("batch_size"), "training.batch_size") != 1:
        raise ConfigError("training.batch_size must be 1")
    sequence_length = _exact_int(
        training.get("sequence_length"), "training.sequence_length"
    )
    if sequence_length < 128 or sequence_length > 256:
        raise ConfigError("training.sequence_length must be in [128, 256]")
    if training.get("gradient_accumulation_steps") != 1:
        raise ConfigError("training.gradient_accumulation_steps must be 1")
    learning_rate = training.get("learning_rate")
    if not isinstance(learning_rate, (int, float)) or not 0 < learning_rate <= 0.001:
        raise ConfigError("training.learning_rate must be in (0, 0.001]")
    _exact_int(training.get("seed"), "training.seed")

    dataset = _mapping(config, "dataset")
    _reject_unknown_keys(
        dataset,
        {"kind", "formal_inputs_allowed", "heldout_allowed"},
        "dataset",
    )
    if dataset.get("kind") != "inline_toy_plumbing_v1":
        raise ConfigError("dataset.kind must be 'inline_toy_plumbing_v1'")
    if dataset.get("formal_inputs_allowed") is not False:
        raise ConfigError("dataset.formal_inputs_allowed must be false")
    if dataset.get("heldout_allowed") is not False:
        raise ConfigError("dataset.heldout_allowed must be false")
    output = _mapping(config, "output")
    _reject_unknown_keys(output, {"adapter_dir"}, "output")
    if (
        not isinstance(output.get("adapter_dir"), str)
        or not output["adapter_dir"].strip()
    ):
        raise ConfigError("output.adapter_dir must be a non-empty path")
    try:
        output["adapter_dir"].format(rank=rank)
    except (KeyError, ValueError) as exc:
        raise ConfigError(
            "output.adapter_dir may only use the {rank} placeholder"
        ) from exc


def load_qwen_diagnostic_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(os.path.abspath(Path(path).expanduser()))
    _assert_physical_path(config_path, require_file=True, label="training config")
    expected_dir = _project_root_from_module() / "configs" / "training"
    if config_path.parent != expected_dir:
        raise ConfigError(
            "Qwen diagnostic config must be a physical file directly under "
            f"{expected_dir}"
        )
    config = _expand_env(_read_mapping(config_path))
    config["_config_path"] = str(config_path)
    validate_qwen_diagnostic_config(config)
    return config


def _project_root(config: Mapping[str, Any]) -> Path:
    if config.get("paths", {}).get("project_root") != "../..":
        raise ConfigError("paths.project_root must remain exactly '../..'")
    return _project_root_from_module()


def _project_root_from_module() -> Path:
    root = Path(os.path.abspath(_REPO_ROOT))
    _assert_physical_path(root, require_directory=True, label="repository root")
    return root


def _is_reparse_or_symlink(path: Path) -> bool:
    if path.is_symlink():
        return True
    try:
        attributes = int(getattr(os.lstat(path), "st_file_attributes", 0))
    except OSError:
        return False
    return bool(attributes & int(getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)))


def _assert_physical_path(
    path: Path,
    *,
    require_file: bool = False,
    require_directory: bool = False,
    label: str,
) -> Path:
    absolute = Path(os.path.abspath(path))
    if os.path.normcase(os.path.realpath(absolute)) != os.path.normcase(str(absolute)):
        raise ConfigError(f"{label} has realpath drift: {absolute}")
    current = absolute
    while True:
        if _lexists(current) and _is_reparse_or_symlink(current):
            raise ConfigError(
                f"{label} contains a symlink/reparse component: {current}"
            )
        parent = current.parent
        if parent == current:
            break
        current = parent
    if require_file and not absolute.is_file():
        raise ConfigError(f"{label} must be a physical file: {absolute}")
    if require_directory and not absolute.is_dir():
        raise ConfigError(f"{label} must be a physical directory: {absolute}")
    return absolute


def _resolved_local_model(config: Mapping[str, Any]) -> Path:
    candidate = Path(str(config["model"]["local_path"])).expanduser()
    if not candidate.is_absolute():
        candidate = _project_root(config) / candidate
    candidate = Path(os.path.abspath(candidate))
    if candidate.suffix.casefold() == ".gguf" or any(
        marker in candidate.name.casefold() for marker in _GGUF_MARKERS
    ):
        raise ConfigError("GGUF cannot be used by the LoRA training diagnostic")
    return candidate


def _resolved_output(config: Mapping[str, Any]) -> Path:
    allowed_root = Path(
        os.path.abspath(_project_root(config) / "artifacts" / "diagnostics")
    )
    candidate = Path(
        str(config["output"]["adapter_dir"]).format(rank=config["lora"]["rank"])
    ).expanduser()
    if not candidate.is_absolute():
        candidate = _project_root(config) / candidate
    candidate = Path(os.path.abspath(candidate))
    if candidate == allowed_root or not candidate.is_relative_to(allowed_root):
        raise ConfigError(
            "output.adapter_dir must be a child of project artifacts/diagnostics"
        )
    _assert_physical_path(candidate.parent, label="diagnostic output parent")
    return candidate


def _lexists(path: Path) -> bool:
    return os.path.lexists(path)


def _safe_json_snapshot(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    if not path.is_file() or path.is_symlink():
        return None, None
    try:
        raw = path.read_bytes()
        value = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None, None
    if not isinstance(value, dict):
        return None, None
    return value, hashlib.sha256(raw).hexdigest()


def _git_value(
    model_path: Path, *arguments: str, allow_empty: bool = False
) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(model_path), *arguments],
            check=False,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    value = result.stdout.strip()
    return value if value or allow_empty else None


def _source_identity(config: Mapping[str, Any], model_path: Path) -> dict[str, Any]:
    try:
        _assert_physical_path(
            model_path / ".git",
            require_directory=True,
            label="model Git metadata",
        )
        git_metadata_physical = True
    except ConfigError:
        git_metadata_physical = False
    observed_revision = _git_value(model_path, "rev-parse", "--verify", "HEAD")
    observed_repo = _git_value(model_path, "remote", "get-url", "origin")
    status = _git_value(
        model_path,
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
        allow_empty=True,
    )
    tracked = _git_value(model_path, "ls-files")
    tracked_files = tuple(line for line in (tracked or "").splitlines() if line)
    tracked_files_sha256 = hashlib.sha256(
        "\n".join(tracked_files).encode("utf-8")
    ).hexdigest()
    expected_revision = str(config["model"]["expected_source_revision"])
    expected_repo = str(config["model"]["expected_source_repo"])
    return {
        "expected_revision": expected_revision,
        "observed_revision": observed_revision,
        "expected_repo": expected_repo,
        "observed_repo": observed_repo,
        "git_metadata_physical": git_metadata_physical,
        "working_tree_clean": status == "",
        "tracked_file_count": len(tracked_files),
        "tracked_files_sha256": tracked_files_sha256,
        "matched": git_metadata_physical
        and observed_revision == expected_revision
        and observed_repo == expected_repo
        and status == ""
        and bool(tracked_files),
    }


def build_preflight(config: Mapping[str, Any]) -> dict[str, Any]:
    model_path = _resolved_local_model(config)
    output_path = _resolved_output(config)
    try:
        _assert_physical_path(
            model_path, require_directory=True, label="local HF model directory"
        )
        model_dir_physical = True
    except ConfigError:
        model_dir_physical = False
    source_identity = (
        _source_identity(config, model_path)
        if model_dir_physical
        else {
            "expected_revision": config["model"]["expected_source_revision"],
            "observed_revision": None,
            "expected_repo": config["model"]["expected_source_repo"],
            "observed_repo": None,
            "git_metadata_physical": False,
            "working_tree_clean": False,
            "tracked_file_count": 0,
            "tracked_files_sha256": hashlib.sha256(b"").hexdigest(),
            "matched": False,
        }
    )
    if model_dir_physical:
        model_config, model_config_sha256 = _safe_json_snapshot(
            model_path / "config.json"
        )
        tokenizer_config, tokenizer_config_observed = _safe_json_snapshot(
            model_path / "tokenizer_config.json"
        )
    else:
        model_config = None
        model_config_sha256 = None
        tokenizer_config = None
        tokenizer_config_observed = None
    artifact_identity: dict[str, dict[str, Any]] = {}
    artifacts_physical = True
    identities_match = True
    expected_present = True
    for expected_key, filename in _EXPECTED_ARTIFACTS.items():
        artifact = model_path / filename
        physical = (
            model_dir_physical and artifact.is_file() and not artifact.is_symlink()
        )
        observed = sha256_file(artifact) if physical else None
        snapshot_sha256 = {
            "config.json": model_config_sha256,
            "tokenizer_config.json": tokenizer_config_observed,
        }.get(filename)
        if snapshot_sha256 is not None and observed != snapshot_sha256:
            raise RuntimeError(f"{filename} changed during preflight")
        expected = str(config["model"].get(expected_key, ""))
        artifact_identity[filename] = {
            "expected_sha256": expected or None,
            "observed_sha256": observed,
            "physical_non_symlink_file": physical,
            "matched": bool(expected and observed == expected),
        }
        artifacts_physical = artifacts_physical and physical
        expected_present = expected_present and bool(expected)
        identities_match = identities_match and bool(expected and observed == expected)
    model_identity_valid = bool(
        model_config
        and model_config.get("model_type") == "qwen2"
        and model_config.get("architectures") == ["Qwen2ForCausalLM"]
        and model_config.get("num_hidden_layers") == 28
        and model_config.get("hidden_size") == 1536
    )
    chat_template_valid = bool(
        tokenizer_config
        and isinstance(tokenizer_config.get("chat_template"), str)
        and tokenizer_config["chat_template"].strip()
    )
    gates = {
        "isolated_toy_data": True,
        "network_disabled": config["model"]["allow_network"] is False,
        "hf_directory_physical": model_dir_physical,
        "source_git_identity": source_identity["matched"],
        "hf_config_closed_and_qwen2": model_identity_valid,
        "tokenizer_chat_template_present": chat_template_valid,
        "required_artifacts_physical": artifacts_physical,
        "expected_artifact_hashes_present": expected_present,
        "artifact_hashes_match": identities_match,
        "gguf_rejected": model_path.suffix.casefold() != ".gguf",
        "one_step_profile": True,
        "q_proj_only_rank": True,
        "output_scoped_and_absent": not _lexists(output_path),
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "config_sha256": config_fingerprint(config),
        "model_path": str(model_path),
        "model_config_sha256": model_config_sha256,
        "source_identity": source_identity,
        "artifact_identity": artifact_identity,
        "output_path": str(output_path),
        "gates": gates,
        "ready": all(gates.values()),
    }


def _tracked_model_files(model_path: Path) -> tuple[str, ...]:
    listing = _git_value(model_path, "ls-files")
    paths = tuple(line for line in (listing or "").splitlines() if line)
    if not paths:
        raise RuntimeError("model Git checkout exposes no tracked files")
    for value in paths:
        relative = PurePosixPath(value)
        if relative.is_absolute() or ".." in relative.parts:
            raise RuntimeError(f"unsafe tracked model path: {value!r}")
    return paths


def _snapshot_identity(
    config: Mapping[str, Any], source: Path, snapshot: Path
) -> dict[str, Any]:
    file_hashes: dict[str, str] = {}
    manifest_digest = hashlib.sha256()
    for relative_value in _tracked_model_files(source):
        relative = PurePosixPath(relative_value)
        copied = snapshot.joinpath(*relative.parts)
        if not copied.is_file() or _is_reparse_or_symlink(copied):
            raise RuntimeError(f"snapshot file is missing or linked: {relative_value}")
        expected_blob = _git_value(source, "rev-parse", f"HEAD:{relative_value}")
        observed_blob = _git_value(
            source,
            "hash-object",
            "--path",
            relative_value,
            str(copied),
        )
        if expected_blob is None or observed_blob != expected_blob:
            raise RuntimeError(
                f"snapshot file does not match source revision: {relative_value}"
            )
        physical_sha256 = sha256_file(copied)
        file_hashes[relative_value] = physical_sha256
        manifest_digest.update(relative_value.encode("utf-8"))
        manifest_digest.update(b"\0")
        manifest_digest.update(physical_sha256.encode("ascii"))
        manifest_digest.update(b"\n")
    for expected_key, filename in _EXPECTED_ARTIFACTS.items():
        if file_hashes.get(filename) != config["model"][expected_key]:
            raise RuntimeError(
                f"authenticated model snapshot hash mismatch: {filename}"
            )
    model_config, _ = _safe_json_snapshot(snapshot / "config.json")
    tokenizer_config, _ = _safe_json_snapshot(snapshot / "tokenizer_config.json")
    if not (
        model_config
        and model_config.get("model_type") == "qwen2"
        and model_config.get("architectures") == ["Qwen2ForCausalLM"]
        and model_config.get("num_hidden_layers") == 28
        and model_config.get("hidden_size") == 1536
    ):
        raise RuntimeError("authenticated snapshot config is not Qwen2ForCausalLM")
    if not (
        tokenizer_config
        and isinstance(tokenizer_config.get("chat_template"), str)
        and tokenizer_config["chat_template"].strip()
    ):
        raise RuntimeError("authenticated snapshot has no chat template")
    return {
        "tracked_file_count": len(file_hashes),
        "tracked_files_manifest_sha256": manifest_digest.hexdigest(),
        "artifact_sha256": {
            filename: file_hashes[filename] for filename in _EXPECTED_ARTIFACTS.values()
        },
    }


@contextmanager
def _authenticated_model_snapshot(
    config: Mapping[str, Any], output_parent: Path
) -> Iterator[tuple[Path, dict[str, Any]]]:
    source = _resolved_local_model(config)
    _assert_physical_path(
        source, require_directory=True, label="local HF model directory"
    )
    _assert_physical_path(output_parent, label="diagnostic output parent")
    output_parent.mkdir(parents=True, exist_ok=True)
    _assert_physical_path(
        output_parent, require_directory=True, label="diagnostic output parent"
    )
    snapshot = output_parent / f".model-snapshot-{uuid.uuid4().hex}"
    snapshot.mkdir(exist_ok=False)
    try:
        for relative_value in _tracked_model_files(source):
            relative = PurePosixPath(relative_value)
            source_file = source.joinpath(*relative.parts)
            _assert_physical_path(
                source_file,
                require_file=True,
                label=f"tracked model file {relative_value}",
            )
            destination = snapshot.joinpath(*relative.parts)
            destination.parent.mkdir(parents=True, exist_ok=True)
            os.link(source_file, destination, follow_symlinks=False)
        identity = _snapshot_identity(config, source, snapshot)
        yield snapshot, identity
    finally:
        if _lexists(snapshot):
            if _is_reparse_or_symlink(snapshot):
                snapshot.unlink()
            else:
                shutil.rmtree(snapshot)


def _tensor_sha256(tensor: Any, torch: Any) -> str:
    value = tensor.detach().to(device="cpu").contiguous()
    raw = value.view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def _capture_base_parameters(
    model: Any, torch: Any
) -> tuple[list[tuple[str, Any, str]], str]:
    captured: list[tuple[str, Any, str]] = []
    aggregate = hashlib.sha256()
    for name, parameter in model.named_parameters():
        digest = _tensor_sha256(parameter, torch)
        captured.append((name, parameter, digest))
        aggregate.update(name.encode("utf-8"))
        aggregate.update(b"\0")
        aggregate.update(digest.encode("ascii"))
        aggregate.update(b"\n")
    if not captured:
        raise RuntimeError("base model exposes no parameters")
    return captured, aggregate.hexdigest()


def _assert_base_parameters_unchanged(
    captured: Sequence[tuple[str, Any, str]], torch: Any
) -> str:
    aggregate = hashlib.sha256()
    changed: list[str] = []
    for name, parameter, before in captured:
        after = _tensor_sha256(parameter, torch)
        if after != before:
            changed.append(name)
        aggregate.update(name.encode("utf-8"))
        aggregate.update(b"\0")
        aggregate.update(after.encode("ascii"))
        aggregate.update(b"\n")
    if changed:
        raise RuntimeError(
            "frozen base hash changed; first tensors: " + ", ".join(changed[:8])
        )
    return aggregate.hexdigest()


def _assert_q_proj_lora_trainable(
    model: Any, *, expected_layers: int, hidden_size: int, rank: int
) -> tuple[list[str], int]:
    names: list[str] = []
    count = 0
    observed: dict[tuple[int, str], str] = {}
    errors: list[str] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        names.append(name)
        count += int(parameter.numel())
        match = _LORA_TENSOR_RE.search(name)
        if match is None:
            errors.append(f"unexpected:{name}")
            continue
        layer = int(match.group(1))
        side = match.group(2)
        key = (layer, side)
        if key in observed:
            errors.append(f"duplicate:{name}")
        observed[key] = name
        expected_shape = (rank, hidden_size) if side == "A" else (hidden_size, rank)
        if tuple(int(value) for value in parameter.shape) != expected_shape:
            errors.append(f"shape:{name}={tuple(parameter.shape)}")
    if not names:
        raise RuntimeError("q_proj LoRA attachment produced zero trainable tensors")
    required = {
        (layer, side) for layer in range(expected_layers) for side in ("A", "B")
    }
    missing = sorted(required - set(observed))
    extra = sorted(set(observed) - required)
    if missing:
        errors.append(f"missing:{missing[:8]}")
    if extra:
        errors.append(f"extra:{extra[:8]}")
    if errors:
        raise RuntimeError(
            f"trainable scope/shape escaped the {expected_layers}-layer q_proj LoRA contract: "
            + ", ".join(errors[:8])
        )
    return names, count


def _assert_nonzero_lora_gradients(model: Any, torch: Any) -> dict[str, Any]:
    coverage = {
        "A": {"tensor_count": 0, "finite_count": 0, "nonzero_count": 0},
        "B": {"tensor_count": 0, "finite_count": 0, "nonzero_count": 0},
    }
    nonfinite: list[str] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        match = _LORA_TENSOR_RE.search(name)
        if match is None:
            raise RuntimeError(f"gradient scope escaped q_proj LoRA: {name}")
        side = match.group(2)
        coverage[side]["tensor_count"] += 1
        gradient = parameter.grad
        if gradient is not None and not bool(torch.isfinite(gradient).all().item()):
            nonfinite.append(name)
        elif gradient is not None:
            coverage[side]["finite_count"] += 1
            if int(torch.count_nonzero(gradient).item()) > 0:
                coverage[side]["nonzero_count"] += 1
    if nonfinite:
        raise RuntimeError(
            "q_proj LoRA tensors produced non-finite gradients: "
            + ", ".join(nonfinite[:8])
        )
    total_nonzero = sum(item["nonzero_count"] for item in coverage.values())
    if total_nonzero < 1:
        raise RuntimeError("q_proj LoRA tensors produced no nonzero gradient")
    return {"by_matrix": coverage, "total_nonzero_tensor_count": total_nonzero}


def _assert_trainable_parameters_finite(model: Any, torch: Any) -> None:
    nonfinite = [
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
        and not bool(torch.isfinite(parameter.detach()).all().item())
    ]
    if nonfinite:
        raise RuntimeError(
            "optimizer produced non-finite q_proj LoRA parameters: "
            + ", ".join(nonfinite[:8])
        )


def _normalize_saved_adapter_base_identity(path: Path) -> None:
    config_path = path / "adapter_config.json"
    if not config_path.is_file() or config_path.is_symlink():
        raise RuntimeError("saved adapter_config.json is missing or non-physical")
    adapter_config = json.loads(config_path.read_text("utf-8"))
    if not isinstance(adapter_config, dict):
        raise RuntimeError("saved adapter_config.json must contain an object")
    adapter_config["base_model_name_or_path"] = EXPECTED_MODEL_ID
    write_json(config_path, adapter_config)


def _validate_saved_adapter(
    path: Path, *, rank: int, expected_layers: int, hidden_size: int
) -> dict[str, str]:
    missing = [
        name
        for name in _ADAPTER_FILES
        if not (path / name).is_file() or (path / name).is_symlink()
    ]
    if missing:
        raise RuntimeError("saved adapter is incomplete: " + ", ".join(missing))
    adapter_config = json.loads((path / "adapter_config.json").read_text("utf-8"))
    if adapter_config.get("r") != rank:
        raise RuntimeError("saved adapter rank does not match the diagnostic config")
    if adapter_config.get("base_model_name_or_path") != EXPECTED_MODEL_ID:
        raise RuntimeError("saved adapter base model identity is not canonical")
    if adapter_config.get("target_modules") not in (["q_proj"], {"q_proj"}):
        raise RuntimeError("saved adapter target_modules escaped q_proj")
    try:
        from safetensors.numpy import load_file as load_safetensors

        state = load_safetensors(path / "adapter_model.safetensors")
    except (ImportError, OSError, ValueError) as exc:
        raise RuntimeError("saved adapter safetensors cannot be authenticated") from exc
    observed: set[tuple[int, str]] = set()
    tensor_errors: list[str] = []
    for name, value in state.items():
        match = _LORA_TENSOR_RE.search(name)
        if match is None:
            tensor_errors.append(f"unexpected:{name}")
            continue
        layer = int(match.group(1))
        side = match.group(2)
        key = (layer, side)
        if key in observed:
            tensor_errors.append(f"duplicate:{name}")
        observed.add(key)
        expected_shape = (rank, hidden_size) if side == "A" else (hidden_size, rank)
        if tuple(int(item) for item in value.shape) != expected_shape:
            tensor_errors.append(f"shape:{name}={tuple(value.shape)}")
    required = {
        (layer, side) for layer in range(expected_layers) for side in ("A", "B")
    }
    missing_tensors = sorted(required - observed)
    extra_tensors = sorted(observed - required)
    if missing_tensors:
        tensor_errors.append(f"missing:{missing_tensors[:8]}")
    if extra_tensors:
        tensor_errors.append(f"extra:{extra_tensors[:8]}")
    if tensor_errors:
        raise RuntimeError(
            "saved adapter tensor scope/shape mismatch: " + ", ".join(tensor_errors[:8])
        )
    return {
        name: hashlib.sha256((path / name).read_bytes()).hexdigest()
        for name in _ADAPTER_FILES
    }


@contextmanager
def _adapter_staging_directory(output_path: Path) -> Iterator[Path]:
    _assert_physical_path(output_path.parent, label="diagnostic output parent")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _assert_physical_path(
        output_path.parent,
        require_directory=True,
        label="diagnostic output parent",
    )
    staging = output_path.with_name(f".{output_path.name}.tmp-{uuid.uuid4().hex}")
    staging.mkdir(exist_ok=False)
    _assert_physical_path(
        staging, require_directory=True, label="adapter staging directory"
    )
    try:
        yield staging
    finally:
        if _lexists(staging):
            if _is_reparse_or_symlink(staging):
                staging.unlink()
            else:
                shutil.rmtree(staging)


def _rename_directory_noreplace(source: Path, destination: Path) -> None:
    if _lexists(destination):
        raise FileExistsError(f"diagnostic output already exists: {destination}")
    if os.name == "nt":
        os.rename(source, destination)
        return
    if sys.platform.startswith("linux"):
        libc = ctypes.CDLL(None, use_errno=True)
        renameat2 = getattr(libc, "renameat2", None)
        if renameat2 is None:
            raise RuntimeError("renameat2(RENAME_NOREPLACE) is unavailable")
        renameat2.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        renameat2.restype = ctypes.c_int
        if (
            renameat2(
                -100,
                os.fsencode(source),
                -100,
                os.fsencode(destination),
                1,
            )
            != 0
        ):
            error = ctypes.get_errno()
            raise OSError(error, os.strerror(error), str(destination))
        return
    raise RuntimeError("atomic no-replace directory publish is unsupported")


def _publish_verified_adapter(
    staging: Path,
    output_path: Path,
    *,
    expected_hashes: Mapping[str, str],
    rank: int,
    expected_layers: int,
    hidden_size: int,
) -> None:
    before_publish = _validate_saved_adapter(
        staging,
        rank=rank,
        expected_layers=expected_layers,
        hidden_size=hidden_size,
    )
    if dict(before_publish) != dict(expected_hashes):
        raise RuntimeError("staging adapter changed after reload verification")
    _rename_directory_noreplace(staging, output_path)
    try:
        final_hashes = _validate_saved_adapter(
            output_path,
            rank=rank,
            expected_layers=expected_layers,
            hidden_size=hidden_size,
        )
        if dict(final_hashes) != dict(expected_hashes):
            raise RuntimeError("published adapter bytes differ from verified staging")
    except Exception:
        if _lexists(output_path):
            if _is_reparse_or_symlink(output_path):
                output_path.unlink()
            else:
                shutil.rmtree(output_path)
        raise


def _toy_messages() -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    prompt = [
        {
            "role": "user",
            "content": "Return one JSON object with key status and value ready.",
        }
    ]
    completed = prompt + [{"role": "assistant", "content": '{"status":"ready"}'}]
    return prompt, completed


def _encoded_toy_batch(
    tokenizer: Any, *, sequence_length: int, torch: Any
) -> dict[str, Any]:
    prompt, completed = _toy_messages()
    prompt_text = tokenizer.apply_chat_template(
        prompt, tokenize=False, add_generation_prompt=True
    )
    completed_text = tokenizer.apply_chat_template(
        completed, tokenize=False, add_generation_prompt=False
    )
    encoded = tokenizer(
        completed_text,
        add_special_tokens=False,
        truncation=True,
        max_length=sequence_length,
        return_tensors="pt",
    )
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    labels = encoded["input_ids"].clone()
    boundary = min(len(prompt_ids), int(labels.shape[-1]))
    labels[:, :boundary] = -100
    if not torch.any(labels != -100):
        raise RuntimeError("toy sample contains no assistant target tokens")
    encoded["labels"] = labels
    return encoded


def _next_token_logits(model: Any, batch: Mapping[str, Any], torch: Any) -> Any:
    inputs = {key: value for key, value in batch.items() if key != "labels"}
    model.eval()
    with torch.inference_mode():
        logits = model(**inputs, use_cache=False).logits[0, -1].detach().float().cpu()
    if not bool(torch.isfinite(logits).all().item()):
        raise RuntimeError("next-token logits contain NaN or infinity")
    return logits


def execute_diagnostic(config: Mapping[str, Any]) -> dict[str, Any]:
    """Execute one local step; never contacts a hub and never reads project data."""

    preflight = build_preflight(config)
    if not preflight["ready"]:
        failed = [name for name, passed in preflight["gates"].items() if not passed]
        raise RuntimeError("diagnostic preflight failed: " + ", ".join(failed))

    output_path = _resolved_output(config)
    if _lexists(output_path):
        raise RuntimeError(f"diagnostic output already exists: {output_path}")
    with _authenticated_model_snapshot(config, output_path.parent) as (
        snapshot_path,
        snapshot_identity,
    ):
        return _execute_authenticated_diagnostic(
            config,
            preflight=preflight,
            source_model_path=_resolved_local_model(config),
            model_path=snapshot_path,
            snapshot_identity=snapshot_identity,
            output_path=output_path,
        )


def _execute_authenticated_diagnostic(
    config: Mapping[str, Any],
    *,
    preflight: Mapping[str, Any],
    source_model_path: Path,
    model_path: Path,
    snapshot_identity: Mapping[str, Any],
    output_path: Path,
) -> dict[str, Any]:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    import torch
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("the Qwen one-step diagnostic requires CUDA")
    torch.manual_seed(int(config["training"]["seed"]))
    torch.cuda.manual_seed_all(int(config["training"]["seed"]))
    kwargs = {
        "local_files_only": True,
        "trust_remote_code": False,
        "torch_dtype": torch.bfloat16,
        "attn_implementation": "sdpa",
        "low_cpu_mem_usage": True,
    }
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, local_files_only=True, trust_remote_code=False
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(model_path, **kwargs).to("cuda")
    post_load_snapshot_identity = _snapshot_identity(
        config, source_model_path, model_path
    )
    if dict(post_load_snapshot_identity) != dict(snapshot_identity):
        raise RuntimeError("authenticated model snapshot changed while loading")
    for parameter in base.parameters():
        parameter.requires_grad = False
    captured, base_hash_before = _capture_base_parameters(base, torch)
    base_tensor_count = len(captured)

    lora = config["lora"]
    model = get_peft_model(
        base,
        LoraConfig(
            r=int(lora["rank"]),
            lora_alpha=int(lora["alpha"]),
            lora_dropout=float(lora["dropout"]),
            target_modules=["q_proj"],
            bias="none",
            task_type="CAUSAL_LM",
        ),
    )
    trainable_names, trainable_parameters = _assert_q_proj_lora_trainable(
        model, expected_layers=28, hidden_size=1536, rank=int(lora["rank"])
    )
    batch = _encoded_toy_batch(
        tokenizer,
        sequence_length=int(config["training"]["sequence_length"]),
        torch=torch,
    )
    batch = {key: value.to("cuda") for key, value in batch.items()}
    model.train()
    model.config.use_cache = False
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=float(config["training"]["learning_rate"]),
    )
    optimizer.zero_grad(set_to_none=True)
    output = model(**batch, use_cache=False)
    loss = output.loss
    if not torch.isfinite(loss):
        raise RuntimeError("one-step diagnostic produced a non-finite loss")
    loss.backward()
    gradient_coverage = _assert_nonzero_lora_gradients(model, torch)
    optimizer.step()
    _assert_trainable_parameters_finite(model, torch)
    train_loss = float(loss.detach().cpu())
    base_hash_after = _assert_base_parameters_unchanged(captured, torch)
    trained_logits = _next_token_logits(model, batch, torch)
    if not hasattr(model, "disable_adapter"):
        raise RuntimeError("PEFT runtime does not expose an adapter-off context")
    with model.disable_adapter():
        adapter_off_logits = _next_token_logits(model, batch, torch)
    max_abs_adapter_effect = float(
        torch.max(torch.abs(adapter_off_logits - trained_logits)).item()
    )
    if not math.isfinite(max_abs_adapter_effect) or max_abs_adapter_effect <= 0:
        raise RuntimeError("one-step adapter produced no observable logits change")

    with _adapter_staging_directory(output_path) as staging:
        model.save_pretrained(staging, safe_serialization=True)
        _normalize_saved_adapter_base_identity(staging)
        artifact_hashes = _validate_saved_adapter(
            staging,
            rank=int(lora["rank"]),
            expected_layers=28,
            hidden_size=1536,
        )

        del output, loss, optimizer, model, base, captured
        gc.collect()
        torch.cuda.empty_cache()

        reload_base = AutoModelForCausalLM.from_pretrained(model_path, **kwargs).to(
            "cuda"
        )
        _, reloaded_base_hash = _capture_base_parameters(reload_base, torch)
        if reloaded_base_hash != base_hash_before:
            raise RuntimeError(
                "fresh reload base hash differs from the pre-step base hash"
            )
        reloaded = PeftModel.from_pretrained(
            reload_base, staging, is_trainable=False, local_files_only=True
        )
        reloaded_logits = _next_token_logits(reloaded, batch, torch)
        max_abs_reload_delta = float(
            torch.max(torch.abs(trained_logits - reloaded_logits)).item()
        )
        if not math.isfinite(max_abs_reload_delta) or max_abs_reload_delta > 1e-4:
            raise RuntimeError(
                "save/reload gate failed: next-token logits changed by "
                f"{max_abs_reload_delta}"
            )
        post_reload_hashes = _validate_saved_adapter(
            staging,
            rank=int(lora["rank"]),
            expected_layers=28,
            hidden_size=1536,
        )
        if post_reload_hashes != artifact_hashes:
            raise RuntimeError("adapter bytes changed while reload verification ran")

        receipt = {
            "schema_version": SCHEMA_VERSION,
            "status": "passed",
            "config_sha256": config_fingerprint(config),
            "global_step": 1,
            "loss": train_loss,
            "base_hash_before": base_hash_before,
            "base_hash_after": base_hash_after,
            "reloaded_base_hash": reloaded_base_hash,
            "trainable_tensor_names": trainable_names,
            "trainable_parameters": trainable_parameters,
            "base_tensor_count": base_tensor_count,
            "trainable_tensor_count": len(trainable_names),
            "gradient_coverage": gradient_coverage,
            "adapter_artifact_sha256": artifact_hashes,
            "max_abs_adapter_effect": max_abs_adapter_effect,
            "max_abs_reload_logit_delta": max_abs_reload_delta,
            "gates": {
                "base_hash_unchanged": base_hash_after == base_hash_before,
                "q_proj_lora_only": True,
                "one_step_completed": True,
                "adapter_effect_observed": True,
                "trainable_gradient_observed": True,
                "save_reload_equivalent": True,
                "toy_data_only_verified": True,
                "network_disabled": True,
            },
            "audit": {
                "dataset_source": "inline_toy_plumbing_v1",
                "external_dataset_inputs": 0,
                "provider_requests": 0,
            },
            "model_identity": {
                "model_id": EXPECTED_MODEL_ID,
                "model_config_sha256": preflight["model_config_sha256"],
                "source": dict(preflight["source_identity"]),
                "artifacts": dict(preflight["artifact_identity"]),
                "private_snapshot": dict(snapshot_identity),
            },
            "claim_scope": {
                "value": "mechanical_diagnostic_not_release_grade",
                "trusted_local_storage_required": True,
                "model_snapshot_mode": "private_hardlinks_same_inode",
            },
        }
        write_json(staging / "diagnostic_receipt.json", receipt)
        _publish_verified_adapter(
            staging,
            output_path,
            expected_hashes=artifact_hashes,
            rank=int(lora["rank"]),
            expected_layers=28,
            hidden_size=1536,
        )
        return receipt


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Isolated Qwen2.5-1.5B q_proj LoRA one-step diagnostic"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--rank", type=int, choices=(4, 8))
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--execute", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        config = load_qwen_diagnostic_config(args.config)
        if args.rank is not None:
            config["lora"]["rank"] = args.rank
            config["lora"]["alpha"] = args.rank * 2
            validate_qwen_diagnostic_config(config)
        preflight = build_preflight(config)
        report: dict[str, Any] = {
            **preflight,
            "mode": "execute" if args.execute else "dry-run",
            "executed": False,
        }
        if args.execute:
            report["result"] = execute_diagnostic(config)
            report["executed"] = True
        print(json.dumps(report, ensure_ascii=False, indent=2))
        if args.execute:
            return 0
        return 0 if preflight["ready"] else 3
    except (ConfigError, RuntimeError, OSError, ValueError) as exc:
        print(f"qwen diagnostic blocked: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

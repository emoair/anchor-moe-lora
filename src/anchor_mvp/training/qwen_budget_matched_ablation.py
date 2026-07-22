"""Budget-matched Qwen LoRA ablation over the synthetic scaffold fixture.

This runner is intentionally isolated from the frozen formal-v3 and q-only
diagnostic paths.  It executes three equal-parameter, diagnostic-only arms and
never grants formal training or held-out evaluation authority.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import re
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from . import qwen_lora_diagnostic as qdiag
from . import qwen_synthetic_scaffold_diagnostic as synth
from .config import ConfigError, _expand_env
from .manifest import config_fingerprint


SCHEMA_VERSION = "anchor.qwen25-1.5b-budget-matched-ablation.v1"
PREFLIGHT_VERSION = "anchor.qwen25-1.5b-budget-matched-ablation-preflight.v1"
CONFIG_VERSION = (
    "anchor.qwen25-1.5b-synthetic-scaffold-budget-matched-ablation-config.v1"
)
CONFIG_PATH = "configs/training/qwen2_5_1_5b_synthetic_scaffold_budget_matched_v1.yaml"
IMPLEMENTATION_PATH = "src/anchor_mvp/training/qwen_budget_matched_ablation.py"
PROFILES = ("q_only", "q_plus_o", "wide_budget_matched")
EXPECTED_PARAMETERS = 1_376_256
EXPECTED_LAYERS = 28
HIDDEN_SIZE = 1536
KV_SIZE = 256
_MAX_BYTES = 2_000_000
_LORA_RE = re.compile(
    r"(?:^|\.)model\.layers\.(\d+)\.self_attn\."
    r"(q_proj|k_proj|v_proj|o_proj)\.lora_([AB])(?:\.[^.]+)?\.weight$"
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
        )
        + "\n"
    ).encode("utf-8")


def _root() -> Path:
    return qdiag._project_root_from_module()


def _canonical_config(path: str | Path) -> Path:
    canonical = _root() / Path(CONFIG_PATH)
    requested = Path(path)
    resolved = requested if requested.is_absolute() else _root() / requested
    if os.path.normcase(str(resolved.resolve())) != os.path.normcase(
        str(canonical.resolve())
    ):
        raise ConfigError(f"config must remain exactly {CONFIG_PATH}")
    qdiag._assert_physical_path(canonical, require_file=True, label="ablation config")
    return canonical


def _profile(config: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    if name not in PROFILES:
        raise ConfigError(f"unsupported ablation profile: {name}")
    value = config["ablation"]["profiles"].get(name)
    if not isinstance(value, Mapping):
        raise ConfigError(f"missing ablation profile: {name}")
    return value


def load_config(path: str | Path) -> dict[str, Any]:
    canonical = _canonical_config(path)
    raw = canonical.read_bytes()
    if len(raw) > _MAX_BYTES or b"\r" in raw:
        raise ConfigError("ablation config must be small LF-only UTF-8 YAML")
    value = yaml.safe_load(raw.decode("utf-8"))
    if not isinstance(value, dict):
        raise ConfigError("ablation config must contain a mapping")
    value = _expand_env(value)
    value["_config_path"] = str(canonical)
    _validate_config(value)
    return value


def _validate_config(config: Mapping[str, Any]) -> None:
    if config.get("schema_version") != CONFIG_VERSION:
        raise ConfigError("ablation config version drifted")
    if config.get("claim_scope") != (
        "synthetic_diagnostic_ablation_only_no_formal_or_training_authority"
    ):
        raise ConfigError("ablation claim scope drifted")
    if config.get("paths") != {"project_root": "../.."}:
        raise ConfigError("ablation project root drifted")
    model = config.get("model")
    if not isinstance(model, Mapping):
        raise ConfigError("model must be a mapping")
    if (
        model.get("id") != qdiag.EXPECTED_MODEL_ID
        or model.get("local_files_only") is not True
        or model.get("allow_network") is not False
        or model.get("trust_remote_code") is not False
        or model.get("expected_source_revision") != qdiag.EXPECTED_SOURCE_REVISION
        or model.get("expected_source_repo") != qdiag.EXPECTED_SOURCE_REPO
    ):
        raise ConfigError("model identity/network contract drifted")
    dataset = config.get("dataset")
    if not isinstance(dataset, Mapping):
        raise ConfigError("dataset must be a mapping")
    if (
        dataset.get("expected_manifest_sha256")
        != "64b1ce813477deef48de16dbdc0d2561bbeaa0ef5d6248862e9f2bedc8acc0dd"
        or dataset.get("train_records") != 80
        or dataset.get("eval_proxy_records") != 20
        or dataset.get("formal_inputs_allowed") is not False
        or dataset.get("heldout_allowed") is not False
        or dataset.get("protected_source_paths_allowed") is not False
    ):
        raise ConfigError("synthetic dataset contract drifted")
    training = config.get("training")
    if not isinstance(training, Mapping) or training != {
        "optimizer_steps": 80,
        "batch_size": 1,
        "gradient_accumulation_steps": 1,
        "sequence_length": 512,
        "learning_rate": 0.00005,
        "seed": 1337,
        "gradient_checkpointing": True,
        "eval_before_after": True,
    }:
        raise ConfigError("training fairness contract drifted")
    precision = config.get("precision")
    if precision != {
        "compute_dtype": "bfloat16",
        "tf32": True,
        "float32_matmul_precision": "high",
    }:
        raise ConfigError("BF16/TF32 precision contract drifted")
    claims = config.get("claims")
    if claims != {
        "diagnostic_only": True,
        "training_authorized": False,
        "formal_training_authorized": False,
        "formal": False,
        "eval_proxy_is_heldout": False,
    }:
        raise ConfigError("diagnostic-only claims drifted")
    if (
        config.get("ablation", {}).get("budget_trainable_parameters")
        != EXPECTED_PARAMETERS
    ):
        raise ConfigError("ablation parameter budget drifted")
    expected = {
        "q_only": ({"q_proj": 16}, {"q_proj": 32}, 56),
        "q_plus_o": (
            {"q_proj": 8, "o_proj": 8},
            {"q_proj": 16, "o_proj": 16},
            112,
        ),
        "wide_budget_matched": (
            {"q_proj": 5, "o_proj": 4, "k_proj": 6, "v_proj": 6},
            {"q_proj": 10, "o_proj": 8, "k_proj": 12, "v_proj": 12},
            224,
        ),
    }
    for name, (ranks, alphas, tensors) in expected.items():
        item = _profile(config, name)
        targets = list(item.get("target_modules", []))
        base_rank = int(item.get("base_rank", 0))
        base_alpha = int(item.get("base_alpha", 0))
        observed_ranks = {
            target: int(item.get("rank_pattern", {}).get(target, base_rank))
            for target in targets
        }
        observed_alphas = {
            target: int(item.get("alpha_pattern", {}).get(target, base_alpha))
            for target in targets
        }
        if (
            observed_ranks != ranks
            or observed_alphas != alphas
            or int(item.get("expected_trainable_tensors", 0)) != tensors
            or any(
                observed_alphas[key] != 2 * rank for key, rank in observed_ranks.items()
            )
        ):
            raise ConfigError(f"profile {name} escaped exact-budget contract")
        if _expected_parameter_count(observed_ranks) != EXPECTED_PARAMETERS:
            raise ConfigError(f"profile {name} is not exactly budget matched")


def _ranks(profile: Mapping[str, Any]) -> dict[str, int]:
    base = int(profile["base_rank"])
    return {
        target: int(profile.get("rank_pattern", {}).get(target, base))
        for target in profile["target_modules"]
    }


def _alphas(profile: Mapping[str, Any]) -> dict[str, int]:
    base = int(profile["base_alpha"])
    return {
        target: int(profile.get("alpha_pattern", {}).get(target, base))
        for target in profile["target_modules"]
    }


def _expected_parameter_count(ranks: Mapping[str, int]) -> int:
    output = {
        "q_proj": HIDDEN_SIZE,
        "o_proj": HIDDEN_SIZE,
        "k_proj": KV_SIZE,
        "v_proj": KV_SIZE,
    }
    return EXPECTED_LAYERS * sum(
        rank * (HIDDEN_SIZE + output[module]) for module, rank in ranks.items()
    )


def _output_path(config: Mapping[str, Any], profile_name: str) -> Path:
    template = str(config["output"]["adapter_dir_template"])
    relative = template.format(profile=profile_name)
    candidate = (_root() / Path(relative)).resolve()
    diagnostics = (_root() / "artifacts" / "diagnostics").resolve()
    if diagnostics not in candidate.parents or candidate == diagnostics:
        raise ConfigError("ablation output escaped artifacts/diagnostics")
    if os.path.lexists(candidate):
        raise ConfigError(f"ablation output already exists: {candidate}")
    return candidate


def _base_config(config: Mapping[str, Any], output_path: Path) -> dict[str, Any]:
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
        "output": {"adapter_dir": output_path.relative_to(_root()).as_posix()},
        "_config_path": config["_config_path"],
    }


def _fixed_train_order(
    dataset: synth.ScaffoldDataset, seed: int
) -> tuple[synth.ScaffoldExample, ...]:
    ordered = tuple(
        sorted(
            dataset.train,
            key=lambda item: (
                _sha256(f"{seed}\0{item.record_id}".encode("utf-8")),
                item.record_id,
            ),
        )
    )
    ids = [item.record_id for item in ordered]
    if len(ids) != 80 or len(set(ids)) != 80:
        raise RuntimeError("fixed train order is not exactly 80 unique records")
    return ordered


def _ordered_training_digest(
    tokenizer: Any, ordered: Sequence[synth.ScaffoldExample], sequence_length: int
) -> dict[str, Any]:
    ids = [item.record_id for item in ordered]
    rows: list[dict[str, str]] = []
    for example in ordered:
        encoded = synth.tokenize_example(
            tokenizer, example, sequence_length=sequence_length
        )
        rows.append(
            {
                "record_id_sha256": _sha256(example.record_id.encode("utf-8")),
                "input_ids_sha256": synth._signed_int64_sequence_sha256(
                    encoded["input_ids"]
                ),
                "labels_sha256": synth._signed_int64_sequence_sha256(encoded["labels"]),
            }
        )
    return {
        "algorithm": "sha256_seeded_record_id_order_and_signed_int64_token_views_v1",
        "ordered_record_ids_sha256": _sha256(("\n".join(ids) + "\n").encode("utf-8")),
        "ordered_tokenized_examples_sha256": synth._compact_json_sha256(rows),
        "records": len(ids),
        "unique_records": len(set(ids)),
        "duplicates": len(ids) - len(set(ids)),
        "missing": 80 - len(set(ids)),
    }


def build_preflight(config: Mapping[str, Any], profile_name: str) -> dict[str, Any]:
    output = _output_path(config, profile_name)
    dataset = synth.load_dataset(config)
    base_config = _base_config(config, output)
    base_preflight = qdiag.build_preflight(base_config)
    tokenizer = synth._load_tokenizer_from_path(
        qdiag._resolved_local_model(base_config)
    )
    token_report = synth.token_length_preflight(
        tokenizer, dataset, sequence_length=int(config["training"]["sequence_length"])
    )
    ordered = _fixed_train_order(dataset, int(config["training"]["seed"]))
    order = _ordered_training_digest(
        tokenizer, ordered, int(config["training"]["sequence_length"])
    )
    profile = _profile(config, profile_name)
    ranks = _ranks(profile)
    alphas = _alphas(profile)
    config_bytes = Path(config["_config_path"]).read_bytes()
    implementation_bytes = (_root() / IMPLEMENTATION_PATH).read_bytes()
    relevant_base_gates = {
        key: bool(base_preflight["gates"][key])
        for key in (
            "network_disabled",
            "hf_directory_physical",
            "source_git_identity",
            "hf_config_closed_and_qwen2",
            "tokenizer_chat_template_present",
            "required_artifacts_physical",
            "expected_artifact_hashes_present",
            "artifact_hashes_match",
            "gguf_rejected",
            "output_scoped_and_absent",
        )
    }
    gates = {
        **relevant_base_gates,
        "dataset_authenticated": len(dataset.train) == 80
        and len(dataset.eval_proxy) == 20,
        "fixed_order_complete": order["unique_records"] == 80
        and order["duplicates"] == 0
        and order["missing"] == 0,
        "token_truncation_absent": token_report["truncated_records"] == 0,
        "parameter_budget_exact": _expected_parameter_count(ranks)
        == EXPECTED_PARAMETERS,
        "alpha_over_rank_two": all(
            alphas[key] == 2 * rank for key, rank in ranks.items()
        ),
        "diagnostic_only": True,
    }
    return {
        "schema_version": PREFLIGHT_VERSION,
        "status": "passed_diagnostic_preflight"
        if all(gates.values())
        else "blocked_diagnostic_preflight",
        "ready": all(gates.values()),
        "profile": profile_name,
        "config_sha256": config_fingerprint(config),
        "config_physical_sha256": _sha256(config_bytes),
        "implementation_sha256": _sha256(implementation_bytes),
        "dataset_manifest_sha256": dataset.manifest_sha256,
        "partition_sha256": dict(dataset.partition_sha256),
        "token_lengths": token_report,
        "train_order": order,
        "lora": {
            "ranks": ranks,
            "alphas": alphas,
            "target_modules": list(profile["target_modules"]),
            "expected_trainable_parameters": EXPECTED_PARAMETERS,
            "expected_trainable_tensors": int(profile["expected_trainable_tensors"]),
            "dropout": 0.0,
            "bias": "none",
            "use_rslora": False,
            "use_dora": False,
        },
        "model_identity": {
            "model_config_sha256": base_preflight["model_config_sha256"],
            "source": base_preflight["source_identity"],
            "artifacts": base_preflight["artifact_identity"],
        },
        "output_path": output.relative_to(_root()).as_posix(),
        "gates": gates,
        "claims": dict(config["claims"]),
        "audit": {
            "protected_body_reads": 0,
            "heldout_reads": 0,
            "provider_requests": 0,
            "network_requests": 0,
            "model_loads": 0,
            "gpu_requests": 0,
        },
    }


def publish_preflight(
    value: Mapping[str, Any], output_dir: str | Path
) -> tuple[Path, str]:
    directory = Path(output_dir)
    if not directory.is_absolute():
        directory = _root() / directory
    if os.path.lexists(directory):
        raise ConfigError(f"preflight output already exists: {directory}")
    directory.mkdir(parents=True, exist_ok=False)
    receipt = directory / "preflight.json"
    data = _canonical_json_bytes(value)
    digest = _sha256(data)
    receipt.write_bytes(data)
    (directory / "preflight.json.sha256").write_bytes(
        f"{digest}  preflight.json\n".encode("ascii")
    )
    return receipt, digest


def _authenticate_preflight(
    path: str | Path, digest: str, expected: Mapping[str, Any]
) -> Mapping[str, Any]:
    receipt = Path(path)
    if not receipt.is_absolute():
        receipt = _root() / receipt
    data = receipt.read_bytes()
    sidecar = receipt.with_name("preflight.json.sha256").read_bytes()
    if _sha256(data) != digest or sidecar != f"{digest}  preflight.json\n".encode(
        "ascii"
    ):
        raise ConfigError("preflight receipt authentication failed")
    value = json.loads(data.decode("utf-8"))
    if value != expected or not value.get("ready"):
        raise ConfigError("preflight receipt differs from current recomputation")
    return value


def _validate_trainable(
    model: Any, ranks: Mapping[str, int], expected_tensors: int
) -> tuple[list[str], int]:
    names: list[str] = []
    observed: set[tuple[int, str, str]] = set()
    count = 0
    errors: list[str] = []
    output = {
        "q_proj": HIDDEN_SIZE,
        "o_proj": HIDDEN_SIZE,
        "k_proj": KV_SIZE,
        "v_proj": KV_SIZE,
    }
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        names.append(name)
        count += int(parameter.numel())
        match = _LORA_RE.search(name)
        if match is None:
            errors.append(f"unexpected:{name}")
            continue
        layer, module, side = int(match.group(1)), match.group(2), match.group(3)
        if module not in ranks:
            errors.append(f"module:{name}")
            continue
        key = (layer, module, side)
        if key in observed:
            errors.append(f"duplicate:{name}")
        observed.add(key)
        rank = ranks[module]
        expected_shape = (rank, HIDDEN_SIZE) if side == "A" else (output[module], rank)
        if tuple(int(item) for item in parameter.shape) != expected_shape:
            errors.append(
                f"shape:{name}={tuple(parameter.shape)} expected={expected_shape}"
            )
    required = {
        (layer, module, side)
        for layer in range(EXPECTED_LAYERS)
        for module in ranks
        for side in ("A", "B")
    }
    if observed != required:
        errors.append(
            f"coverage:missing={len(required - observed)} extra={len(observed - required)}"
        )
    if len(names) != expected_tensors or count != EXPECTED_PARAMETERS:
        errors.append(f"budget:tensors={len(names)} params={count}")
    if errors:
        raise RuntimeError("LoRA trainable contract failed: " + ", ".join(errors[:8]))
    return names, count


def _gradient_step_coverage(model: Any, torch: Any, seen_nonzero: set[str]) -> None:
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        gradient = parameter.grad
        if gradient is None or not bool(torch.isfinite(gradient).all().item()):
            raise RuntimeError(f"missing/non-finite LoRA gradient: {name}")
        if int(torch.count_nonzero(gradient).item()) > 0:
            seen_nonzero.add(name)


def _eval_metrics(
    model: Any,
    tokenizer: Any,
    examples: Sequence[synth.ScaffoldExample],
    sequence_length: int,
    torch: Any,
) -> dict[str, Any]:
    model.eval()
    rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for example in examples:
            batch = synth._cuda_batch(tokenizer, example, sequence_length, torch)
            loss = model(**batch, use_cache=False).loss
            if not torch.isfinite(loss):
                raise RuntimeError("eval proxy produced non-finite loss")
            target_tokens = int(
                torch.count_nonzero(batch["labels"][:, 1:] != -100).item()
            )
            rows.append(
                {
                    "record_id": example.record_id,
                    "bundle": example.source_bundle_id,
                    "variant": example.variant,
                    "loss": float(loss.detach().cpu()),
                    "target_tokens": target_tokens,
                }
            )
    macro = sum(row["loss"] for row in rows) / len(rows)
    tokens = sum(row["target_tokens"] for row in rows)
    micro = sum(row["loss"] * row["target_tokens"] for row in rows) / tokens
    bundles: dict[str, list[float]] = {}
    for row in rows:
        bundles.setdefault(row["bundle"], []).append(row["loss"])
    return {
        "macro_loss": macro,
        "micro_target_token_nll": micro,
        "micro_target_token_ppl": math.exp(micro),
        "target_tokens": tokens,
        "bundle_macro_loss": {
            key: sum(values) / len(values) for key, values in sorted(bundles.items())
        },
        "records": len(rows),
    }


def _adapter_hashes(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for name in ("adapter_config.json", "adapter_model.safetensors"):
        candidate = path / name
        qdiag._assert_physical_path(candidate, require_file=True, label=name)
        result[name] = _sha256(candidate.read_bytes())
    return result


def execute(
    config: Mapping[str, Any],
    profile_name: str,
    preflight_path: str | Path,
    preflight_sha256: str,
) -> dict[str, Any]:
    preflight = build_preflight(config, profile_name)
    _authenticate_preflight(preflight_path, preflight_sha256, preflight)
    output_path = _output_path(config, profile_name)
    dataset = synth.load_dataset(config)
    ordered = _fixed_train_order(dataset, int(config["training"]["seed"]))
    base_config = _base_config(config, output_path)
    with qdiag._authenticated_model_snapshot(base_config, output_path.parent) as (
        model_path,
        snapshot_identity,
    ):
        return _execute_authenticated(
            config,
            profile_name,
            preflight,
            dataset,
            ordered,
            model_path,
            snapshot_identity,
            output_path,
        )


def _execute_authenticated(
    config: Mapping[str, Any],
    profile_name: str,
    preflight: Mapping[str, Any],
    dataset: synth.ScaffoldDataset,
    ordered: Sequence[synth.ScaffoldExample],
    model_path: Path,
    snapshot_identity: Mapping[str, Any],
    output_path: Path,
) -> dict[str, Any]:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    import torch
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoModelForCausalLM

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("budget-matched diagnostic requires BF16-capable CUDA")
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    seed = int(config["training"]["seed"])
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    kwargs = {
        "local_files_only": True,
        "trust_remote_code": False,
        "torch_dtype": torch.bfloat16,
        "attn_implementation": "sdpa",
        "low_cpu_mem_usage": True,
    }
    total_started = time.perf_counter()
    base = AutoModelForCausalLM.from_pretrained(model_path, **kwargs).to("cuda")
    for parameter in base.parameters():
        parameter.requires_grad = False
    base.config.use_cache = False
    base.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    if hasattr(base, "enable_input_require_grads"):
        base.enable_input_require_grads()
    captured, base_hash_before = qdiag._capture_base_parameters(base, torch)
    profile = _profile(config, profile_name)
    ranks = _ranks(profile)
    alphas = _alphas(profile)
    lora = LoraConfig(
        r=int(profile["base_rank"]),
        lora_alpha=int(profile["base_alpha"]),
        rank_pattern=dict(profile.get("rank_pattern", {})),
        alpha_pattern=dict(profile.get("alpha_pattern", {})),
        lora_dropout=0.0,
        target_modules=list(profile["target_modules"]),
        bias="none",
        task_type="CAUSAL_LM",
        use_rslora=False,
        use_dora=False,
    )
    model = get_peft_model(base, lora)
    trainable_names, trainable_parameters = _validate_trainable(
        model, ranks, int(profile["expected_trainable_tensors"])
    )
    sequence_length = int(config["training"]["sequence_length"])
    tokenizer = synth._load_tokenizer_from_path(model_path)
    if (
        _ordered_training_digest(tokenizer, ordered, sequence_length)
        != preflight["train_order"]
    ):
        raise RuntimeError("authenticated tokenizer changed fixed training views")
    probe = synth._cuda_batch(tokenizer, dataset.eval_proxy[0], sequence_length, torch)
    model.eval()
    initial_logits = qdiag._next_token_logits(model, probe, torch)
    with model.disable_adapter():
        initial_off_logits = qdiag._next_token_logits(model, probe, torch)
    step0_effect = float(
        torch.max(torch.abs(initial_logits - initial_off_logits)).item()
    )
    if not math.isfinite(step0_effect) or step0_effect > 1e-6:
        raise RuntimeError(f"fresh LoRA adapter is not a zero delta: {step0_effect}")
    eval_before = _eval_metrics(
        model, tokenizer, dataset.eval_proxy, sequence_length, torch
    )
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=float(config["training"]["learning_rate"]),
    )
    losses: list[float] = []
    seen_nonzero: set[str] = set()
    train_tokens = 0
    torch.cuda.reset_peak_memory_stats()
    train_started = time.perf_counter()
    for example in ordered:
        model.train()
        batch = synth._cuda_batch(tokenizer, example, sequence_length, torch)
        train_tokens += int(batch["attention_mask"].sum().item())
        optimizer.zero_grad(set_to_none=True)
        loss = model(**batch, use_cache=False).loss
        if not torch.isfinite(loss):
            raise RuntimeError("training produced non-finite loss")
        loss.backward()
        _gradient_step_coverage(model, torch, seen_nonzero)
        optimizer.step()
        qdiag._assert_trainable_parameters_finite(model, torch)
        losses.append(float(loss.detach().cpu()))
    torch.cuda.synchronize()
    train_seconds = time.perf_counter() - train_started
    peak_vram = int(torch.cuda.max_memory_allocated())
    if set(trainable_names) != seen_nonzero:
        missing = sorted(set(trainable_names) - seen_nonzero)
        raise RuntimeError(
            "some LoRA tensors never received nonzero gradients: "
            + ", ".join(missing[:8])
        )
    eval_after = _eval_metrics(
        model, tokenizer, dataset.eval_proxy, sequence_length, torch
    )
    base_hash_after = qdiag._assert_base_parameters_unchanged(captured, torch)
    trained_logits = qdiag._next_token_logits(model, probe, torch)
    with model.disable_adapter():
        adapter_off_logits = qdiag._next_token_logits(model, probe, torch)
    adapter_effect = float(
        torch.max(torch.abs(trained_logits - adapter_off_logits)).item()
    )
    if not math.isfinite(adapter_effect) or adapter_effect <= 0:
        raise RuntimeError("trained adapter produced no observable effect")
    with qdiag._adapter_staging_directory(output_path) as staging:
        model.save_pretrained(staging, safe_serialization=True)
        readme = staging / "README.md"
        if os.path.lexists(readme):
            readme.unlink()
        qdiag._normalize_saved_adapter_base_identity(staging)
        adapter_hashes = _adapter_hashes(staging)
        del optimizer, model, base, captured
        gc.collect()
        torch.cuda.empty_cache()
        reload_base = AutoModelForCausalLM.from_pretrained(model_path, **kwargs).to(
            "cuda"
        )
        _, reload_base_hash = qdiag._capture_base_parameters(reload_base, torch)
        if reload_base_hash != base_hash_before:
            raise RuntimeError("reloaded base hash differs from initial base")
        reloaded = PeftModel.from_pretrained(
            reload_base, staging, is_trainable=False, local_files_only=True
        )
        reload_logits = qdiag._next_token_logits(reloaded, probe, torch)
        reload_delta = float(
            torch.max(torch.abs(trained_logits - reload_logits)).item()
        )
        if not math.isfinite(reload_delta) or reload_delta > 1e-4:
            raise RuntimeError("save/reload logits gate failed")
        before_bundles = eval_before["bundle_macro_loss"]
        after_bundles = eval_after["bundle_macro_loss"]
        bundle_delta = {
            key: after_bundles[key] - before_bundles[key] for key in before_bundles
        }
        receipt = {
            "schema_version": SCHEMA_VERSION,
            "status": "passed_controlled_proxy_only",
            "profile": profile_name,
            "config_sha256": config_fingerprint(config),
            "preflight_sha256": _sha256(_canonical_json_bytes(preflight)),
            "dataset_manifest_sha256": dataset.manifest_sha256,
            "partition_sha256": dict(dataset.partition_sha256),
            "train_order": dict(preflight["train_order"]),
            "optimizer_steps": len(losses),
            "train_loss_first": losses[0],
            "train_loss_last": losses[-1],
            "eval_proxy_before": eval_before,
            "eval_proxy_after": eval_after,
            "eval_proxy_macro_loss_delta": eval_after["macro_loss"]
            - eval_before["macro_loss"],
            "eval_proxy_macro_loss_delta_percent": 100.0
            * (eval_after["macro_loss"] - eval_before["macro_loss"])
            / eval_before["macro_loss"],
            "bundle_macro_loss_delta": bundle_delta,
            "bundle_improved_count": sum(value < 0 for value in bundle_delta.values()),
            "lora": {
                "ranks": ranks,
                "alphas": alphas,
                "target_modules": list(profile["target_modules"]),
                "trainable_parameters": trainable_parameters,
                "trainable_tensor_count": len(trainable_names),
                "trainable_tensor_names": trainable_names,
                "all_tensors_nonzero_gradient_observed": True,
                "step0_adapter_effect": step0_effect,
                "trained_adapter_effect": adapter_effect,
            },
            "base_hash_before": base_hash_before,
            "base_hash_after": base_hash_after,
            "reloaded_base_hash": reload_base_hash,
            "max_abs_reload_logit_delta": reload_delta,
            "adapter_artifact_sha256": adapter_hashes,
            "performance": {
                "train_wall_seconds": train_seconds,
                "total_wall_seconds_before_receipt": time.perf_counter()
                - total_started,
                "train_full_tokens": train_tokens,
                "train_full_tokens_per_second": train_tokens / train_seconds,
                "peak_allocated_vram_bytes": peak_vram,
            },
            "precision": {
                "compute_dtype": "bfloat16",
                "tf32": True,
                "sequence_length": sequence_length,
                "gradient_checkpointing": True,
                "batch_size": 1,
                "gradient_accumulation_steps": 1,
            },
            "model_identity": {
                "preflight": preflight["model_identity"],
                "private_snapshot": dict(snapshot_identity),
            },
            "claims": {
                "diagnostic_only": True,
                "controlled_proxy_only": True,
                "training_authorized": False,
                "formal_training_authorized": False,
                "formal": False,
                "eval_proxy_is_heldout": False,
                "quality_validated": False,
                "statistical_significance_claimed": False,
            },
            "audit": {
                "protected_body_reads": 0,
                "heldout_reads": 0,
                "provider_requests": 0,
                "network_requests": 0,
                "model_loads": 2,
                "gpu_requests": 1,
            },
        }
        receipt_bytes = _canonical_json_bytes(receipt)
        receipt_sha = _sha256(receipt_bytes)
        (staging / "diagnostic_receipt.json").write_bytes(receipt_bytes)
        (staging / "diagnostic_receipt.json.sha256").write_bytes(
            f"{receipt_sha}  diagnostic_receipt.json\n".encode("ascii")
        )
        qdiag._rename_directory_noreplace(staging, output_path)
        return receipt


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Qwen2.5-1.5B exact-budget LoRA ablation"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--profile", required=True, choices=PROFILES)
    parser.add_argument("--preflight-output")
    parser.add_argument("--preflight-receipt")
    parser.add_argument("--preflight-receipt-sha256")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--execute", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    config = load_config(args.config)
    if args.dry_run:
        if (
            not args.preflight_output
            or args.preflight_receipt
            or args.preflight_receipt_sha256
        ):
            parser.error("--dry-run requires only --preflight-output")
        value = build_preflight(config, args.profile)
        path, digest = publish_preflight(value, args.preflight_output)
        print(
            json.dumps(
                {"preflight": str(path), "sha256": digest, "ready": value["ready"]},
                sort_keys=True,
            )
        )
        return 0 if value["ready"] else 2
    if (
        args.preflight_output
        or not args.preflight_receipt
        or not args.preflight_receipt_sha256
    ):
        parser.error("--execute requires preflight receipt+sha and forbids output")
    receipt = execute(
        config, args.profile, args.preflight_receipt, args.preflight_receipt_sha256
    )
    print(json.dumps(receipt, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

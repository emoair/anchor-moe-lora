#!/usr/bin/env python3
"""Export authenticated self-verified live checkpoints into formal training Gold."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from anchor_mvp.data.snapshot import SnapshotConfig  # noqa: E402
from anchor_mvp.swebench.formal_gold import (  # noqa: E402
    FormalGoldExportConfig,
    FormalGoldExportError,
    export_formal_gold,
)
from anchor_mvp.swebench.schema import canonical_json  # noqa: E402
from anchor_mvp.tooling.swebench_execution_v3 import (  # noqa: E402
    DISTILLATION_VALIDATOR_RESULT_SCHEMA,
    DISTILLATION_VALIDATOR_VERSION,
)
from anchor_mvp.tooling.swebench_runtime_v3 import (  # noqa: E402
    load_distillation_supervisor_receipt_key,
)
from anchor_mvp.training.manifest import sha256_file  # noqa: E402


def _mapping(path: Path, code: str) -> Mapping[str, Any]:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as exc:
        raise FormalGoldExportError(code) from exc
    if not isinstance(value, Mapping):
        raise FormalGoldExportError(code)
    return value


def _project_path(value: object, code: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise FormalGoldExportError(code)
    path = (PROJECT_ROOT / value).resolve()
    try:
        path.relative_to(PROJECT_ROOT)
    except ValueError as exc:
        raise FormalGoldExportError(code) from exc
    return path


def _nested(value: object, code: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise FormalGoldExportError(code)
    return value


def build_config(
    coordinator_path: Path,
    snapshot_path: Path,
    *,
    output_dir: Path | None,
    minimum_gold_per_stage: int,
) -> tuple[FormalGoldExportConfig, str]:
    coordinator = _mapping(coordinator_path, "formal_gold_coordinator_config_invalid")
    bank = _nested(coordinator.get("bank"), "formal_gold_coordinator_config_invalid")
    runtime = _nested(
        coordinator.get("runtime"), "formal_gold_coordinator_config_invalid"
    )
    execution = _nested(
        coordinator.get("execution_contract"),
        "formal_gold_coordinator_config_invalid",
    )
    expected_tasks = bank.get("expected_tasks")
    if (
        isinstance(expected_tasks, bool)
        or not isinstance(expected_tasks, int)
        or expected_tasks < 1
    ):
        raise FormalGoldExportError("formal_gold_coordinator_config_invalid")
    bank_root = _project_path(bank.get("root"), "formal_gold_bank_path_invalid")
    bank_manifest = _project_path(
        bank.get("manifest"), "formal_gold_bank_path_invalid"
    )
    runtime_root = _project_path(
        runtime.get("output_dir"), "formal_gold_runtime_path_invalid"
    )
    lock_path = _project_path(
        execution.get("lock"), "formal_gold_execution_lock_invalid"
    )
    lock_sha = str(execution.get("lock_sha256", ""))
    if sha256_file(lock_path) != lock_sha:
        raise FormalGoldExportError("formal_gold_execution_lock_invalid")
    config_sha = sha256_file(coordinator_path)
    binding = {
        "config_sha256": config_sha,
        "execution_lock_sha256": lock_sha,
        "source_bank_manifest_sha256": sha256_file(bank_manifest),
        "output_dir": runtime_root.relative_to(PROJECT_ROOT).as_posix(),
        "bank_manifest": bank_manifest.relative_to(PROJECT_ROOT).as_posix(),
        "expected_tasks": expected_tasks,
        "stages": [
            "planner",
            "tool_policy",
            "domain_builder",
            "domain_review",
            "security",
        ],
    }
    checkpoint_id = hashlib.sha256(
        canonical_json(binding).encode("utf-8")
    ).hexdigest()
    snapshot = SnapshotConfig.load(snapshot_path)
    if snapshot.formal_v3_split is None:
        raise FormalGoldExportError("formal_gold_split_metadata_missing")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if (
        not isinstance(lock, Mapping)
        or not isinstance(lock.get("runtime"), Mapping)
        or not isinstance(lock.get("validator"), Mapping)
    ):
        raise FormalGoldExportError("formal_gold_execution_lock_invalid")
    wsl_distro = str(lock["runtime"].get("wsl_distro", ""))
    validator = lock["validator"]
    validator_path = _project_path(
        validator.get("distillation_validator"),
        "formal_gold_execution_lock_invalid",
    )
    containerfile_path = _project_path(
        validator.get("distillation_containerfile"),
        "formal_gold_execution_lock_invalid",
    )
    validator_sha256 = str(validator.get("distillation_validator_sha256", ""))
    containerfile_sha256 = str(
        validator.get("distillation_containerfile_sha256", "")
    )
    validator_family = str(validator.get("distillation_validator_family", ""))
    image_reference = str(validator.get("distillation_image_reference", ""))
    image_id_sha256 = str(validator.get("distillation_image_id_sha256", ""))
    image_name, separator, image_digest = image_reference.rpartition("@")
    if (
        not image_name.startswith("localhost/anchor-train-sandbox")
        or separator != "@"
        or not image_digest.startswith("sha256:")
        or len(image_digest) != 71
        or any(
            character not in "0123456789abcdef"
            for character in image_digest.removeprefix("sha256:")
        )
        or sha256_file(validator_path) != validator_sha256
        or sha256_file(containerfile_path) != containerfile_sha256
        or validator.get("distillation_validator_version")
        != DISTILLATION_VALIDATOR_VERSION
        or validator.get("distillation_result_schema")
        != DISTILLATION_VALIDATOR_RESULT_SCHEMA
        or validator.get("distillation_allowed_actions") != ["compile", "test"]
        or not validator_family.strip()
        or len(image_id_sha256) != 64
        or any(character not in "0123456789abcdef" for character in image_id_sha256)
    ):
        raise FormalGoldExportError("formal_gold_execution_lock_invalid")
    resolved_output = (
        output_dir.resolve()
        if output_dir is not None
        else runtime_root / "training-export"
    )
    try:
        resolved_output.relative_to(PROJECT_ROOT)
    except ValueError as exc:
        raise FormalGoldExportError("formal_gold_output_path_invalid") from exc
    return (
        FormalGoldExportConfig(
            bank_root=bank_root,
            tasks_glob=str(bank.get("tasks_glob", "")),
            work_orders_glob=str(bank.get("work_orders_glob", "")),
            source_bank_manifest=bank_manifest,
            runtime_root=runtime_root,
            output_dir=resolved_output,
            coordinator_config_sha256=config_sha,
            checkpoint_id=checkpoint_id,
            execution_lock_sha256=lock_sha,
            train_sandbox_image_digest=image_digest,
            train_sandbox_image_id_sha256=image_id_sha256,
            validator_version_sha256=validator_sha256,
            validator_family=validator_family,
            heldout_manifest=snapshot.formal_v3_split.heldout_manifest,
            heldout_leak_audit=snapshot.formal_v3_split.heldout_leak_audit,
            minimum_gold_per_stage=minimum_gold_per_stage,
        ),
        wsl_distro,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Publish only HMAC-authenticated, real-sandbox self-verified "
            "five-stage chains without claiming official SWE-bench PASS; "
            "does not call a provider or start training"
        )
    )
    parser.add_argument(
        "--coordinator-config",
        default="configs/data/swebench_five_stage.ccswitch.yaml",
    )
    parser.add_argument(
        "--snapshot-config",
        default="configs/orchestration/full_v3_snapshot.yaml",
    )
    parser.add_argument("--output-dir")
    parser.add_argument("--minimum-gold-per-stage", type=int, default=256)
    args = parser.parse_args(argv)
    try:
        coordinator = _project_path(
            args.coordinator_config, "formal_gold_coordinator_config_invalid"
        )
        snapshot = _project_path(
            args.snapshot_config, "formal_gold_snapshot_config_invalid"
        )
        output = (
            _project_path(args.output_dir, "formal_gold_output_path_invalid")
            if args.output_dir
            else None
        )
        config, wsl_distro = build_config(
            coordinator,
            snapshot,
            output_dir=output,
            minimum_gold_per_stage=args.minimum_gold_per_stage,
        )
        # The protocol-separated root-owned WSL train key is read into this
        # process only. It is never placed in argv, environment, export files,
        # or logs.
        key = load_distillation_supervisor_receipt_key(wsl_distro)
        result = export_formal_gold(config, trusted_receipt_key=key)
    except (FormalGoldExportError, OSError, ValueError) as exc:
        code = exc.code if isinstance(exc, FormalGoldExportError) else "formal_gold_export_failed"
        print(json.dumps({"status": "blocked", "reason_code": code}, sort_keys=True))
        return 2
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from anchor_mvp.training.config import load_training_config  # noqa: E402
from anchor_mvp.training.formal_v3_schedule import (  # noqa: E402
    FORMAL_ARMS,
    derive_exposure_plan,
)
from anchor_mvp.training.preflight import (  # noqa: E402
    inspect_dataset_snapshot_manifest,
    inspect_gate_datasets,
)


def _project_root(config: dict[str, Any]) -> Path:
    config_path = Path(str(config["_config_path"]))
    return (
        config_path.parent / config.get("paths", {}).get("project_root", "../..")
    ).resolve()


def materialize(config_path: Path, arm: str, output: Path) -> dict[str, Any]:
    """Write a deterministic, snapshot-bound config; never starts a trainer."""

    if arm not in FORMAL_ARMS:
        raise ValueError(f"arm must be one of {FORMAL_ARMS}")
    config = load_training_config(config_path)
    root = _project_root(config)
    datasets = inspect_gate_datasets(config, root)
    snapshot = inspect_dataset_snapshot_manifest(config, root, datasets)
    if not snapshot.get("passed"):
        reasons = snapshot.get("errors", [])
        raise ValueError(
            "formal-v3 schedule requires a valid immutable split snapshot: "
            + "; ".join(str(item) for item in reasons)
        )
    split = snapshot.get("split_contract", {})
    train_counts = split.get("train_records_per_expert")
    if not isinstance(train_counts, dict):
        raise ValueError("formal-v3 snapshot did not expose balanced train counts")
    exposure = config["training"]["exposure_control"]
    plan = derive_exposure_plan(
        arm=arm,
        train_records_per_expert=train_counts,
        gradient_accumulation_steps=int(
            config["training"]["gradient_accumulation_steps"]
        ),
        epochs=int(exposure["epochs"]),
    )
    evaluation_contract = {
        "schema_version": "anchor.formal-v3-af-evaluation.v1",
        "heldout_ids_sha256": split.get("heldout_ids_sha256"),
        "heldout_manifest_sha256": split.get("heldout_manifest_sha256"),
        "leakage_audit_sha256": split.get("leakage_audit_sha256"),
        "normalization": "A_equals_100",
        "base_arm": "A",
        "arm_topology": {
            "A": "frozen_q4_base_no_adapter",
            "B": "single_mixed_adapter",
            "C": "five_stage_serial_hot_swap",
            "D": "five_stage_serial_hot_swap",
            "E": "five_stage_serial_hot_swap",
            "F": "five_stage_serial_hot_swap",
        },
        "artifact_isolation": "formal_v3_arm_and_version_scoped",
        "formal_v2_artifacts_allowed": False,
        "evaluation_executed": False,
    }

    resolved = {
        key: value for key, value in config.items() if not key.startswith("_")
    }
    resolved["training"]["max_steps"] = plan["max_steps_per_adapter_job"]
    # Four durable checkpoints per adapter job, rounded upward.  This replaces
    # the obsolete every-8/every-32 schedule from the 640-exposure experiment.
    resolved["training"]["save_steps"] = max(
        1, (plan["max_steps_per_adapter_job"] + 3) // 4
    )
    resolved["training"]["resolved_exposure"] = {
        **plan,
        "dataset_snapshot_sha256": snapshot["computed_snapshot_sha256"],
        "snapshot_manifest_sha256": snapshot["manifest_sha256"],
        "split_leakage_audit_passed": True,
        "heldout_content_read": False,
        "evaluation_contract": evaluation_contract,
    }
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        project_root = os.path.relpath(root, output.parent).replace("\\", "/")
    except ValueError:
        # A caller may audit materialization on a different Windows volume.
        # Checked-in launchers always emit under the project, but an absolute
        # path is still unambiguous and keeps the generated config runnable.
        project_root = str(root)
    resolved["paths"]["project_root"] = project_root
    encoded = (
        json.dumps(resolved, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_bytes(encoded)
    os.replace(temporary, output)
    digest = hashlib.sha256(encoded).hexdigest()
    Path(str(output) + ".sha256").write_text(
        f"{digest}  {output.name}\n", encoding="ascii", newline="\n"
    )
    return {
        "schema_version": "anchor.formal-v3-materialized-config.v1",
        "arm": arm,
        "output": str(output),
        "sha256": digest,
        "dataset_snapshot_sha256": snapshot["computed_snapshot_sha256"],
        "exposure_plan": plan,
        "evaluation_contract": evaluation_contract,
        "heldout_content_read": False,
        "training_started": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Materialize a snapshot-sized formal-v3 A-F training config"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--arm", choices=FORMAL_ARMS, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = materialize(args.config.resolve(), args.arm, args.output)
    except (OSError, ValueError) as exc:
        print(json.dumps({"ok": False, "error": str(exc)}), file=sys.stderr)
        return 2
    print(json.dumps({"ok": True, **result}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

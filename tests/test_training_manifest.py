from __future__ import annotations

from pathlib import Path

from anchor_mvp.training.config import load_training_config, select_adapter
from anchor_mvp.training.manifest import build_manifest


ROOT = Path(__file__).resolve().parents[1]
FORMAL_V2 = ROOT / "configs" / "training" / "formal_v2_lowmem_mixed.yaml"


def test_formal_v2_manifest_records_balanced_sample_exposure_plan() -> None:
    config = select_adapter(load_training_config(FORMAL_V2), "mixed_all", 16)
    manifest = build_manifest(
        config,
        dependency_report={"ready": True},
        datasets=[{"exists": True, "valid_records": 15} for _ in range(5)],
        mode="dry-run",
    )

    assert manifest["training_profile"]["runtime_engine"] == "manual_active_labels_v2"
    assert manifest["sample_exposure_plan"] == {
        "order": "deterministic_epoch_shuffle_v1",
        "dataset_records": 75,
        "sample_exposures": 300,
        "complete_epochs": 4,
        "balanced_complete_epochs": True,
    }
    assert manifest["safety_checkpoints"] == {
        "save_steps": 25,
        "resume_capability": "adapter_weights_warm_start_only",
        "optimizer_state_saved": False,
        "scheduler_state_saved": False,
        "rng_state_saved": False,
    }

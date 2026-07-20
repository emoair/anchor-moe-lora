from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from anchor_mvp.swebench.training_release import (  # noqa: E402
    TrainingReleaseError,
    freeze_generic_execution_contract,
    freeze_source_disjoint,
    freeze_training_release,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Freeze hash-only SWE-bench training release metadata"
    )
    commands = parser.add_subparsers(dest="command", required=True)

    generic = commands.add_parser("generic")
    generic.add_argument("--offline-preflight", type=Path, required=True)
    generic.add_argument("--offline-preflight-sha256", required=True)
    generic.add_argument("--execution-lock", type=Path, required=True)
    generic.add_argument("--execution-lock-sha256", required=True)
    generic.add_argument("--attestation", type=Path, required=True)
    generic.add_argument("--attestation-sha256", required=True)
    generic.add_argument("--coordinator-config", type=Path, required=True)
    generic.add_argument("--coordinator-config-sha256", required=True)
    generic.add_argument("--source-bank-manifest", type=Path, required=True)
    generic.add_argument("--source-bank-manifest-sha256", required=True)
    generic.add_argument("--output-dir", type=Path, required=True)

    source = commands.add_parser("source-disjoint")
    source.add_argument("--snapshot-dir", type=Path, required=True)
    source.add_argument("--snapshot-manifest-sha256", required=True)
    source.add_argument("--projector-dir", type=Path, required=True)
    source.add_argument("--projector-manifest-sha256", required=True)
    source.add_argument("--heldout-manifest", type=Path, required=True)
    source.add_argument("--heldout-manifest-sha256", required=True)
    source.add_argument("--output-dir", type=Path, required=True)

    release = commands.add_parser("release")
    release.add_argument("--projector-dir", type=Path, required=True)
    release.add_argument("--projector-manifest-sha256", required=True)
    release.add_argument("--source-disjoint-dir", type=Path, required=True)
    release.add_argument("--source-disjoint-manifest-sha256", required=True)
    release.add_argument("--generic-contract-dir", type=Path, required=True)
    release.add_argument("--generic-contract-sha256", required=True)
    release.add_argument("--consumer-contract", type=Path, required=True)
    release.add_argument("--consumer-contract-sha256", required=True)
    release.add_argument("--execution-lock", type=Path, required=True)
    release.add_argument("--execution-lock-sha256", required=True)
    release.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    try:
        if args.command == "generic":
            result = freeze_generic_execution_contract(
                args.offline_preflight,
                args.offline_preflight_sha256,
                args.execution_lock,
                args.execution_lock_sha256,
                args.attestation,
                args.attestation_sha256,
                args.coordinator_config,
                args.coordinator_config_sha256,
                args.source_bank_manifest,
                args.source_bank_manifest_sha256,
                args.output_dir,
            )
        elif args.command == "source-disjoint":
            result = freeze_source_disjoint(
                args.snapshot_dir,
                args.snapshot_manifest_sha256,
                args.projector_dir,
                args.projector_manifest_sha256,
                args.heldout_manifest,
                args.heldout_manifest_sha256,
                args.output_dir,
            )
        else:
            result = freeze_training_release(
                args.projector_dir,
                args.projector_manifest_sha256,
                args.source_disjoint_dir,
                args.source_disjoint_manifest_sha256,
                args.generic_contract_dir,
                args.generic_contract_sha256,
                args.consumer_contract,
                args.consumer_contract_sha256,
                args.execution_lock,
                args.execution_lock_sha256,
                args.output_dir,
            )
    except (OSError, TrainingReleaseError) as exc:
        code = exc.code if isinstance(exc, TrainingReleaseError) else "io_error"
        print(json.dumps({"ok": False, "error": code}), file=sys.stderr)
        return 2
    print(json.dumps({"ok": True, **result}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

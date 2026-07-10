from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path

from ..serving import ClientConfig, MockBackend, OpenAICompatibleClient
from .heldout import (
    HeldoutGateError,
    check_training_leakage,
    freeze_heldout_manifest,
    verify_heldout_manifest,
    verify_leak_audit,
    write_leak_audit,
)
from .heldout_eval import evaluate_heldout_records
from .heldout_mock import heldout_mock_handler
from .heldout_runner import HeldoutBenchmarkRunner
from .metrics import compute_metrics
from .models import load_cases_jsonl, load_specs, write_records_jsonl
from .report import generate_report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Frozen five-stage held-out benchmark")
    subparsers = parser.add_subparsers(dest="command", required=True)

    freeze = subparsers.add_parser("freeze", help="Freeze case and fixture hashes")
    _integrity_arguments(freeze, include_audit=False)

    check = subparsers.add_parser("check-leakage", help="Run the local-only leakage gate")
    _integrity_arguments(check)
    check.add_argument("--training-jsonl", action="append", default=[])
    check.add_argument("--sop-source", action="append", default=[])
    check.add_argument("--threshold", type=float, default=0.86)

    verify = subparsers.add_parser("verify", help="Verify the pre-bulk freeze and leak gates")
    _integrity_arguments(verify)

    run = subparsers.add_parser("run", help="Run a live five-stage benchmark")
    _run_arguments(run)
    run.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    run.add_argument("--api-key-env", default="ANCHOR_VLLM_API_KEY")

    mock = subparsers.add_parser("mock-e2e", help="Run a deterministic no-network E2E")
    _run_arguments(mock)
    mock.add_argument("--output-dir", required=True)

    evaluate = subparsers.add_parser("evaluate", help="Run trusted sandbox validation")
    _integrity_arguments(evaluate)
    evaluate.add_argument("--records", required=True)
    evaluate.add_argument("--output", required=True)
    evaluate.add_argument("--workspace-root", required=True)
    evaluate.add_argument("--keep-workspaces", action="store_true")
    return parser


def _integrity_arguments(parser: argparse.ArgumentParser, *, include_audit: bool = True) -> None:
    parser.add_argument("--cases", required=True)
    parser.add_argument("--fixtures-root", required=True)
    parser.add_argument("--manifest", required=True)
    if include_audit:
        parser.add_argument("--leak-audit", required=True)


def _run_arguments(parser: argparse.ArgumentParser) -> None:
    _integrity_arguments(parser)
    parser.add_argument("--specs", required=True)
    parser.add_argument("--output", required=False)
    parser.add_argument("--metrics", required=False)
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    parser.add_argument("--max-attempts", type=int, default=2)
    parser.add_argument("--no-vram", action="store_true")


async def _run_live(args: argparse.Namespace) -> None:
    manifest_digest = verify_heldout_manifest(args.cases, args.fixtures_root, args.manifest)
    verify_leak_audit(args.leak_audit, manifest_digest)
    backend = OpenAICompatibleClient(
        ClientConfig(
            base_url=args.base_url,
            api_key=os.environ.get(args.api_key_env),
            timeout_seconds=args.timeout_seconds,
            max_attempts=args.max_attempts,
        )
    )
    records = await HeldoutBenchmarkRunner(
        backend,
        timeout_seconds=args.timeout_seconds,
        max_attempts=args.max_attempts,
        sample_vram=not args.no_vram,
        backend_label="vllm-heldout",
        manifest_sha256=manifest_digest,
        require_verified_q4=True,
    ).run_suite(load_specs(args.specs), load_cases_jsonl(args.cases))
    if not args.output or not args.metrics:
        raise HeldoutGateError("run requires --output and --metrics")
    _write_run(records, Path(args.output), Path(args.metrics))


async def _run_mock(args: argparse.Namespace) -> None:
    manifest_digest = verify_heldout_manifest(args.cases, args.fixtures_root, args.manifest)
    verify_leak_audit(args.leak_audit, manifest_digest)
    specs = load_specs(args.specs)
    model_ids = {model for spec in specs for model in spec.stage_models.values()}
    backend = MockBackend(handlers={model: heldout_mock_handler for model in model_ids})
    records = await HeldoutBenchmarkRunner(
        backend,
        timeout_seconds=args.timeout_seconds,
        max_attempts=args.max_attempts,
        sample_vram=False,
        backend_label="mock-no-network-five-stage",
        manifest_sha256=manifest_digest,
    ).run_suite(specs, load_cases_jsonl(args.cases))
    destination = Path(args.output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    raw = destination / "records.raw.jsonl"
    write_records_jsonl(records, raw)
    evaluated = destination / "records.evaluated.jsonl"
    evaluated_records = evaluate_heldout_records(
        raw,
        args.cases,
        args.fixtures_root,
        args.manifest,
        args.leak_audit,
        destination / "workspaces",
        evaluated,
    )
    metrics_path = destination / "metrics.json"
    metrics_path.write_text(
        json.dumps(compute_metrics(evaluated_records), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    generate_report(evaluated, metrics_path, destination / "report")


def _write_run(records: list, output: Path, metrics: Path) -> None:
    write_records_jsonl(records, output)
    metrics.parent.mkdir(parents=True, exist_ok=True)
    metrics.write_text(
        json.dumps(compute_metrics(records), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = build_parser().parse_args()
    try:
        if args.command == "freeze":
            freeze_heldout_manifest(args.cases, args.fixtures_root, args.manifest)
        elif args.command == "check-leakage":
            report = check_training_leakage(
                args.cases,
                args.fixtures_root,
                args.manifest,
                args.training_jsonl,
                args.sop_source,
                similarity_threshold=args.threshold,
            )
            write_leak_audit(report, args.leak_audit)
            if report["status"] != "PASS":
                raise HeldoutGateError("leakage collision detected")
        elif args.command == "run":
            asyncio.run(_run_live(args))
        elif args.command == "verify":
            digest = verify_heldout_manifest(args.cases, args.fixtures_root, args.manifest)
            verify_leak_audit(args.leak_audit, digest)
        elif args.command == "mock-e2e":
            asyncio.run(_run_mock(args))
        elif args.command == "evaluate":
            evaluate_heldout_records(
                args.records,
                args.cases,
                args.fixtures_root,
                args.manifest,
                args.leak_audit,
                args.workspace_root,
                args.output,
                keep_workspaces=args.keep_workspaces,
            )
    except HeldoutGateError as exc:
        raise SystemExit(f"held-out gate failed: {exc}") from exc


if __name__ == "__main__":
    main()

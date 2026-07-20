from __future__ import annotations

import argparse
import asyncio
from hashlib import sha256
import json
import os
from pathlib import Path
import urllib.error
import urllib.parse
import urllib.request

from ..serving import (
    ClientConfig,
    MockBackend,
    OpenAICompatibleClient,
    RuntimeAdapterAdmin,
)
from .formal_af_preflight import (
    SCHEMA as FORMAL_AF_SCHEMA,
    FormalAFPreflightError,
    preflight as formal_af_preflight,
)
from .formal_v3_preflight import (
    BENCHMARK_SCHEMA as FORMAL_V3_AF_SCHEMA,
    FormalV3PreflightError,
    preflight as formal_v3_af_preflight,
)
from .formal_checkpoint import (
    FormalCheckpointBindings,
    FormalCheckpointError,
    FormalRunCheckpoint,
)
from .heldout import (
    HeldoutGateError,
    check_training_leakage,
    freeze_heldout_manifest,
    verify_heldout_manifest,
    verify_leak_audit,
    write_leak_audit,
    file_sha256,
)
from .heldout_eval import evaluate_heldout_records
from .heldout_mock import heldout_mock_handler
from .heldout_runner import HeldoutBenchmarkRunner
from .metrics import compute_metrics
from .models import load_cases_jsonl, load_specs, write_records_jsonl
from .report import generate_report
from .serial_backend import (
    SerialBackendError,
    SerialLoraBackend,
    require_loopback_http_url,
)


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

    formal = subparsers.add_parser(
        "formal-run",
        help="Run the formal A--F live benchmark, sandbox evaluation, metrics and report",
    )
    _benchmark_arguments(formal)
    formal.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    formal.add_argument("--api-key-env", default="ANCHOR_VLLM_API_KEY")
    formal.add_argument("--output-dir", required=True)
    formal.add_argument("--keep-workspaces", action="store_true")
    formal.add_argument(
        "--resume",
        action="store_true",
        help="resume an exactly matching atomic formal-run checkpoint",
    )
    formal.add_argument(
        "--serial-runtime-lora",
        action="store_true",
        help="keep the base resident and dynamically load at most one formal LoRA",
    )
    formal.add_argument(
        "--admin-base-url",
        default="http://127.0.0.1:8000",
        help="local vLLM runtime-LoRA admin endpoint (serial mode only)",
    )
    formal.add_argument(
        "--server-project-root",
        default="",
        help="project root as seen by the model server, for example /mnt/d/LLM/anchor-moe-lora",
    )

    evaluate = subparsers.add_parser("evaluate", help="Run trusted sandbox validation")
    _integrity_arguments(evaluate)
    evaluate.add_argument("--records", required=True)
    evaluate.add_argument("--output", required=True)
    evaluate.add_argument("--workspace-root", required=True)
    evaluate.add_argument("--keep-workspaces", action="store_true")
    _heldout_authorization_argument(evaluate)
    return parser


def _integrity_arguments(parser: argparse.ArgumentParser, *, include_audit: bool = True) -> None:
    parser.add_argument("--cases", required=True)
    parser.add_argument("--fixtures-root", required=True)
    parser.add_argument("--manifest", required=True)
    if include_audit:
        parser.add_argument("--leak-audit", required=True)


def _run_arguments(parser: argparse.ArgumentParser) -> None:
    _benchmark_arguments(parser)
    parser.add_argument("--output", required=False)
    parser.add_argument("--metrics", required=False)


def _benchmark_arguments(parser: argparse.ArgumentParser) -> None:
    _integrity_arguments(parser)
    parser.add_argument("--specs", required=True)
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    parser.add_argument("--max-attempts", type=int, default=2)
    parser.add_argument("--no-vram", action="store_true")
    parser.add_argument(
        "--project-root",
        default=".",
        help="project root used by the formal A--F registry preflight",
    )
    _heldout_authorization_argument(parser)


def _heldout_authorization_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--authorize-heldout-access",
        action="store_true",
        help="explicitly authorize this command to open frozen held-out cases",
    )


def _require_heldout_authorization(args: argparse.Namespace) -> None:
    if not getattr(args, "authorize_heldout_access", False):
        raise HeldoutGateError(
            "explicit held-out access authorization is required; "
            "pass --authorize-heldout-access"
        )


def _enforce_formal_af_preflight(
    args: argparse.Namespace, *, require_formal: bool = False
) -> dict | None:
    """Fail before opening held-out cases when a formal A--F spec is not ready."""

    try:
        payload = json.loads(Path(args.specs).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HeldoutGateError("benchmark spec is missing or invalid") from exc
    schema = payload.get("schema_version")
    if schema not in {FORMAL_AF_SCHEMA, FORMAL_V3_AF_SCHEMA}:
        if require_formal:
            raise HeldoutGateError("formal-run requires the frozen formal A--F spec")
        return None
    try:
        if schema == FORMAL_V3_AF_SCHEMA:
            return formal_v3_af_preflight(args.specs, args.project_root)
        return formal_af_preflight(args.specs, args.project_root)
    except (FormalAFPreflightError, FormalV3PreflightError) as exc:
        raise HeldoutGateError(
            f"formal A--F preflight blocked ({exc.code}): {exc}"
        ) from exc


def _verify_model_catalog(
    base_url: str,
    api_key: str | None,
    required_models: set[str],
    timeout_seconds: float,
) -> list[str]:
    """Fail before held-out access unless the backend exposes every frozen model id."""

    headers: dict[str, str] = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/models", headers=headers, method="GET"
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (OSError, urllib.error.URLError, json.JSONDecodeError) as exc:
        raise HeldoutGateError(
            "formal backend model-catalog preflight failed before held-out access"
        ) from exc
    items = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(items, list):
        raise HeldoutGateError("formal backend returned an invalid model catalog")
    available = {
        str(item.get("id")) for item in items if isinstance(item, dict) and item.get("id")
    }
    missing = sorted(required_models - available)
    if missing:
        raise HeldoutGateError(
            "formal backend is missing frozen model ids: " + ", ".join(missing)
        )
    return sorted(available)


def _verify_local_serial_endpoints(base_url: str, admin_base_url: str) -> None:
    """Require the completion and runtime-admin surfaces to be one local server."""

    origins: list[tuple[str, str, int | None]] = []
    for label, raw in (("completion", base_url), ("runtime-LoRA admin", admin_base_url)):
        try:
            require_loopback_http_url(raw, label=f"formal {label}")
            parsed = urllib.parse.urlsplit(raw)
            port = parsed.port
        except (SerialBackendError, ValueError) as exc:
            raise HeldoutGateError(f"formal {label} URL is invalid") from exc
        assert parsed.hostname is not None  # guaranteed by require_loopback_http_url
        origins.append((parsed.scheme, parsed.hostname, port))
        normalized_path = parsed.path.rstrip("/")
        if label == "completion" and normalized_path != "/v1":
            raise HeldoutGateError("formal completion URL must end at the local /v1 root")
        if label == "runtime-LoRA admin" and normalized_path:
            raise HeldoutGateError("formal runtime-LoRA admin URL must not include a path")
    if origins[0] != origins[1]:
        raise HeldoutGateError(
            "formal completion and runtime-LoRA admin URLs must share one local origin"
        )


async def _run_live(args: argparse.Namespace) -> None:
    _require_heldout_authorization(args)
    if not args.output or not args.metrics:
        raise HeldoutGateError("run requires --output and --metrics")
    _enforce_formal_af_preflight(args)
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
    _write_run(records, Path(args.output), Path(args.metrics))


async def _run_mock(args: argparse.Namespace) -> None:
    _require_heldout_authorization(args)
    _enforce_formal_af_preflight(args)
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


async def _run_formal_live(args: argparse.Namespace) -> None:
    """Execute the complete formal run without silently opening held-out data."""

    _require_heldout_authorization(args)
    destination = Path(args.output_dir)
    if destination.exists():
        if not destination.is_dir():
            raise HeldoutGateError("formal output directory is not a directory")
        if any(destination.iterdir()) and not getattr(args, "resume", False):
            raise HeldoutGateError("formal output directory must be new or empty")
    elif getattr(args, "resume", False):
        raise HeldoutGateError("formal --resume requires an existing checkpoint directory")
    preflight_result = _enforce_formal_af_preflight(args, require_formal=True)
    if preflight_result is None:  # pragma: no cover - guarded by require_formal
        raise HeldoutGateError("formal A--F preflight returned no execution contract")
    if preflight_result.get("formal_version") == "formal-v3":
        project_root = Path(args.project_root).resolve()
        namespace = preflight_result.get("output_namespace")
        if not isinstance(namespace, str) or not namespace:
            raise HeldoutGateError("formal-v3 output namespace is missing")
        namespace_root = (project_root / namespace).resolve()
        resolved_destination = destination.resolve()
        try:
            relative_output = resolved_destination.relative_to(namespace_root)
        except ValueError as exc:
            raise HeldoutGateError(
                "formal-v3 output must remain inside its version-isolated namespace"
            ) from exc
        if relative_output == Path("."):
            raise HeldoutGateError(
                "formal-v3 output must be a child of its version namespace"
            )
    serial_contract = preflight_result["serial_runtime_contract"]
    if (
        not args.serial_runtime_lora
        and serial_contract.get("allow_static_lora_modules") is False
    ):
        raise HeldoutGateError(
            "the frozen formal contract requires explicit --serial-runtime-lora"
        )

    api_key = os.environ.get(args.api_key_env)
    client = OpenAICompatibleClient(
        ClientConfig(
            base_url=args.base_url,
            api_key=api_key,
            timeout_seconds=args.timeout_seconds,
            max_attempts=args.max_attempts,
        )
    )
    serial: SerialLoraBackend | None = None
    if args.serial_runtime_lora:
        _verify_local_serial_endpoints(args.base_url, args.admin_base_url)
        try:
            serial = SerialLoraBackend(
                client,
                RuntimeAdapterAdmin(args.admin_base_url, api_key),
                preflight_result["runtime_bindings"],
                project_root=args.project_root,
                server_project_root=args.server_project_root or None,
            )
        except SerialBackendError as exc:
            raise HeldoutGateError(
                "serial runtime-LoRA binding validation failed before held-out access"
            ) from exc
        backend = serial
        required_models = {serial.base_model_id}
        backend_label = "formal-af-serial-runtime-lora"
    else:
        backend = client
        required_models = {
            str(binding["model_id"])
            for group in preflight_result["runtime_bindings"].values()
            for binding in group.values()
        }
        backend_label = "formal-af-static-catalog"
    catalog_models = await asyncio.to_thread(
        _verify_model_catalog,
        args.base_url,
        api_key,
        required_models,
        args.timeout_seconds,
    )
    if serial is not None:
        try:
            await serial.probe()
        except SerialBackendError as exc:
            raise HeldoutGateError(
                "serial runtime-LoRA probe failed before held-out access"
            ) from exc

    # Only after the registry and backend gates pass may this command open cases.
    try:
        manifest_digest = verify_heldout_manifest(
            args.cases, args.fixtures_root, args.manifest
        )
        verify_leak_audit(args.leak_audit, manifest_digest)
        specs = load_specs(args.specs)
        cases = load_cases_jsonl(args.cases)
        backend_identity = {
            "backend_label": backend_label,
            "transport": "openai-compatible-loopback",
            "completion_base_url": args.base_url.rstrip("/"),
            "admin_base_url": (
                args.admin_base_url.rstrip("/") if args.serial_runtime_lora else None
            ),
            "required_model_ids": sorted(required_models),
            "validated_catalog_model_ids_sha256": sha256(
                json.dumps(catalog_models, separators=(",", ":")).encode("utf-8")
            ).hexdigest(),
            "runtime_bindings_sha256": sha256(
                json.dumps(
                    preflight_result["runtime_bindings"],
                    ensure_ascii=False,
                    separators=(",", ":"),
                    sort_keys=True,
                ).encode("utf-8")
            ).hexdigest(),
        }
        execution_options = {
            "timeout_seconds": args.timeout_seconds,
            "max_attempts": args.max_attempts,
            "sample_vram": not args.no_vram,
            "serial_runtime_lora": bool(args.serial_runtime_lora),
            "server_project_root": args.server_project_root or None,
        }
        try:
            checkpoint = FormalRunCheckpoint.open(
                destination,
                resume=bool(getattr(args, "resume", False)),
                bindings=FormalCheckpointBindings(
                    config_sha256=preflight_result["config_sha256"],
                    execution_contract_sha256=preflight_result.get(
                        "execution_contract_sha256",
                        preflight_result["config_sha256"],
                    ),
                    run_manifest_sha256=preflight_result["run_manifest_sha256"],
                    case_manifest_sha256=manifest_digest,
                    leak_audit_sha256=file_sha256(args.leak_audit),
                    backend_identity=backend_identity,
                    execution_options=execution_options,
                ),
                specs=specs,
                cases=cases,
                backend_label=backend_label,
            )
        except FormalCheckpointError as exc:
            raise HeldoutGateError(f"formal checkpoint blocked: {exc}") from exc
        try:
            await HeldoutBenchmarkRunner(
                backend,
                timeout_seconds=args.timeout_seconds,
                max_attempts=args.max_attempts,
                sample_vram=not args.no_vram,
                backend_label=backend_label,
                manifest_sha256=manifest_digest,
                require_verified_q4=True,
            ).run_suite(
                specs,
                cases,
                completed_records=checkpoint.records,
                record_callback=checkpoint.commit,
            )
        except FormalCheckpointError as exc:
            raise HeldoutGateError(f"formal checkpoint commit failed: {exc}") from exc
    finally:
        if serial is not None:
            try:
                await serial.close()
            except SerialBackendError as exc:
                raise HeldoutGateError("serial runtime-LoRA cleanup failed") from exc

    destination.mkdir(parents=True, exist_ok=True)
    runtime_backend = (
        serial.stats
        if serial is not None
        else {"mode": "static_catalog", "required_model_count": len(required_models)}
    )
    runtime_backend["latency_accounting"] = (
        "previous-arm cleanup excluded; target-arm activation included"
    )
    preflight_result["runtime_backend"] = runtime_backend
    (destination / "preflight.json").write_text(
        json.dumps(preflight_result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    raw = destination / "records.raw.jsonl"
    evaluated = destination / "records.evaluated.jsonl"
    evaluated_records = evaluate_heldout_records(
        raw,
        args.cases,
        args.fixtures_root,
        args.manifest,
        args.leak_audit,
        destination / "workspaces",
        evaluated,
        keep_workspaces=args.keep_workspaces,
    )
    metrics_path = destination / "metrics.json"
    metrics_path.write_text(
        json.dumps(compute_metrics(evaluated_records), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    report = generate_report(evaluated, metrics_path, destination / "report")
    try:
        checkpoint.mark_complete()
    except FormalCheckpointError as exc:  # pragma: no cover - generation invariant
        raise HeldoutGateError(f"formal checkpoint completion failed: {exc}") from exc
    print(
        json.dumps(
            {
                "status": "complete",
                "heldout_access_authorized": True,
                "preflight": str(destination / "preflight.json"),
                "records": str(evaluated),
                "metrics": str(metrics_path),
                "report": str(report.summary),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


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
        elif args.command == "formal-run":
            asyncio.run(_run_formal_live(args))
        elif args.command == "verify":
            digest = verify_heldout_manifest(args.cases, args.fixtures_root, args.manifest)
            verify_leak_audit(args.leak_audit, digest)
        elif args.command == "mock-e2e":
            asyncio.run(_run_mock(args))
        elif args.command == "evaluate":
            _require_heldout_authorization(args)
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

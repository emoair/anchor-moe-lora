from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import tempfile


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from anchor_mvp.tooling import (  # noqa: E402
    AnchorSandboxOptions,
    LiveBatchConfig,
    ControlledSessionCapture,
    OpenCodeExecutor,
    KimiRoutePlan,
    RouteMode,
    RoutePreflightError,
    SampleSpec,
    SkillSourceRegistry,
    ToolPolicy,
    ToolingHarness,
    batch_run_succeeded,
    load_candidate_samples,
    merge_attempts_jsonl,
    persist_attempts_and_gold,
    prepare_kimi_route_plan,
    run_live_batch,
    verify_execution_split,
    write_opencode_config,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run audited isolated OpenCode gold samples")
    parser.add_argument(
        "--batch-config",
        type=Path,
        help="Audited optional-stage batch configuration; mutually exclusive with single mode",
    )
    parser.add_argument(
        "--session-candidates",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "tooling" / "session_candidates.jsonl",
    )
    parser.add_argument(
        "--session-quarantine",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "tooling" / "session_quarantine.jsonl",
    )
    parser.add_argument(
        "--session-staging",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "tooling" / "session_staging.raw.jsonl",
        help="Safe collect-first trajectories awaiting offline partitioning",
    )
    parser.add_argument(
        "--capture-mode",
        choices=("strict", "collect"),
        default="strict",
        help=(
            "strict keeps the one-sample readiness gate; collect stages every safe, "
            "structurally complete trajectory and defers quality filtering"
        ),
    )
    parser.add_argument(
        "--max-stages",
        type=_positive_int,
        default=1,
        help=(
            "Maximum configured stages to run. Defaults to the single-concurrency gate; "
            "a value above the configured stage count is rejected."
        ),
    )
    parser.add_argument("--sample-id")
    parser.add_argument("--source", type=Path)
    parser.add_argument("--prompt-file", type=Path)
    parser.add_argument(
        "--opencode-executable",
        type=Path,
        default=PROJECT_ROOT
        / "artifacts"
        / "tooling"
        / "opencode-patched"
        / "opencode-anchor.exe",
        help="Explicit patched OpenCode binary; the global PATH binary is never used",
    )
    parser.add_argument(
        "--sandbox-linux-executable",
        type=Path,
        default=PROJECT_ROOT
        / "artifacts"
        / "tooling"
        / "opencode-patched"
        / "linux-x64"
        / "opencode-anchor",
        help="Linux x64 OpenCode artifact mounted read-only in the Debian/WSL job",
    )
    parser.add_argument(
        "--sandbox-wsl-distro",
        default="Ubuntu-22.04" if os.name == "nt" else None,
        help="Dedicated WSL distro; required by the patched command on Windows",
    )
    parser.add_argument(
        "--sandbox-supervisor",
        choices=("direct", "wsl-root-systemd"),
        default="wsl-root-systemd" if os.name == "nt" else "direct",
        help="Patched OpenCode supervisor backend",
    )
    parser.add_argument("--sandbox-memory", default="4G")
    parser.add_argument("--sandbox-cpus", default="2")
    parser.add_argument("--sandbox-pids", type=_positive_int, default=256)
    parser.add_argument("--sandbox-timeout-seconds", type=_positive_int, default=900)
    parser.add_argument(
        "--route-mode",
        choices=("prompt", "current", "direct", "abort"),
        default="prompt",
        help=(
            "Kimi IPv4 route policy: prompt interactively, keep current routes, "
            "temporarily add only Kimi /32 direct routes, or abort"
        ),
    )
    parser.add_argument(
        "--skill",
        action="append",
        help="Audited source id from configs/data/skill_sources.yaml; repeat as needed",
    )
    parser.add_argument(
        "--skill-registry",
        type=Path,
        default=PROJECT_ROOT / "configs" / "data" / "skill_sources.yaml",
    )
    parser.add_argument(
        "--workspace-root",
        type=Path,
        default=PROJECT_ROOT / "runs" / "tooling-live",
    )
    parser.add_argument(
        "--retain-workspace",
        action="store_true",
        help="Keep copied task workspaces after capture for local debugging only",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT
        / "artifacts"
        / "tooling"
        / "live_gold.accepted.jsonl",
    )
    parser.add_argument(
        "--attempts-output",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "tooling" / "live_attempts.jsonl",
        help="Append-only audit ledger; failed attempts never enter --output",
    )
    parser.add_argument("--required", nargs="*", default=["build"])
    parser.add_argument(
        "--max-iterations",
        type=_positive_int,
        default=None,
        help="Optional OpenCode agent step limit; omitted by default",
    )
    parser.add_argument("--timeout-seconds", type=float, default=900.0)
    parser.add_argument(
        "--confirm-live",
        action="store_true",
        help="Required acknowledgement that one quota-consuming API session may run",
    )
    return parser.parse_args()


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a positive integer") from error
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def patched_preflight(
    executor: OpenCodeExecutor, policy: ToolPolicy
) -> tuple[bool, str]:
    runs = PROJECT_ROOT / "runs"
    runs.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="opencode-preflight-", dir=runs) as raw:
        config_path = write_opencode_config(Path(raw) / "opencode.json", policy)
        return executor.probe_patched(config_path)


def route_preflight(
    executor: OpenCodeExecutor, route_mode: RouteMode
) -> KimiRoutePlan | None:
    try:
        plan = prepare_kimi_route_plan(
            distro=executor.sandbox_options.wsl_distro,
            requested_mode=route_mode,
            stdin=sys.stdin,
            stdout=sys.stdout,
        )
    except RoutePreflightError as error:
        if error.code == "route_mode_abort":
            print("Live run aborted before any API request.", file=sys.stderr)
            return None
        print(f"Live run refused: Kimi route preflight failed: {error.code}", file=sys.stderr)
        return None
    print(f"route_mode={plan.mode}")
    return plan


def main() -> int:
    args = parse_args()
    if args.batch_config:
        if any((args.sample_id, args.source, args.prompt_file, args.skill)):
            print("--batch-config cannot be combined with single-sample inputs", file=sys.stderr)
            return 2
        config = LiveBatchConfig.load(PROJECT_ROOT, args.batch_config)
        if args.max_stages > len(config.concurrency_stages):
            print(
                "--max-stages exceeds the configured concurrency_stages length",
                file=sys.stderr,
            )
            return 2
        if config.opencode_executable is None:
            print("batch config requires opencode_executable", file=sys.stderr)
            return 2
        if config.attempts_output is None:
            print("batch config requires attempts_output", file=sys.stderr)
            return 2
        executor = OpenCodeExecutor(
            executable=str(config.opencode_executable),
            session_capture=config.controlled_capture(mode=args.capture_mode),
            sandbox_options=config.anchor_sandbox_options(),
        )
        policy = ToolPolicy(
            max_iterations=config.max_iterations,
            timeout_seconds=config.timeout_seconds,
        )
        registry = SkillSourceRegistry(PROJECT_ROOT, config.skill_registry)
        heldout_ids, heldout_requirements = verify_execution_split(
            PROJECT_ROOT, config.split_policy, config.candidate_manifest
        )
        samples = load_candidate_samples(
            PROJECT_ROOT,
            config.candidate_manifest,
            registry,
            heldout_identifiers=heldout_ids,
            heldout_requirements=heldout_requirements,
        )
        if not args.confirm_live:
            print("DRY RUN: batch, Skills, hashes, and held-out separation validated.")
            print(f"candidate_count={len(samples)}")
            print(f"concurrency_ramp={','.join(map(str, config.concurrency_stages))}")
            print(f"requested_stages={args.max_stages}")
            print(f"capture_mode={args.capture_mode}")
            print(
                "planned_concurrency="
                + ",".join(map(str, config.concurrency_stages[: args.max_stages]))
            )
            print(f"opencode_available={executor.available()}")
            print("patched_capability=not-run-in-dry-run")
            print(f"api_key_present={bool(os.environ.get('KIMI_CODE_API_KEY'))}")
            return 0
        route_plan = route_preflight(executor, args.route_mode)
        if route_plan is None:
            return 2
        patched, patched_reason = patched_preflight(executor, policy)
        if not patched:
            print(
                f"Live run refused: patched OpenCode preflight failed: {patched_reason}",
                file=sys.stderr,
            )
            return 2
        if not executor.available() or not os.environ.get("KIMI_CODE_API_KEY"):
            print("OpenCode and process-local KIMI_CODE_API_KEY are required.", file=sys.stderr)
            return 2
        try:
            def persist_stage(records):
                if args.capture_mode == "collect":
                    merge_attempts_jsonl(records, config.attempts_output)
                else:
                    persist_attempts_and_gold(
                        records,
                        attempts_path=config.attempts_output,
                        gold_path=config.gold_output,
                    )

            with route_plan.activate():
                stages = run_live_batch(
                    samples=samples,
                    config=config,
                    executor=executor,
                    max_stages=args.max_stages,
                    collection_mode=args.capture_mode == "collect",
                    on_stage=persist_stage,
                )
        except RoutePreflightError as error:
            print(
                f"Live run refused: temporary Kimi route failed: {error.code}",
                file=sys.stderr,
            )
            return 2
        for stage in stages:
            print(
                f"stage={stage.concurrency} records={len(stage.records)} "
                f"passed={stage.passed_gate}"
            )
        print(
            config.session_staging
            if args.capture_mode == "collect"
            else config.gold_output
        )
        return 0 if batch_run_succeeded(stages, args.max_stages) else 1

    executor = OpenCodeExecutor(
        executable=str(args.opencode_executable),
        session_capture=ControlledSessionCapture(
            args.session_candidates.resolve(),
            args.session_quarantine.resolve(),
            (PROJECT_ROOT / "configs" / "benchmark" / "heldout_cases_v1.jsonl").resolve(),
            (PROJECT_ROOT / "examples" / "benchmark" / "fixtures").resolve(),
            (PROJECT_ROOT / "artifacts" / "benchmark" / "heldout_v1" / "manifest.json").resolve(),
            staging_path=args.session_staging.resolve(),
            mode=args.capture_mode,
        ),
        sandbox_options=AnchorSandboxOptions(
            linux_executable=args.sandbox_linux_executable.resolve(),
            wsl_distro=args.sandbox_wsl_distro,
            supervisor=args.sandbox_supervisor,
            memory=args.sandbox_memory,
            cpus=args.sandbox_cpus,
            pids=args.sandbox_pids,
            timeout_seconds=args.sandbox_timeout_seconds,
        ),
    )
    policy = ToolPolicy(
        max_iterations=args.max_iterations,
        timeout_seconds=args.timeout_seconds,
    )
    if not all((args.sample_id, args.source, args.prompt_file, args.skill)):
        print(
            "single mode requires --sample-id, --source, --prompt-file, and --skill",
            file=sys.stderr,
        )
        return 2
    if not args.confirm_live:
        print("DRY RUN: no workspace, OpenCode process, or API request was created.")
        print("Add --confirm-live only after reviewing source, prompt, and policy.")
        print(f"opencode_available={executor.available()}")
        print("patched_capability=not-run-in-dry-run")
        print(f"api_key_present={bool(os.environ.get('KIMI_CODE_API_KEY'))}")
        return 0
    route_plan = route_preflight(executor, args.route_mode)
    if route_plan is None:
        return 2
    patched, patched_reason = patched_preflight(executor, policy)
    if not patched:
        print(
            f"Live run refused: patched OpenCode preflight failed: {patched_reason}",
            file=sys.stderr,
        )
        return 2
    if not executor.available():
        print("OpenCode is not installed or not on PATH.", file=sys.stderr)
        return 2
    if not os.environ.get("KIMI_CODE_API_KEY"):
        print("KIMI_CODE_API_KEY is not set in this process.", file=sys.stderr)
        return 2
    if any(name not in {"build", "test", "lint"} for name in args.required):
        print("--required accepts only build, test, and lint", file=sys.stderr)
        return 2
    task = args.prompt_file.read_text(encoding="utf-8")
    registry = SkillSourceRegistry(PROJECT_ROOT, args.skill_registry)
    prompt, skill_provenance = registry.compose_execution_prompt(task, tuple(args.skill))
    try:
        with route_plan.activate():
            record = ToolingHarness(
                args.workspace_root,
                executor,
                policy=policy,
                retain_workspace=args.retain_workspace,
            ).run_sample(
                SampleSpec(
                    args.sample_id,
                    prompt,
                    args.source,
                    tuple(args.required),
                    skill_provenance,
                )
            )
    except RoutePreflightError as error:
        print(
            f"Live run refused: temporary Kimi route failed: {error.code}",
            file=sys.stderr,
        )
        return 2
    if args.capture_mode == "collect":
        merge_attempts_jsonl([record], args.attempts_output)
        print(args.session_staging)
    else:
        persist_attempts_and_gold(
            [record], attempts_path=args.attempts_output, gold_path=args.output
        )
        print(args.output)
    if args.capture_mode == "collect":
        return (
            1
            if any(code.startswith("session_hard_reject_") for code in record.error_codes)
            else 0
        )
    return 0 if record.success else 1


if __name__ == "__main__":
    raise SystemExit(main())

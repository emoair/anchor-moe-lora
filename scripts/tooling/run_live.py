from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import tempfile


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from anchor_mvp.tooling import (  # noqa: E402
    LiveBatchConfig,
    ControlledSessionCapture,
    OpenCodeExecutor,
    SampleSpec,
    SkillSourceRegistry,
    ToolPolicy,
    ToolingHarness,
    batch_run_succeeded,
    load_candidate_samples,
    persist_attempts_and_gold,
    run_live_batch,
    verify_execution_split,
    write_opencode_config,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run audited isolated OpenCode gold samples")
    parser.add_argument(
        "--batch-config",
        type=Path,
        help="Audited 1->2->4->8 batch configuration; mutually exclusive with single mode",
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
        "--max-stages",
        type=int,
        choices=(1, 2, 3, 4),
        default=1,
        help=(
            "Maximum ramp stages to run. Defaults to the single-concurrency gate; "
            "use 2, 3, or 4 only after reviewing prior-stage gold."
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
    parser.add_argument("--max-iterations", type=int, default=8)
    parser.add_argument("--timeout-seconds", type=float, default=900.0)
    parser.add_argument(
        "--confirm-live",
        action="store_true",
        help="Required acknowledgement that one quota-consuming API session may run",
    )
    return parser.parse_args()


def patched_preflight(
    executor: OpenCodeExecutor, policy: ToolPolicy
) -> tuple[bool, str]:
    runs = PROJECT_ROOT / "runs"
    runs.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="opencode-preflight-", dir=runs) as raw:
        config_path = write_opencode_config(Path(raw) / "opencode.json", policy)
        return executor.probe_patched(config_path)


def main() -> int:
    args = parse_args()
    if args.batch_config:
        if any((args.sample_id, args.source, args.prompt_file, args.skill)):
            print("--batch-config cannot be combined with single-sample inputs", file=sys.stderr)
            return 2
        config = LiveBatchConfig.load(PROJECT_ROOT, args.batch_config)
        if config.opencode_executable is None:
            print("batch config requires opencode_executable", file=sys.stderr)
            return 2
        executor = OpenCodeExecutor(
            executable=str(config.opencode_executable),
            session_capture=config.controlled_capture(),
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
            print(
                "planned_concurrency="
                + ",".join(map(str, config.concurrency_stages[: args.max_stages]))
            )
            print(f"opencode_available={executor.available()}")
            print("patched_capability=not-run-in-dry-run")
            print(f"api_key_present={bool(os.environ.get('KIMI_CODE_API_KEY'))}")
            return 0
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
        stages = run_live_batch(
            samples=samples,
            config=config,
            executor=executor,
            max_stages=args.max_stages,
            on_stage=lambda records: persist_attempts_and_gold(
                records,
                attempts_path=config.attempts_output,
                gold_path=config.gold_output,
            ),
        )
        for stage in stages:
            print(
                f"stage={stage.concurrency} records={len(stage.records)} "
                f"passed={stage.passed_gate}"
            )
        print(config.gold_output)
        return 0 if batch_run_succeeded(stages, args.max_stages) else 1

    executor = OpenCodeExecutor(
        executable=str(args.opencode_executable),
        session_capture=ControlledSessionCapture(
            args.session_candidates.resolve(),
            args.session_quarantine.resolve(),
            (PROJECT_ROOT / "configs" / "benchmark" / "heldout_cases_v1.jsonl").resolve(),
            (PROJECT_ROOT / "examples" / "benchmark" / "fixtures").resolve(),
            (PROJECT_ROOT / "artifacts" / "benchmark" / "heldout_v1" / "manifest.json").resolve(),
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
    record = ToolingHarness(args.workspace_root, executor, policy=policy).run_sample(
        SampleSpec(
            args.sample_id,
            prompt,
            args.source,
            tuple(args.required),
            skill_provenance,
        )
    )
    persist_attempts_and_gold(
        [record], attempts_path=args.attempts_output, gold_path=args.output
    )
    print(args.output)
    return 0 if record.success else 1


if __name__ == "__main__":
    raise SystemExit(main())

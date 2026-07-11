from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from anchor_mvp.tooling import (  # noqa: E402
    LiveBatchConfig,
    OpenCodeExecutor,
    SampleSpec,
    SkillSourceRegistry,
    ToolPolicy,
    ToolingHarness,
    load_candidate_samples,
    merge_gold_jsonl,
    run_live_batch,
    verify_execution_split,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run audited isolated OpenCode gold samples")
    parser.add_argument(
        "--batch-config",
        type=Path,
        help="Audited 1->2->4->8 batch configuration; mutually exclusive with single mode",
    )
    parser.add_argument("--sample-id")
    parser.add_argument("--source", type=Path)
    parser.add_argument("--prompt-file", type=Path)
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
        default=PROJECT_ROOT / "artifacts" / "tooling" / "live_gold.jsonl",
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


def main() -> int:
    args = parse_args()
    executor = OpenCodeExecutor()
    if args.batch_config:
        if any((args.sample_id, args.source, args.prompt_file, args.skill)):
            print("--batch-config cannot be combined with single-sample inputs", file=sys.stderr)
            return 2
        config = LiveBatchConfig.load(PROJECT_ROOT, args.batch_config)
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
            print(f"opencode_available={executor.available()}")
            print(f"api_key_present={bool(os.environ.get('KIMI_CODE_API_KEY'))}")
            return 0
        if not executor.available() or not os.environ.get("KIMI_CODE_API_KEY"):
            print("OpenCode and process-local KIMI_CODE_API_KEY are required.", file=sys.stderr)
            return 2
        stages = run_live_batch(
            samples=samples,
            config=config,
            executor=executor,
            on_stage=lambda records: merge_gold_jsonl(records, config.gold_output),
        )
        for stage in stages:
            print(
                f"stage={stage.concurrency} records={len(stage.records)} "
                f"passed={stage.passed_gate}"
            )
        print(config.gold_output)
        return 0 if len(stages) == 4 and all(stage.passed_gate for stage in stages) else 1

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
        print(f"api_key_present={bool(os.environ.get('KIMI_CODE_API_KEY'))}")
        return 0
    if not executor.available():
        print("OpenCode is not installed or not on PATH.", file=sys.stderr)
        return 2
    if not os.environ.get("KIMI_CODE_API_KEY"):
        print("KIMI_CODE_API_KEY is not set in this process.", file=sys.stderr)
        return 2
    if any(name not in {"build", "test", "lint"} for name in args.required):
        print("--required accepts only build, test, and lint", file=sys.stderr)
        return 2
    policy = ToolPolicy(
        max_iterations=args.max_iterations,
        timeout_seconds=args.timeout_seconds,
    )
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
    merge_gold_jsonl([record], args.output)
    print(args.output)
    return 0 if record.success else 1


if __name__ == "__main__":
    raise SystemExit(main())

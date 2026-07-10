from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path

from ..serving import ClientConfig, OpenAICompatibleClient

from .metrics import compute_metrics
from .models import load_cases_jsonl, load_specs, write_records_jsonl
from .runner import BenchmarkRunner


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Anchor-MoE-LoRA serving benchmarks")
    parser.add_argument("--specs", required=True, help="Benchmark JSON configuration")
    parser.add_argument("--cases", required=True, help="Input cases in JSONL format")
    parser.add_argument("--output", required=True, help="Output record JSONL")
    parser.add_argument("--metrics", required=True, help="Output aggregate metrics JSON")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--api-key-env", default="ANCHOR_VLLM_API_KEY")
    parser.add_argument(
        "--backend-label",
        default="vllm",
        help="Recorded backend identity, e.g. vllm-safe or low-vram-control",
    )
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    parser.add_argument("--max-attempts", type=int, default=2)
    parser.add_argument("--no-vram", action="store_true")
    return parser


async def run(args: argparse.Namespace) -> None:
    backend = OpenAICompatibleClient(
        ClientConfig(
            base_url=args.base_url,
            api_key=os.environ.get(args.api_key_env),
            timeout_seconds=args.timeout_seconds,
            max_attempts=args.max_attempts,
        )
    )
    runner = BenchmarkRunner(
        backend,
        timeout_seconds=args.timeout_seconds,
        max_attempts=args.max_attempts,
        sample_vram=not args.no_vram,
        backend_label=args.backend_label,
    )
    records = await runner.run_suite(load_specs(args.specs), load_cases_jsonl(args.cases))
    write_records_jsonl(records, args.output)
    metrics_path = Path(args.metrics)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(
        json.dumps(compute_metrics(records), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    asyncio.run(run(build_parser().parse_args()))


if __name__ == "__main__":
    main()

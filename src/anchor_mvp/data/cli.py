"""Command-line entry point for the Anchor-MoE-LoRA data subsystem."""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

from .pipeline import DistillationPipeline, PipelineReport
from .schema import TASK_TYPES
from .teacher import CompatibleTeacher, MockTeacher, Teacher


def _simple_config(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.casefold() == ".json":
        value = json.loads(text)
    else:
        try:
            import yaml  # type: ignore[import-not-found]

            value = yaml.safe_load(text)
        except ImportError:
            value = {}
            for line in text.splitlines():
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    continue
                key, separator, raw = stripped.partition(":")
                if not separator:
                    raise ValueError(f"unsupported config line: {line!r}")
                raw = raw.strip().strip("'\"")
                if raw.isdigit():
                    value[key.strip()] = int(raw)
                else:
                    value[key.strip()] = raw
    if not isinstance(value, dict):
        raise ValueError("data config root must be a mapping")
    return {str(key): item for key, item in value.items()}


def _setting(args: argparse.Namespace, config: Mapping[str, Any], name: str, fallback: Any) -> Any:
    cli_value = getattr(args, name, None)
    return cli_value if cli_value is not None else config.get(name, fallback)


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().casefold()
    if normalized in ("1", "true", "yes", "on"):
        return True
    if normalized in ("0", "false", "no", "off"):
        return False
    raise ValueError(f"expected boolean setting, got {value!r}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Anchor-MoE-LoRA defensive data distillation")
    parser.add_argument("command", nargs="?", choices=("run", "seeds", "probe"), default="run")
    parser.add_argument("--config", type=Path, help="flat YAML or JSON config; never put secrets here")
    parser.add_argument("--dry-run", action="store_true", help="use the deterministic offline mock teacher")
    parser.add_argument("--base-url", dest="base_url")
    parser.add_argument("--fallback-base-url", dest="fallback_base_url")
    parser.add_argument("--model")
    parser.add_argument("--protocol", choices=("anthropic", "openai"))
    parser.add_argument("--no-fallback", action="store_true")
    parser.add_argument("--api-key-env", dest="api_key_env")
    parser.add_argument("--anthropic-version", dest="anthropic_version")
    parser.add_argument("--user-agent", dest="user_agent")
    parser.add_argument("--max-requests", dest="max_requests", type=int)
    parser.add_argument("--max-output-tokens-total", dest="max_output_tokens_total", type=int)
    parser.add_argument("--max-tokens", dest="max_tokens", type=int)
    parser.add_argument("--timeout-seconds", dest="timeout_seconds", type=float)
    parser.add_argument("--max-retries", dest="max_retries", type=int)
    parser.add_argument(
        "--wall-clock-deadline-seconds",
        dest="wall_clock_deadline_seconds",
        type=float,
    )
    thinking = parser.add_mutually_exclusive_group()
    thinking.add_argument("--thinking-enabled", dest="thinking_enabled", action="store_true")
    thinking.add_argument("--no-thinking", dest="thinking_enabled", action="store_false")
    parser.set_defaults(thinking_enabled=None)
    parser.add_argument("--thinking-effort", dest="thinking_effort")
    parser.add_argument("--thinking-budget-tokens", dest="thinking_budget_tokens", type=int)
    streaming = parser.add_mutually_exclusive_group()
    streaming.add_argument("--stream-openai", dest="stream_openai", action="store_true")
    streaming.add_argument("--no-stream-openai", dest="stream_openai", action="store_false")
    parser.set_defaults(stream_openai=None)
    usage_stream = parser.add_mutually_exclusive_group()
    usage_stream.add_argument(
        "--stream-options-include-usage",
        dest="stream_options_include_usage",
        action="store_true",
    )
    usage_stream.add_argument(
        "--no-stream-options-include-usage",
        dest="stream_options_include_usage",
        action="store_false",
    )
    parser.set_defaults(stream_options_include_usage=None)
    parser.add_argument("--seed-count", dest="seed_count", type=int)
    parser.add_argument("--concurrency", type=int)
    parser.add_argument("--sop-dir", dest="sop_dir", type=Path)
    parser.add_argument("--output-dir", dest="output_dir", type=Path)
    parser.add_argument("--tasks", nargs="+", choices=TASK_TYPES)
    return parser


def _teacher(args: argparse.Namespace, config: Mapping[str, Any]) -> Teacher:
    if args.dry_run:
        return MockTeacher()
    protocol = str(_setting(args, config, "protocol", "anthropic"))
    default_base = "https://api.kimi.com/coding/" if protocol == "anthropic" else "https://api.kimi.com/coding/v1"
    base_url = str(_setting(args, config, "base_url", default_base))
    model = str(args.model or os.environ.get("KIMI_MODEL_ID") or config.get("model", "kimi-for-coding"))
    return CompatibleTeacher(
        base_url=base_url,
        model=model,
        protocol=protocol,  # type: ignore[arg-type]
        fallback_protocol=None if args.no_fallback else "openai",
        fallback_base_url=str(
            _setting(args, config, "fallback_base_url", "https://api.kimi.com/coding/v1")
        ),
        api_key_env=str(_setting(args, config, "api_key_env", "KIMI_API_KEY")),
        anthropic_version=str(_setting(args, config, "anthropic_version", "2023-06-01")),
        user_agent=str(_setting(args, config, "user_agent", "anchor-moe-lora/0.1")),
        max_requests=int(_setting(args, config, "max_requests", 4100)),
        max_output_tokens_total=int(
            _setting(args, config, "max_output_tokens_total", 12_500_000)
        ),
        max_tokens=int(_setting(args, config, "max_tokens", 4096)),
        timeout_seconds=float(_setting(args, config, "timeout_seconds", 600)),
        max_retries=int(_setting(args, config, "max_retries", 1)),
        wall_clock_deadline_seconds=float(
            _setting(args, config, "wall_clock_deadline_seconds", 900)
        ),
        thinking_enabled=_as_bool(_setting(args, config, "thinking_enabled", True)),
        thinking_effort=str(_setting(args, config, "thinking_effort", "medium")),
        thinking_budget_tokens=int(
            _setting(args, config, "thinking_budget_tokens", 1024)
        ),
        stream_openai=_as_bool(_setting(args, config, "stream_openai", True)),
        stream_options_include_usage=_as_bool(
            _setting(args, config, "stream_options_include_usage", False)
        ),
    )


async def run_from_args(args: argparse.Namespace) -> PipelineReport | list[dict[str, Any]]:
    config = _simple_config(args.config.resolve()) if args.config else {}
    teacher = _teacher(args, config)
    if args.command == "probe":
        probe = getattr(teacher, "probe", None)
        if probe is None:
            raise RuntimeError("selected teacher does not support a probe")
        await probe()
        return [{"ok": True, "model": teacher.model, "protocol": teacher.protocol}]
    repo_root = Path(__file__).resolve().parents[3]
    pipeline = DistillationPipeline(
        teacher=teacher,
        sop_dir=Path(_setting(args, config, "sop_dir", repo_root / "skills")),
        output_dir=Path(_setting(args, config, "output_dir", repo_root / "data")),
        concurrency=int(_setting(args, config, "concurrency", 8)),
    )
    seed_count = int(_setting(args, config, "seed_count", 12))
    if args.command == "seeds":
        return [seed.to_dict() for seed in await pipeline.generate_seeds(seed_count)]
    raw_tasks = args.tasks or config.get("tasks") or list(TASK_TYPES)
    if isinstance(raw_tasks, str):
        raw_tasks = [item.strip() for item in raw_tasks.split(",") if item.strip()]
    return await pipeline.run(seed_count=seed_count, tasks=raw_tasks)  # type: ignore[arg-type]


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = asyncio.run(run_from_args(args))
    except (OSError, ValueError, RuntimeError) as error:
        print(f"anchor-data: {error}", file=sys.stderr)
        return 2
    if isinstance(result, PipelineReport):
        print(json.dumps(asdict(result), ensure_ascii=False, indent=2))
        return 1 if result.errors else 0
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

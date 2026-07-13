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
from .provider import (
    PRESETS,
    ProviderSelection,
    discover_models,
    provider_spec,
    query_quota,
    select_provider_model,
)
from .schema import TASK_TYPES
from .teacher import APIProtocol, CompatibleTeacher, MockTeacher, Teacher


def _simple_config(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.casefold() == ".json":
        value = json.loads(text)
    else:
        try:
            import yaml

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


def _setting(
    args: argparse.Namespace, config: Mapping[str, Any], name: str, fallback: Any
) -> Any:
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
    parser = argparse.ArgumentParser(
        description="Anchor-MoE-LoRA defensive data distillation"
    )
    parser.add_argument(
        "command",
        nargs="?",
        choices=("run", "seeds", "probe", "models", "quota"),
        default="run",
    )
    parser.add_argument(
        "--config", type=Path, help="flat YAML or JSON config; never put secrets here"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="use the deterministic offline mock teacher",
    )
    parser.add_argument("--base-url", dest="base_url")
    parser.add_argument("--fallback-base-url", dest="fallback_base_url")
    parser.add_argument("--model")
    parser.add_argument("--provider", choices=tuple(PRESETS))
    parser.add_argument(
        "--discover-models",
        dest="discover_models",
        action="store_true",
        default=None,
        help="call the protocol's official model-list endpoint before selection",
    )
    parser.add_argument(
        "--force-model",
        dest="force_model",
        action="store_true",
        default=None,
        help="skip discovery and use --model or the preset default",
    )
    parser.add_argument(
        "--model-index",
        dest="model_index",
        type=int,
        help="select a zero-based entry returned by model discovery",
    )
    parser.add_argument(
        "--protocol", choices=("anthropic", "openai", "openai_responses")
    )
    parser.add_argument("--no-fallback", action="store_true")
    parser.add_argument("--api-key-env", dest="api_key_env")
    parser.add_argument("--anthropic-version", dest="anthropic_version")
    parser.add_argument("--user-agent", dest="user_agent")
    parser.add_argument("--max-requests", dest="max_requests", type=int)
    parser.add_argument(
        "--max-output-tokens-total", dest="max_output_tokens_total", type=int
    )
    parser.add_argument("--max-tokens", dest="max_tokens", type=int)
    parser.add_argument("--timeout-seconds", dest="timeout_seconds", type=float)
    parser.add_argument("--max-retries", dest="max_retries", type=int)
    parser.add_argument(
        "--wall-clock-deadline-seconds",
        dest="wall_clock_deadline_seconds",
        type=float,
    )
    thinking = parser.add_mutually_exclusive_group()
    thinking.add_argument(
        "--thinking-enabled", dest="thinking_enabled", action="store_true"
    )
    thinking.add_argument(
        "--no-thinking", dest="thinking_enabled", action="store_false"
    )
    parser.set_defaults(thinking_enabled=None)
    parser.add_argument("--thinking-effort", dest="thinking_effort")
    parser.add_argument(
        "--thinking-budget-tokens", dest="thinking_budget_tokens", type=int
    )
    streaming = parser.add_mutually_exclusive_group()
    streaming.add_argument("--stream-openai", dest="stream_openai", action="store_true")
    streaming.add_argument(
        "--no-stream-openai", dest="stream_openai", action="store_false"
    )
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
    parser.add_argument("--seed-index-offset", dest="seed_index_offset", type=int)
    parser.add_argument("--task-card-config", dest="task_card_config", type=Path)
    parser.add_argument("--concurrency", type=int)
    parser.add_argument("--sop-dir", dest="sop_dir", type=Path)
    parser.add_argument("--output-dir", dest="output_dir", type=Path)
    parser.add_argument("--tasks", nargs="+", choices=TASK_TYPES)
    return parser


def _selection(
    args: argparse.Namespace, config: Mapping[str, Any]
) -> ProviderSelection:
    spec = provider_spec(
        config,
        preset_name=args.provider,
        protocol=args.protocol,
        base_url=args.base_url,
        api_key_env=args.api_key_env,
    )
    legacy_model = os.environ.get("KIMI_MODEL_ID") if "provider" not in config else None
    model = args.model or config.get("model") or legacy_model
    discover = _as_bool(_setting(args, config, "discover_models", False))
    force_model = _as_bool(_setting(args, config, "force_model", False))
    return select_provider_model(
        spec,
        requested_model=str(model) if model is not None else None,
        discover=discover,
        force_model=force_model,
        model_index=(
            args.model_index
            if args.model_index is not None
            else int(config["model_index"])
            if config.get("model_index") is not None
            else None
        ),
        timeout_seconds=float(config.get("discovery_timeout_seconds", 20)),
    )


def _teacher(args: argparse.Namespace, config: Mapping[str, Any]) -> Teacher:
    if args.dry_run:
        return MockTeacher()
    selection = _selection(args, config)
    spec = selection.spec
    configured_fallback = config.get("fallback_protocol")
    fallback_protocol: APIProtocol | None
    if configured_fallback is None:
        fallback_protocol = "openai" if spec.preset == "kimi-code-anthropic" else None
    elif str(configured_fallback) == "anthropic":
        fallback_protocol = "anthropic"
    elif str(configured_fallback) == "openai":
        fallback_protocol = "openai"
    elif str(configured_fallback) == "openai_responses":
        fallback_protocol = "openai_responses"
    else:
        raise ValueError(
            "fallback_protocol must be anthropic, openai, openai_responses, or null"
        )
    if args.no_fallback:
        fallback_protocol = None
    fallback_base = str(
        args.fallback_base_url
        or config.get(
            "fallback_base_url",
            PRESETS["kimi-code-openai"].base_url
            if spec.preset == "kimi-code-anthropic"
            else spec.base_url,
        )
    )
    return CompatibleTeacher(
        base_url=spec.base_url,
        model=selection.model,
        protocol=spec.protocol,
        fallback_protocol=fallback_protocol,
        fallback_base_url=fallback_base,
        api_key_env=spec.api_key_env,
        anthropic_version=str(
            _setting(args, config, "anthropic_version", "2023-06-01")
        ),
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
        provider_preset=spec.preset,
        model_source=selection.model_source,
        discovery_status=selection.discovery.status,
        discovery_model_count=len(selection.discovery.models),
    )


async def run_from_args(
    args: argparse.Namespace,
) -> PipelineReport | list[dict[str, Any]]:
    config = _simple_config(args.config.resolve()) if args.config else {}
    if args.command in {"models", "quota"}:
        spec = provider_spec(
            config,
            preset_name=args.provider,
            protocol=args.protocol,
            base_url=args.base_url,
            api_key_env=args.api_key_env,
        )
        if args.command == "models":
            result = discover_models(
                spec, timeout_seconds=float(config.get("discovery_timeout_seconds", 20))
            ).to_public_dict()
            result["provider"] = spec.preset
            result["protocol"] = spec.protocol
            result["base_url"] = spec.base_url
            return [result]
        return [
            query_quota(
                spec, timeout_seconds=float(config.get("quota_timeout_seconds", 20))
            )
        ]
    teacher = _teacher(args, config)
    if args.command == "probe":
        probe = getattr(teacher, "probe", None)
        if probe is None:
            raise RuntimeError("selected teacher does not support a probe")
        await probe()
        return [{"ok": True, "model": teacher.model, "protocol": teacher.protocol}]
    repo_root = Path(__file__).resolve().parents[3]
    raw_task_card_config = _setting(args, config, "task_card_config", None)
    task_card_config = None
    if raw_task_card_config is not None:
        task_card_config = Path(raw_task_card_config)
        if not task_card_config.is_absolute():
            task_card_config = repo_root / task_card_config
    pipeline = DistillationPipeline(
        teacher=teacher,
        sop_dir=Path(_setting(args, config, "sop_dir", repo_root / "skills")),
        output_dir=Path(_setting(args, config, "output_dir", repo_root / "data")),
        concurrency=int(_setting(args, config, "concurrency", 8)),
        seed_index_offset=int(_setting(args, config, "seed_index_offset", 0)),
        task_card_config=task_card_config,
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

"""Run a content-safe synthetic demo of the Neural Swarm stream multiplexer.

The demo loads logical model bindings and dispatch defaults from the research
configuration, fans one synthetic shared input out to the selected bindings,
and writes interleaved events as JSON Lines.  It does not load model weights,
read training data, define evaluation groups, or make a throughput claim.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import sys
import uuid
from collections.abc import AsyncIterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from anchor_mvp.research.neural_swarm_streaming import (  # noqa: E402
    STREAM_EVENT_SCHEMA_VERSION,
    BackendChunk,
    ExpertBinding,
    NeuralSwarmStreamController,
    NeuralSwarmStreamingError,
    StreamEventType,
    SwarmEvent,
    SwarmRequest,
    summarize_swarm_events,
)


CONFIG_SCHEMA_VERSION = "anchor.neural-swarm-streaming-config.v1"
DEFAULT_CONFIG = REPO_ROOT / "configs/research/neural_swarm_streaming_mvp.yaml"
CLAIM_SCOPE = "execution_scaffold_only"
EXPECTED_NON_CLAIMS = {
    "cuda_stream_overlap",
    "evaluation_arm_definition",
    "production_readiness",
    "shared_kv_correctness",
    "shared_kv_speedup",
}
EXPECTED_OPTIONAL_BACKEND_METRICS = {
    "completion_tokens",
    "kv_cache_bytes",
    "peak_vram_bytes",
    "private_kv_bytes",
    "prompt_tokens",
    "shared_kv_bytes",
    "tokens_per_second",
}
EVENT_FIELDS = {
    "backend_model_id",
    "delta",
    "elapsed_ms",
    "error_message",
    "error_type",
    "event_type",
    "expert_id",
    "global_sequence",
    "metadata",
    "per_stream_sequence",
    "request_model_id",
    "run_id",
    "schema_version",
    "stream_id",
    "task_bundle_sha256",
}
REQUIRED_EVENT_FIELDS = EVENT_FIELDS - {
    "delta",
    "error_message",
    "error_type",
    "metadata",
}


@dataclass(frozen=True)
class Arguments:
    config: Path
    max_concurrency: int | None
    fail_fast: bool | None
    request_model_ids: tuple[str, ...]


@dataclass(frozen=True)
class DemoConfig:
    bindings: tuple[ExpertBinding, ...]
    max_concurrency: int
    fail_fast: bool
    queue_capacity: int
    shared_input_contract: str
    event_schema: Path


def _parse_args(argv: Sequence[str] | None = None) -> Arguments:
    parser = argparse.ArgumentParser(
        description="Emit a synthetic, multiplexed Neural Swarm JSONL stream."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--max-concurrency",
        type=int,
        help="override dispatch.max_concurrency",
    )
    parser.add_argument(
        "--fail-fast",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="override dispatch.fail_fast",
    )
    parser.add_argument(
        "--request-model-id",
        action="append",
        default=[],
        help="logical model id to select; repeat it (default: every configured id)",
    )
    raw = parser.parse_args(argv)
    if raw.max_concurrency is not None and raw.max_concurrency < 1:
        parser.error("--max-concurrency must be positive")
    return Arguments(
        config=raw.config,
        max_concurrency=raw.max_concurrency,
        fail_fast=raw.fail_fast,
        request_model_ids=tuple(raw.request_model_id),
    )


def _mapping(value: Any, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise NeuralSwarmStreamingError(f"{field} must be a mapping")
    if any(not isinstance(key, str) for key in value):
        raise NeuralSwarmStreamingError(f"{field} keys must be strings")
    return value


def _nonempty_string(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise NeuralSwarmStreamingError(f"{field} must be a non-empty string")
    return value


def _positive_int(value: Any, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise NeuralSwarmStreamingError(f"{field} must be a positive integer")
    return value


def _exact_fields(
    value: Mapping[str, Any], *, field: str, expected: set[str]
) -> None:
    missing = sorted(expected.difference(value))
    unknown = sorted(set(value).difference(expected))
    if missing or unknown:
        raise NeuralSwarmStreamingError(
            f"{field} fields mismatch; missing={missing}, unknown={unknown}"
        )


def _required_true(value: Mapping[str, Any], *, field: str, keys: set[str]) -> None:
    for key in sorted(keys):
        if value.get(key) is not True:
            raise NeuralSwarmStreamingError(f"{field}.{key} must be true")


def _exact_string_set(value: Any, *, field: str, expected: set[str]) -> None:
    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item.strip() for item in value
    ):
        raise NeuralSwarmStreamingError(
            f"{field} must be a list of non-empty strings"
        )
    if len(value) != len(set(value)):
        raise NeuralSwarmStreamingError(f"{field} must not contain duplicates")
    actual = set(value)
    if actual != expected:
        raise NeuralSwarmStreamingError(
            f"{field} differs from the audited contract; "
            f"missing={sorted(expected - actual)}, unknown={sorted(actual - expected)}"
        )


def _validate_event_schema(path: Path) -> None:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise NeuralSwarmStreamingError(
            f"cannot load streaming.event_schema: {path}"
        ) from exc
    schema = _mapping(raw, field="streaming.event_schema document")
    if schema.get("$id") != STREAM_EVENT_SCHEMA_VERSION:
        raise NeuralSwarmStreamingError(
            "streaming.event_schema $id must match the runtime schema version"
        )
    if schema.get("additionalProperties") is not False:
        raise NeuralSwarmStreamingError(
            "streaming.event_schema must reject additional properties"
        )
    required = schema.get("required")
    properties = schema.get("properties")
    if (
        not isinstance(required, list)
        or any(not isinstance(field, str) for field in required)
        or len(required) != len(set(required))
        or set(required) != REQUIRED_EVENT_FIELDS
    ):
        raise NeuralSwarmStreamingError(
            "streaming.event_schema required fields differ from the runtime event"
        )
    properties = _mapping(properties, field="streaming.event_schema.properties")
    if set(properties) != EVENT_FIELDS:
        raise NeuralSwarmStreamingError(
            "streaming.event_schema properties differ from the runtime event"
        )
    schema_version = _mapping(
        properties.get("schema_version"),
        field="streaming.event_schema.properties.schema_version",
    )
    if schema_version.get("const") != STREAM_EVENT_SCHEMA_VERSION:
        raise NeuralSwarmStreamingError(
            "streaming.event_schema schema_version const differs from the runtime"
        )
    event_type = _mapping(
        properties.get("event_type"),
        field="streaming.event_schema.properties.event_type",
    )
    event_types = event_type.get("enum")
    expected_event_types = {value.value for value in StreamEventType}
    if (
        not isinstance(event_types, list)
        or any(not isinstance(value, str) for value in event_types)
        or len(event_types) != len(set(event_types))
        or set(event_types) != expected_event_types
    ):
        raise NeuralSwarmStreamingError(
            "streaming.event_schema event types differ from the runtime"
        )


def _load_config(path: Path) -> DemoConfig:
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise NeuralSwarmStreamingError(f"cannot load config: {path}") from exc
    root = _mapping(raw, field="config")
    _exact_fields(
        root,
        field="config",
        expected={
            "schema_version",
            "claim_scope",
            "bindings",
            "dispatch",
            "streaming",
            "observability",
            "non_claims",
        },
    )
    if root.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise NeuralSwarmStreamingError(
            f"config.schema_version must be {CONFIG_SCHEMA_VERSION}"
        )
    if root.get("claim_scope") != CLAIM_SCOPE:
        raise NeuralSwarmStreamingError(
            f"config.claim_scope must be {CLAIM_SCOPE}"
        )
    _exact_string_set(
        root.get("non_claims"),
        field="config.non_claims",
        expected=EXPECTED_NON_CLAIMS,
    )

    raw_bindings = root.get("bindings")
    if not isinstance(raw_bindings, list) or not raw_bindings:
        raise NeuralSwarmStreamingError("config.bindings must be a non-empty list")
    bindings: list[ExpertBinding] = []
    expected_fields = {"request_model_id", "expert_id", "backend_model_id"}
    for index, value in enumerate(raw_bindings):
        binding = _mapping(value, field=f"bindings[{index}]")
        unknown = sorted(set(binding).difference(expected_fields))
        missing = sorted(expected_fields.difference(binding))
        if unknown or missing:
            raise NeuralSwarmStreamingError(
                f"bindings[{index}] fields mismatch; missing={missing}, unknown={unknown}"
            )
        bindings.append(
            ExpertBinding(
                request_model_id=_nonempty_string(
                    binding["request_model_id"],
                    field=f"bindings[{index}].request_model_id",
                ),
                expert_id=_nonempty_string(
                    binding["expert_id"], field=f"bindings[{index}].expert_id"
                ),
                backend_model_id=_nonempty_string(
                    binding["backend_model_id"],
                    field=f"bindings[{index}].backend_model_id",
                ),
            )
        )

    dispatch = _mapping(root.get("dispatch"), field="config.dispatch")
    _exact_fields(
        dispatch,
        field="config.dispatch",
        expected={
            "cancellation",
            "fail_fast",
            "max_concurrency",
            "queue_capacity",
            "shared_input_contract",
            "terminal_barrier",
        },
    )
    fail_fast = dispatch.get("fail_fast")
    if not isinstance(fail_fast, bool):
        raise NeuralSwarmStreamingError("dispatch.fail_fast must be a boolean")
    if dispatch.get("terminal_barrier") != "all_settled":
        raise NeuralSwarmStreamingError(
            "synthetic demo requires dispatch.terminal_barrier=all_settled"
        )
    if dispatch.get("cancellation") != "explicit_event":
        raise NeuralSwarmStreamingError(
            "synthetic demo requires dispatch.cancellation=explicit_event"
        )

    streaming = _mapping(root.get("streaming"), field="config.streaming")
    _exact_fields(
        streaming,
        field="config.streaming",
        expected={
            "emit_barrier",
            "emit_started",
            "emit_swarm_completed",
            "emit_terminal_events",
            "event_schema",
            "require_global_sequence",
            "require_per_stream_sequence",
        },
    )
    _required_true(
        streaming,
        field="config.streaming",
        keys={
            "emit_barrier",
            "emit_started",
            "emit_swarm_completed",
            "emit_terminal_events",
            "require_global_sequence",
            "require_per_stream_sequence",
        },
    )
    event_schema_value = _nonempty_string(
        streaming.get("event_schema"), field="config.streaming.event_schema"
    )
    event_schema = Path(event_schema_value)
    if not event_schema.is_absolute():
        event_schema = REPO_ROOT / event_schema
    event_schema = event_schema.resolve()
    _validate_event_schema(event_schema)

    observability = _mapping(
        root.get("observability"), field="config.observability"
    )
    _exact_fields(
        observability,
        field="config.observability",
        expected={
            "optional_backend_metrics",
            "record_delta_count",
            "record_elapsed_ms",
            "record_output_units",
            "record_time_to_first_delta_ms",
        },
    )
    _required_true(
        observability,
        field="config.observability",
        keys={
            "record_delta_count",
            "record_elapsed_ms",
            "record_output_units",
            "record_time_to_first_delta_ms",
        },
    )
    _exact_string_set(
        observability.get("optional_backend_metrics"),
        field="config.observability.optional_backend_metrics",
        expected=EXPECTED_OPTIONAL_BACKEND_METRICS,
    )
    return DemoConfig(
        bindings=tuple(bindings),
        max_concurrency=_positive_int(
            dispatch.get("max_concurrency"), field="dispatch.max_concurrency"
        ),
        fail_fast=fail_fast,
        queue_capacity=_positive_int(
            dispatch.get("queue_capacity"), field="dispatch.queue_capacity"
        ),
        shared_input_contract=_nonempty_string(
            dispatch.get("shared_input_contract"),
            field="dispatch.shared_input_contract",
        ),
        event_schema=event_schema,
    )


class _SyntheticBackend:
    """Deterministic mock that verifies every route receives one shared snapshot."""

    def __init__(self, bindings: Sequence[ExpertBinding]) -> None:
        self._ordinal = {
            binding.request_model_id: index for index, binding in enumerate(bindings)
        }
        self._shared_snapshot_identity: int | None = None

    async def stream(
        self,
        *,
        binding: ExpertBinding,
        shared_input: Mapping[str, Any],
        run_id: str,
        task_bundle_sha256: str,
        cancel_event: asyncio.Event,
    ) -> AsyncIterator[BackendChunk]:
        del run_id, task_bundle_sha256
        snapshot_identity = id(shared_input)
        if self._shared_snapshot_identity is None:
            self._shared_snapshot_identity = snapshot_identity
        elif snapshot_identity != self._shared_snapshot_identity:
            raise RuntimeError("synthetic routes did not receive one shared snapshot")

        ordinal = self._ordinal[binding.request_model_id]
        for chunk_index in range(3):
            if cancel_event.is_set():
                return
            await asyncio.sleep(0.003 * (1 + ((ordinal + chunk_index) % 3)))
            yield BackendChunk(
                delta=f"[{binding.expert_id}:synthetic-{chunk_index}]",
                metadata={"synthetic": True, "chunk_index": chunk_index},
            )


def _shared_input(contract: str) -> tuple[dict[str, Any], str]:
    value = {
        "schema_version": contract,
        "task": {
            "kind": "synthetic_streaming_demo",
            "input_id": "content-free-placeholder",
        },
        "contains_real_sample": False,
    }
    canonical = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return value, hashlib.sha256(canonical).hexdigest()


def _write_jsonl(value: Mapping[str, Any]) -> None:
    print(json.dumps(value, ensure_ascii=False, sort_keys=True), flush=True)


async def _run(args: Arguments) -> int:
    config = _load_config(args.config.resolve())
    selected = args.request_model_ids or tuple(
        binding.request_model_id for binding in config.bindings
    )
    if len(set(selected)) != len(selected):
        raise NeuralSwarmStreamingError("duplicate --request-model-id values")

    shared_input, task_bundle_sha256 = _shared_input(config.shared_input_contract)
    backend = _SyntheticBackend(config.bindings)
    controller = NeuralSwarmStreamController(
        bindings=config.bindings,
        backend=backend,
        max_concurrency=args.max_concurrency or config.max_concurrency,
        fail_fast=config.fail_fast if args.fail_fast is None else args.fail_fast,
        queue_capacity=config.queue_capacity,
    )
    request = SwarmRequest(
        run_id=f"synthetic-{uuid.uuid4().hex[:12]}",
        task_bundle_sha256=task_bundle_sha256,
        request_model_ids=selected,
        shared_input=shared_input,
    )
    events: list[SwarmEvent] = []
    async for event in controller.stream(request):
        events.append(event)
        # Event lines are the schema object itself.  Adding an envelope field
        # would violate the event schema's additionalProperties=false contract.
        _write_jsonl(event.to_dict())

    summary = summarize_swarm_events(events)
    _write_jsonl(
        {
            "record_type": "summary",
            "claim_scope": "synthetic_demo_only",
            "contains_real_sample": False,
            **summary.to_dict(),
        }
    )
    return 0 if summary.failed_streams == 0 else 1


def _main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        return asyncio.run(_run(args))
    except NeuralSwarmStreamingError as exc:
        _write_jsonl(
            {
                "record_type": "error",
                "error_type": type(exc).__name__,
                "error_message": str(exc),
            }
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(_main())

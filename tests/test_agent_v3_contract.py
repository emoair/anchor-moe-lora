from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from anchor_mvp.data.agent_v3_contract import (
    SOURCE_SCHEMA_VERSION,
    TRAINING_VIEW_SCHEMA_VERSION,
    AgentV3ValidationError,
    build_training_view,
    canonical_sha256,
    validate_source_snapshot,
    validate_training_view,
)


ROOT = Path(__file__).resolve().parents[1]


def _tool() -> dict[str, object]:
    return {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "SyntheticHarness branded file reader.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "env": {"type": "string"},
                },
                "required": ["path"],
                "additionalProperties": False,
            },
        },
    }


def source_snapshot() -> dict[str, object]:
    system = "SyntheticHarness release prompt: obey its branded conventions."
    developer = "SyntheticHarness developer layer: use the available tools."
    call = {
        "id": "call_001",
        "type": "function",
        "function": {
            "name": "read_file",
            "arguments": '{"path":"notes.txt"}',
        },
    }
    first_request_messages = [
        {"role": "system", "content": system},
        {"role": "developer", "content": developer},
        {"role": "user", "content": "Summarize the workspace note."},
    ]
    assistant_call = {
        "role": "assistant",
        "content": None,
        "reasoning_content": "private synthetic rationale",
        "tool_calls": [call],
    }
    result = {
        "role": "tool",
        "tool_call_id": "call_001",
        "name": "read_file",
        "content": "The note says the fixture is ready.",
    }
    second_request_messages = [
        *deepcopy(first_request_messages),
        deepcopy(assistant_call),
        deepcopy(result),
    ]
    return {
        "schema_version": SOURCE_SCHEMA_VERSION,
        "sample_id": "agent-v3-synthetic-001",
        "dataset_partition": "train",
        "source": {
            "harness": "SyntheticHarness",
            "harness_version": "9.4.1",
            "protocol": "openai-chat-2026-01",
            "prompt_profile": {
                "id": "synthetic-default",
                "version": "9.4.1+sha.fixture",
                "sha256": "a" * 64,
            },
        },
        "exchanges": [
            {
                "request_id": "req_001",
                "request": {
                    "model": "teacher-fixture-model",
                    "messages": first_request_messages,
                    "tools": [_tool()],
                    "tool_choice": "auto",
                    "generation": {"temperature": 0},
                    "extensions": {"reasoning_effort": "medium"},
                },
                "response": {
                    "id": "resp_001",
                    "model": "teacher-fixture-model",
                    "assistant": assistant_call,
                    "finish_reason": "tool_calls",
                    "usage": {"input_tokens": 17, "output_tokens": 11},
                    "extensions": {"provider_trace_id": "trace_fixture"},
                },
                "tool_results": [result],
            },
            {
                "request_id": "req_002",
                "request": {
                    "model": "teacher-fixture-model",
                    "messages": second_request_messages,
                    "tools": [_tool()],
                    "tool_choice": "auto",
                    "generation": {"temperature": 0},
                    "extensions": {},
                },
                "response": {
                    "id": "resp_002",
                    "model": "teacher-fixture-model",
                    "assistant": {
                        "role": "assistant",
                        "content": "The fixture note is ready.",
                    },
                    "finish_reason": "stop",
                    "usage": {"input_tokens": 39, "output_tokens": 7},
                    "extensions": {},
                },
                "tool_results": [],
            },
        ],
    }


def test_source_snapshot_retains_versioned_prompt_tools_calls_and_results() -> None:
    snapshot = source_snapshot()

    validate_source_snapshot(snapshot)

    source = snapshot["source"]
    assert source["prompt_profile"]["version"] == "9.4.1+sha.fixture"
    first = snapshot["exchanges"][0]
    assert [message["role"] for message in first["request"]["messages"]] == [
        "system",
        "developer",
        "user",
    ]
    assert first["request"]["tools"][0]["function"]["name"] == "read_file"
    assert first["response"]["assistant"]["tool_calls"][0]["id"] == "call_001"
    assert first["tool_results"][0]["tool_call_id"] == "call_001"


def test_training_view_replaces_mutable_prompt_and_drops_brand_and_reasoning() -> None:
    snapshot = source_snapshot()

    view = build_training_view(snapshot)

    validate_training_view(view)
    assert view["source_snapshot_sha256"] == canonical_sha256(snapshot)
    serialized_training_payload = json.dumps(
        {
            "stable_core": view["stable_core"],
            "examples": view["examples"],
        },
        ensure_ascii=False,
    )
    assert "SyntheticHarness" not in serialized_training_payload
    assert "teacher-fixture-model" not in serialized_training_payload
    assert "private synthetic rationale" not in serialized_training_payload
    assert "reasoning_content" not in serialized_training_payload
    assert (
        "description"
        not in view["examples"][0]["dynamic_tools"]["tools"][0]["function"]
    )
    assert [item["role"] for item in view["examples"][0]["context_messages"]] == [
        "user"
    ]
    assert [item["role"] for item in view["examples"][1]["context_messages"]] == [
        "user",
        "assistant",
        "tool",
    ]


def test_canonical_tool_description_is_opt_in_and_provider_neutral() -> None:
    view = build_training_view(
        source_snapshot(),
        canonical_tool_descriptions={
            "read_file": "Read a UTF-8 file inside the isolated workspace."
        },
    )

    function = view["examples"][0]["dynamic_tools"]["tools"][0]["function"]
    assert function["description"] == (
        "Read a UTF-8 file inside the isolated workspace."
    )
    assert view["normalization"]["tool_descriptions"] == "canonical_registry"


def test_hidden_reasoning_content_parts_are_removed_from_targets_and_history() -> None:
    snapshot = source_snapshot()
    source_assistant = snapshot["exchanges"][0]["response"]["assistant"]
    source_assistant["content"] = [
        {"type": "reasoning", "text": "private part"},
        {"type": "text", "text": "Calling the declared reader."},
    ]
    history_assistant = snapshot["exchanges"][1]["request"]["messages"][3]
    history_assistant["content"] = deepcopy(source_assistant["content"])

    view = build_training_view(snapshot)

    assert view["examples"][0]["target"]["content"] == [
        {"type": "text", "text": "Calling the declared reader."}
    ]
    assert view["examples"][1]["context_messages"][1]["content"] == [
        {"type": "text", "text": "Calling the declared reader."}
    ]


def test_prompt_profile_can_change_without_changing_stable_core() -> None:
    first = source_snapshot()
    second = source_snapshot()
    second["source"]["prompt_profile"] = {
        "id": "synthetic-default",
        "version": "10.0.0+sha.next",
        "sha256": "b" * 64,
    }
    second["source"]["harness_version"] = "10.0.0"
    second["exchanges"][0]["request"]["messages"][0]["content"] = (
        "A completely different mutable release prompt."
    )
    second["exchanges"][1]["request"]["messages"][0]["content"] = (
        "A completely different mutable release prompt."
    )

    first_view = build_training_view(first)
    second_view = build_training_view(second)

    assert first_view["stable_core"] == second_view["stable_core"]
    assert first_view["source_snapshot_sha256"] != second_view["source_snapshot_sha256"]
    assert first_view["examples"] == second_view["examples"]


def test_plain_user_assistant_pair_is_not_agent_tool_training_evidence() -> None:
    snapshot = source_snapshot()
    snapshot["exchanges"] = [
        {
            "request_id": "req_plain",
            "request": {
                "model": "teacher-fixture-model",
                "messages": [{"role": "user", "content": "Return a short answer."}],
                "tools": [],
                "tool_choice": "none",
                "generation": {},
                "extensions": {},
            },
            "response": {
                "id": "resp_plain",
                "model": "teacher-fixture-model",
                "assistant": {"role": "assistant", "content": "Done."},
                "finish_reason": "stop",
                "usage": {},
                "extensions": {},
            },
            "tool_results": [],
        }
    ]

    with pytest.raises(AgentV3ValidationError, match="must not be empty"):
        validate_source_snapshot(snapshot)


def test_undeclared_tool_call_and_missing_result_fail_closed() -> None:
    undeclared = source_snapshot()
    undeclared["exchanges"][0]["response"]["assistant"]["tool_calls"][0]["function"][
        "name"
    ] = "write_file"
    with pytest.raises(AgentV3ValidationError, match="undeclared tool"):
        validate_source_snapshot(undeclared)

    missing = source_snapshot()
    missing["exchanges"][0]["tool_results"] = []
    with pytest.raises(AgentV3ValidationError, match="one result for every tool call"):
        validate_source_snapshot(missing)


def test_next_exchange_must_include_previous_tool_result() -> None:
    snapshot = source_snapshot()
    snapshot["exchanges"][1]["request"]["messages"] = [
        message
        for message in snapshot["exchanges"][1]["request"]["messages"]
        if message["role"] != "tool"
    ]

    with pytest.raises(AgentV3ValidationError, match="carry forward"):
        validate_source_snapshot(snapshot)


def test_next_exchange_must_include_previous_assistant_tool_call() -> None:
    snapshot = source_snapshot()
    snapshot["exchanges"][1]["request"]["messages"] = [
        message
        for message in snapshot["exchanges"][1]["request"]["messages"]
        if message["role"] != "assistant"
    ]

    with pytest.raises(AgentV3ValidationError, match="assistant tool_calls"):
        validate_source_snapshot(snapshot)


def test_transport_credentials_are_rejected_but_tool_env_schema_is_allowed() -> None:
    snapshot = source_snapshot()
    validate_source_snapshot(snapshot)

    snapshot["exchanges"][0]["request"]["extensions"] = {
        "headers": {"Authorization": "synthetic-not-a-real-secret"}
    }
    with pytest.raises(AgentV3ValidationError, match="credentials are forbidden"):
        validate_source_snapshot(snapshot)


def test_published_json_schemas_match_runtime_versions() -> None:
    source_schema = json.loads(
        (ROOT / "configs/data/agent_v3_source_snapshot.schema.json").read_text(
            encoding="utf-8"
        )
    )
    training_schema = json.loads(
        (ROOT / "configs/data/agent_v3_training_view.schema.json").read_text(
            encoding="utf-8"
        )
    )

    assert source_schema["properties"]["schema_version"]["const"] == (
        SOURCE_SCHEMA_VERSION
    )
    assert training_schema["properties"]["schema_version"]["const"] == (
        TRAINING_VIEW_SCHEMA_VERSION
    )

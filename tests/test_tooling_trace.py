import hashlib
import json
from dataclasses import asdict

from anchor_mvp.tooling import ToolPolicy
from anchor_mvp.tooling.trace import (
    classify_error_metadata,
    classify_error_text,
    parse_opencode_jsonl,
)


def test_event_reducer_keeps_safe_command_metadata_and_drops_model_text():
    events = [
        {"type": "text", "content": "private chain of thought"},
        {
            "type": "tool",
            "tool": "bash",
            "state": {
                "status": "completed",
                "input": {"command": "npm run test --if-present"},
                "exitCode": 0,
                "output": "secret output",
            },
        },
        {
            "type": "tool",
            "tool": "bash",
            "state": {
                "status": "completed",
                "input": {"command": "curl https://bad.invalid?key=SECRET"},
            },
        },
    ]
    stdout = "\n".join(json.dumps(item) for item in events)

    trace, rejected = parse_opencode_jsonl(stdout, ToolPolicy())

    assert rejected == 1
    assert trace[0].command == "npm run test --if-present"
    assert trace[0].exit_code == 0
    assert trace[0].output_sha256 == hashlib.sha256(b"secret output").hexdigest()
    assert trace[1].command is None
    assert trace[1].command_sha256 is not None
    assert "SECRET" not in json.dumps([item.__dict__ for item in trace])
    assert "private chain" not in json.dumps([item.__dict__ for item in trace])


def test_opencode_1_17_nested_tool_use_schema_is_reduced():
    stdout = json.dumps(
        {
            "type": "tool_use",
            "timestamp": 1234,
            "sessionID": "discarded-session-id",
            "part": {
                "type": "tool",
                "tool": "edit",
                "state": {"status": "completed", "input": {"filePath": "src/a.js"}},
            },
        }
    )

    trace, rejected = parse_opencode_jsonl(stdout, ToolPolicy())

    assert rejected == 0
    assert len(trace) == 1
    assert trace[0].tool == "edit"
    assert trace[0].status == "completed"
    assert "session" not in json.dumps(trace[0].__dict__)


def test_opencode_write_tool_is_accepted_as_edit_permission_alias():
    stdout = json.dumps(
        {
            "type": "tool_use",
            "part": {
                "type": "tool",
                "tool": "write",
                "state": {"status": "completed", "input": {"filePath": "src/a.js"}},
            },
        }
    )

    trace, rejected = parse_opencode_jsonl(stdout, ToolPolicy())

    assert rejected == 0
    assert trace[0].tool == "write"


def test_tool_input_is_retained_only_as_a_canonical_hash():
    secret = "private tool input must not persist"
    tool_input = {"content": secret, "filePath": "src/a.js"}
    stdout = json.dumps(
        {
            "type": "tool_use",
            "part": {
                "id": "part-write-1",
                "type": "tool",
                "tool": "write",
                "state": {
                    "status": "completed",
                    "input": tool_input,
                    "output": "ok",
                },
            },
        }
    )

    trace, rejected = parse_opencode_jsonl(stdout, ToolPolicy())

    expected = hashlib.sha256(
        json.dumps(
            tool_input,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    assert rejected == 0
    assert trace[0].input_sha256 == expected
    assert secret not in json.dumps(asdict(trace[0]))


def test_distinct_read_inputs_are_not_collapsed_without_call_ids():
    stdout = "\n".join(
        json.dumps(
            {
                "type": "tool_use",
                "part": {
                    "type": "tool",
                    "tool": "read",
                    "state": {
                        "status": "completed",
                        "input": {"filePath": f"src/{index}.js"},
                        "output": "ok",
                    },
                },
            }
        )
        for index in range(2)
    )

    trace, rejected = parse_opencode_jsonl(stdout, ToolPolicy())

    assert rejected == 0
    assert len(trace) == 2
    assert trace[0].input_sha256 != trace[1].input_sha256


def test_identical_no_id_calls_on_separate_lines_are_distinct_invocations():
    event = {
        "type": "tool_use",
        "part": {
            "type": "tool",
            "tool": "read",
            "state": {
                "status": "completed",
                "input": {"filePath": "src/a.js"},
                "output": "ok",
            },
        },
    }

    trace, rejected = parse_opencode_jsonl(
        "\n".join((json.dumps(event), json.dumps(event))), ToolPolicy()
    )

    assert rejected == 0
    assert len(trace) == 2


def test_duplicate_no_id_representations_are_deduplicated_only_within_one_line():
    state = {
        "status": "completed",
        "input": {"filePath": "src/a.js"},
        "output": "ok",
    }
    event = {
        "type": "tool_use",
        "tool": "read",
        "state": state,
        "part": {
            "type": "tool",
            "tool": "read",
            "state": state,
        },
    }

    trace, rejected = parse_opencode_jsonl(json.dumps(event), ToolPolicy())

    assert rejected == 0
    assert len(trace) == 1


def test_duplicate_representations_with_one_call_id_are_deduplicated():
    event = {
        "type": "tool_use",
        "part": {
            "id": "part-read-1",
            "type": "tool",
            "tool": "read",
            "state": {
                "status": "completed",
                "input": {"filePath": "src/a.js"},
                "output": "ok",
            },
        },
    }

    trace, rejected = parse_opencode_jsonl(
        "\n".join((json.dumps(event), json.dumps(event))), ToolPolicy()
    )

    assert rejected == 0
    assert len(trace) == 1


def test_call_lifecycle_keeps_terminal_state_and_original_input_identity():
    call_id = "part-read-lifecycle-1"
    tool_input = {"filePath": "src/a.js"}
    stdout = "\n".join(
        (
            json.dumps(
                {
                    "type": "tool_use",
                    "part": {
                        "id": call_id,
                        "type": "tool",
                        "tool": "read",
                        "state": {"status": "pending", "input": tool_input},
                    },
                }
            ),
            json.dumps(
                {
                    "type": "tool_result",
                    "id": call_id,
                    "tool": "read",
                    "state": {
                        "status": "completed",
                        "exitCode": 0,
                        "output": "ok",
                    },
                }
            ),
        )
    )

    trace, rejected = parse_opencode_jsonl(stdout, ToolPolicy())

    assert rejected == 0
    assert len(trace) == 1
    assert trace[0].sequence == 1
    assert trace[0].status == "completed"
    assert trace[0].input_sha256 == hashlib.sha256(
        json.dumps(
            tool_input,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    assert trace[0].exit_code == 0
    assert trace[0].output_sha256 == hashlib.sha256(b"ok").hexdigest()


def test_400_and_499_are_classified_without_persisting_raw_error():
    codes = classify_error_text(
        "(invalid_url) missing scheme; HTTP 499 context canceled; status code: 429"
    )

    assert codes == ("invalid_url", "client_cancelled", "rate_limited")


def test_kimi_400_missing_reasoning_content_is_classified_without_raw_error_text():
    secret = "private provider diagnostic must not persist"
    stdout = json.dumps(
        {
            "type": "error",
            "error": {
                "status": 400,
                "message": f"reasoning_content is required; {secret}",
            },
        }
    )

    codes = classify_error_metadata(stdout, "")

    assert "missing_reasoning_content" in codes
    assert all(secret not in code for code in codes)


def test_structured_opencode_api_error_classifies_kimi_body_without_retaining_it():
    secret = "sk-private-provider-body"
    stdout = json.dumps(
        {
            "type": "error",
            "error": {
                "name": "APIError",
                "data": {
                    "statusCode": 400,
                    "message": "Bad request",
                    "responseBody": json.dumps(
                        {
                            "error": {
                                "message": (
                                    "function name read is duplicated; " + secret
                                )
                            }
                        }
                    ),
                },
            },
        }
    )

    codes = classify_error_metadata(stdout, "")

    assert "kimi_400_duplicate_function_name" in codes
    assert all(secret not in code for code in codes)


def test_reasoning_content_message_without_http_400_is_not_classified_as_kimi_contract_error():
    codes = classify_error_text("HTTP 401 reasoning_content is required")

    assert "missing_reasoning_content" not in codes


def test_structural_error_metadata_is_classified_without_messages():
    stdout = json.dumps(
        {
            "type": "error",
            "error": {"name": "APIError", "statusCode": 403, "message": "discard me"},
        }
    )

    codes = classify_error_metadata(stdout, "")

    assert "agent_error_event" in codes
    assert "agent_apierror" in codes
    assert "forbidden" in codes
    assert all("discard" not in code for code in codes)


def test_structural_invalid_url_400_is_not_mislabeled_as_generic_request_failure():
    stdout = json.dumps(
        {"type": "error", "error": {"code": "invalid_url", "status": 400}}
    )

    codes = classify_error_metadata(stdout, "")

    assert "invalid_url" in codes
    assert "invalid_request" not in codes


def test_http_499_is_client_cancellation_not_upstream_server_failure():
    stdout = json.dumps({"type": "error", "error": {"status": 499}})

    codes = classify_error_metadata(stdout, "")

    assert "client_cancelled" in codes
    assert "upstream_server_error" not in codes

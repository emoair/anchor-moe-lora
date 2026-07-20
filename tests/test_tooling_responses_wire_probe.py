import inspect

from anchor_mvp.tooling.behavioral_probe import (
    DEFAULT_PROBE_TIMEOUT_SECONDS as BEHAVIORAL_PROBE_TIMEOUT_SECONDS,
    run_behavioral_probe,
)
from anchor_mvp.tooling import OpenCodeProvider
from anchor_mvp.tooling.responses_wire_probe import (
    DEFAULT_PROBE_TIMEOUT_SECONDS as RESPONSES_PROBE_TIMEOUT_SECONDS,
    ResponsesWireTranscript,
    WIRE_CALL_ID,
    WIRE_MARKER,
    run_responses_wire_probe,
)


def _provider() -> OpenCodeProvider:
    return OpenCodeProvider(
        provider_id="anchor-ark-glm52",
        npm="@ai-sdk/openai",
        base_url="https://ark.cn-beijing.volces.com/api/coding/v3",
        model="glm-5-2-260617",
        variant="max",
        key_env="ARK_CODING_API_KEY",
        route_host="ark.cn-beijing.volces.com",
    )


def _read_tool() -> dict[str, object]:
    return {
        "type": "function",
        "name": "read",
        "description": "Local read tool",
        "strict": False,
        "parameters": {
            "type": "object",
            "properties": {
                "filePath": {
                    "description": "File path",
                    "type": "string",
                },
                "limit": {
                    "description": "Line limit",
                    "type": "integer",
                },
                "offset": {
                    "description": "Line offset",
                    "type": "integer",
                },
            },
            "required": ["filePath"],
        },
    }


def _requests() -> list[dict[str, object]]:
    base = {
        "include": ["reasoning.encrypted_content"],
        "model": "glm-5-2-260617",
        "input": [],
        "max_output_tokens": 32768,
        "reasoning": {"effort": "max"},
        "store": False,
        "stream": True,
        "tool_choice": "auto",
        "tools": [_read_tool()],
    }
    return [
        base,
        {
            **base,
            "input": [
                {
                    "type": "function_call_output",
                    "call_id": WIRE_CALL_ID,
                    "output": WIRE_MARKER,
                }
            ],
        },
    ]


def test_responses_wire_contract_accepts_max_and_completed_tool_result():
    transcript = ResponsesWireTranscript(
        paths=["/v1/responses", "/v1/responses"], requests=_requests()
    )

    assert transcript.validate(_provider()) == (
        True,
        "Responses endpoint, max effort, and tool-result replay verified",
    )


def test_responses_wire_contract_rejects_store_value_drift():
    requests = _requests()
    requests[1]["store"] = True
    transcript = ResponsesWireTranscript(
        paths=["/v1/responses", "/v1/responses"], requests=requests
    )

    assert transcript.validate(_provider()) == (
        False,
        "Responses wire store drift on turn 2: expected false",
    )


def test_responses_wire_contract_rejects_missing_store():
    requests = _requests()
    del requests[0]["store"]
    transcript = ResponsesWireTranscript(
        paths=["/v1/responses", "/v1/responses"], requests=requests
    )

    assert transcript.validate(_provider()) == (
        False,
        "Responses wire store drift on turn 1: expected false",
    )


def test_responses_wire_contract_rejects_include_value_drift():
    requests = _requests()
    requests[0]["include"] = ["reasoning.encrypted_content", "message.output_text"]
    transcript = ResponsesWireTranscript(
        paths=["/v1/responses", "/v1/responses"], requests=requests
    )

    assert transcript.validate(_provider()) == (
        False,
        'Responses wire include drift on turn 1: expected ["reasoning.encrypted_content"]',
    )


def test_responses_wire_contract_rejects_missing_include():
    requests = _requests()
    del requests[1]["include"]
    transcript = ResponsesWireTranscript(
        paths=["/v1/responses", "/v1/responses"], requests=requests
    )

    assert transcript.validate(_provider()) == (
        False,
        'Responses wire include drift on turn 2: expected ["reasoning.encrypted_content"]',
    )


def test_responses_wire_contract_rejects_any_additional_top_level_field():
    requests = _requests()
    requests[0]["metadata"] = {"unexpected": True}
    transcript = ResponsesWireTranscript(
        paths=["/v1/responses", "/v1/responses"], requests=requests
    )

    assert transcript.validate(_provider()) == (
        False,
        "unverified Responses wire fields: metadata",
    )


def test_responses_wire_contract_rejects_second_turn_model_drift():
    requests = _requests()
    requests[1]["model"] = "different-model"

    assert ResponsesWireTranscript(
        paths=["/v1/responses", "/v1/responses"], requests=requests
    ).validate(_provider()) == (
        False,
        "Responses wire model drift on turn 2: expected audited model",
    )


def test_responses_wire_contract_rejects_second_turn_reasoning_drift():
    requests = _requests()
    requests[1]["reasoning"] = {"effort": "high"}

    assert ResponsesWireTranscript(
        paths=["/v1/responses", "/v1/responses"], requests=requests
    ).validate(_provider()) == (
        False,
        "Responses wire reasoning drift on turn 2: expected max",
    )


def test_responses_wire_contract_rejects_tool_schema_or_allowlist_drift_on_any_turn():
    for turn in (0, 1):
        requests = _requests()
        requests[turn]["tools"] = [
            _read_tool(),
            {
                **_read_tool(),
                "name": "bash",
            },
        ]

        assert ResponsesWireTranscript(
            paths=["/v1/responses", "/v1/responses"], requests=requests
        ).validate(_provider()) == (
            False,
            "Responses wire tools drift on turn "
            f"{turn + 1}: expected the exact local read allowlist",
        )

    requests = _requests()
    changed = _read_tool()
    changed["parameters"] = {"type": "object"}
    requests[1]["tools"] = [changed]
    assert (
        ResponsesWireTranscript(
            paths=["/v1/responses", "/v1/responses"], requests=requests
        ).validate(_provider())[0]
        is False
    )


def test_responses_wire_contract_rejects_any_current_read_schema_drift():
    mutations = []

    extra_tool_field = _read_tool()
    extra_tool_field["unexpected"] = True
    mutations.append(extra_tool_field)

    missing_description = _read_tool()
    del missing_description["description"]
    mutations.append(missing_description)

    extra_parameter_field = _read_tool()
    extra_parameter_field["parameters"]["additionalProperties"] = False
    mutations.append(extra_parameter_field)

    changed_required = _read_tool()
    changed_required["parameters"]["required"] = ["filePath", "offset"]
    mutations.append(changed_required)

    extra_property = _read_tool()
    extra_property["parameters"]["properties"]["encoding"] = {
        "description": "Encoding",
        "type": "string",
    }
    mutations.append(extra_property)

    changed_property_type = _read_tool()
    changed_property_type["parameters"]["properties"]["limit"]["type"] = (
        "number"
    )
    mutations.append(changed_property_type)

    extra_property_keyword = _read_tool()
    extra_property_keyword["parameters"]["properties"]["offset"]["minimum"] = 0
    mutations.append(extra_property_keyword)

    for tool in mutations:
        requests = _requests()
        requests[0]["tools"] = [tool]
        assert ResponsesWireTranscript(
            paths=["/v1/responses", "/v1/responses"], requests=requests
        ).validate(_provider()) == (
            False,
            "Responses wire tools drift on turn 1: expected the exact local read allowlist",
        )


def test_offline_probe_defaults_allow_current_binary_startup_latency():
    assert BEHAVIORAL_PROBE_TIMEOUT_SECONDS >= 150.0
    assert RESPONSES_PROBE_TIMEOUT_SECONDS >= 150.0
    assert (
        inspect.signature(run_behavioral_probe)
        .parameters["timeout_seconds"]
        .default
        == BEHAVIORAL_PROBE_TIMEOUT_SECONDS
    )
    assert (
        inspect.signature(run_responses_wire_probe)
        .parameters["timeout_seconds"]
        .default
        == RESPONSES_PROBE_TIMEOUT_SECONDS
    )


def test_responses_wire_contract_rejects_missing_or_changed_tool_choice():
    for turn, value in ((0, None), (1, "required")):
        requests = _requests()
        requests[turn]["tool_choice"] = value

        assert ResponsesWireTranscript(
            paths=["/v1/responses", "/v1/responses"], requests=requests
        ).validate(_provider()) == (
            False,
            f"Responses wire tool_choice drift on turn {turn + 1}: expected auto",
        )


def test_responses_wire_contract_rejects_wrong_tool_result_call_id():
    requests = _requests()
    requests[1]["input"][0]["call_id"] = "call_unrelated"

    assert ResponsesWireTranscript(
        paths=["/v1/responses", "/v1/responses"], requests=requests
    ).validate(_provider()) == (
        False,
        "Responses second turn tool result call_id did not match the first call",
    )


def test_responses_wire_contract_rejects_missing_or_ambiguous_tool_result_marker():
    requests = _requests()
    requests[1]["input"][0]["output"] = "different local content"
    assert ResponsesWireTranscript(
        paths=["/v1/responses", "/v1/responses"], requests=requests
    ).validate(_provider()) == (
        False,
        "Responses second turn omitted the completed local tool result marker",
    )

    requests = _requests()
    requests[1]["input"].append(
        {
            "type": "function_call_output",
            "call_id": WIRE_CALL_ID,
            "output": WIRE_MARKER,
        }
    )
    assert ResponsesWireTranscript(
        paths=["/v1/responses", "/v1/responses"], requests=requests
    ).validate(_provider()) == (
        False,
        "Responses second turn must contain exactly one local tool result",
    )

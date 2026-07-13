from __future__ import annotations

import asyncio
import io
from http.client import IncompleteRead
import json
import sys
import threading
import time
from pathlib import Path
from urllib.error import HTTPError, URLError

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import anchor_mvp.data.teacher as teacher_module  # noqa: E402
from anchor_mvp.data.schema import DistilledRecord, ExpertSOP, SeedDemand  # noqa: E402
from anchor_mvp.data.teacher import CompatibleTeacher  # noqa: E402


class _Response:
    def __init__(self, payload: dict) -> None:
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return None

    def read(self, size: int = -1) -> bytes:
        return json.dumps(self.payload).encode("utf-8")


class _ChunkedResponse:
    def __init__(self, chunks: list[bytes]) -> None:
        self.chunks = list(chunks)
        self.closed = False

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return None

    def read(self, size: int = -1) -> bytes:
        return self.chunks.pop(0) if self.chunks else b""

    def close(self) -> None:
        self.closed = True


def test_anthropic_request_uses_verified_headers(monkeypatch) -> None:
    captured = {}

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["timeout"] = timeout
        captured["headers"] = {
            key.casefold(): value for key, value in request.header_items()
        }
        captured["body"] = json.loads(request.data.decode("utf-8"))
        return _Response(
            {
                "content": [
                    {"type": "thinking", "thinking": "must not be distilled"},
                    {"type": "redacted_thinking", "data": "must also be ignored"},
                    {"type": "text", "text": '{"ok":true}'},
                ],
                "usage": {"output_tokens": 4},
            }
        )

    monkeypatch.setenv("KIMI_API_KEY", "secret-for-test")
    monkeypatch.setattr(teacher_module, "urlopen", fake_urlopen)
    teacher = CompatibleTeacher(max_retries=0)
    result = teacher._request_sync(
        "anthropic",
        teacher.base_url,
        "system",
        "user",
        teacher.thinking_budget_tokens + 1,
    )

    assert result == '{"ok":true}'
    assert captured["url"] == "https://api.kimi.com/coding/v1/messages"
    assert captured["timeout"] == 600.0
    assert captured["headers"]["x-api-key"] == "secret-for-test"
    assert captured["headers"]["anthropic-version"] == "2023-06-01"
    assert captured["headers"]["content-type"] == "application/json"
    assert captured["headers"]["user-agent"] == "anchor-moe-lora/0.1"
    assert captured["body"]["model"] == "kimi-for-coding"
    assert captured["body"]["thinking"] == {"type": "enabled", "budget_tokens": 1024}
    assert "temperature" not in captured["body"]
    assert "secret-for-test" not in str(teacher.generation_params)
    assert teacher.generation_params["thinking_enabled"] is True
    assert teacher.generation_params["thinking_effort"] == "medium"
    assert teacher.generation_params["thinking_budget_tokens"] == 1024


def test_openai_endpoint_fallback_shape(monkeypatch) -> None:
    captured = {}

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["headers"] = {
            key.casefold(): value for key, value in request.header_items()
        }
        captured["body"] = json.loads(request.data.decode("utf-8"))
        return _Response(
            {
                "choices": [{"message": {"content": "{}"}}],
                "usage": {"completion_tokens": 1},
            }
        )

    monkeypatch.setenv("KIMI_API_KEY", "secret-for-test")
    monkeypatch.setattr(teacher_module, "urlopen", fake_urlopen)
    teacher = CompatibleTeacher(
        protocol="openai", fallback_protocol=None, max_retries=0, stream_openai=False
    )
    assert (
        teacher._request_sync("openai", "https://api.kimi.com/coding/v1", "s", "u", 32)
        == "{}"
    )
    assert captured["url"] == "https://api.kimi.com/coding/v1/chat/completions"
    assert captured["headers"]["authorization"] == "Bearer secret-for-test"
    assert captured["body"]["reasoning_effort"] == "medium"
    assert "temperature" not in captured["body"]
    assert "stream" not in captured["body"]


@pytest.mark.parametrize(
    ("base_url", "expected"),
    [
        (
            "https://api.example.com",
            "https://api.example.com/v1/chat/completions",
        ),
        (
            "https://api.example.com/v1",
            "https://api.example.com/v1/chat/completions",
        ),
        (
            "https://ark.cn-beijing.volces.com/api/coding/v3",
            "https://ark.cn-beijing.volces.com/api/coding/v3/chat/completions",
        ),
        (
            "https://gateway.example.com/openai/v2.1",
            "https://gateway.example.com/openai/v2.1/chat/completions",
        ),
    ],
)
def test_openai_endpoint_preserves_versioned_api_roots(
    base_url: str, expected: str
) -> None:
    assert teacher_module._openai_endpoint(base_url) == expected


@pytest.mark.parametrize(
    ("protocol", "effort"),
    [("openai", "high"), ("openai_responses", "max")],
)
def test_compatible_teacher_accepts_high_and_max_reasoning_efforts(
    monkeypatch, protocol: str, effort: str
) -> None:
    captured = {}

    def fake_urlopen(request, timeout):
        captured["body"] = json.loads(request.data.decode("utf-8"))
        if protocol == "openai_responses":
            return _Response(
                {
                    "id": "resp_effort",
                    "status": "completed",
                    "output_text": "{}",
                    "usage": {"output_tokens": 1},
                }
            )
        return _Response(
            {
                "choices": [{"message": {"content": "{}"}}],
                "usage": {"completion_tokens": 1},
            }
        )

    monkeypatch.setenv("ARK_TEST_KEY", "secret-for-test")
    monkeypatch.setattr(teacher_module, "urlopen", fake_urlopen)
    teacher = CompatibleTeacher(
        base_url="https://ark.cn-beijing.volces.com/api/coding/v3",
        model="ark-model-id",
        protocol=protocol,
        fallback_protocol=None,
        api_key_env="ARK_TEST_KEY",
        thinking_effort=effort,
        stream_openai=False,
        max_retries=0,
    )
    assert teacher._request_sync(protocol, teacher.base_url, "s", "u", 64) == "{}"
    assert teacher.generation_params["thinking_effort"] == effort
    if protocol == "openai_responses":
        assert captured["body"]["reasoning"] == {"effort": "max"}
        assert "reasoning_effort" not in captured["body"]
    else:
        assert captured["body"]["reasoning_effort"] == "high"
        assert "reasoning" not in captured["body"]


def test_compatible_teacher_rejects_unknown_reasoning_effort() -> None:
    with pytest.raises(ValueError, match="thinking_effort must be one of"):
        CompatibleTeacher(thinking_effort="ultra")


def test_openai_sse_split_chunks_usage_and_reasoning_ignored(monkeypatch) -> None:
    captured = {}
    events = [
        {
            "choices": [
                {
                    "delta": {
                        "reasoning_content": "private-one",
                        "reasoning": "private-two",
                        "reasoning_details": [{"text": "private-three"}],
                        "content": '{"message":"你',
                    }
                }
            ]
        },
        {
            "choices": [
                {"delta": {"reasoning_content": "private-four", "content": '好"}'}}
            ]
        },
        {"choices": [], "usage": {"completion_tokens": 11}},
    ]
    wire = "".join(
        f"data: {json.dumps(event, ensure_ascii=False)}\r\n\r\n" for event in events
    )
    wire += "data: [DONE]\r\n\r\n"
    encoded = wire.encode("utf-8")
    chinese_boundary = encoded.index("你".encode("utf-8")) + 1
    chunks = [
        encoded[:13],
        encoded[13:chinese_boundary],
        encoded[chinese_boundary : chinese_boundary + 2],
        encoded[chinese_boundary + 2 :],
    ]

    def fake_urlopen(request, timeout):
        captured["body"] = json.loads(request.data.decode("utf-8"))
        return _ChunkedResponse(chunks)

    monkeypatch.setenv("KIMI_API_KEY", "secret-for-test")
    monkeypatch.setattr(teacher_module, "urlopen", fake_urlopen)
    teacher = CompatibleTeacher(
        protocol="openai", fallback_protocol=None, max_retries=0
    )
    result = teacher._request_sync("openai", teacher.base_url, "s", "u", 64)

    assert result == '{"message":"你好"}'
    assert "private" not in result
    assert teacher._budget.output_tokens == 11
    assert captured["body"]["stream"] is True
    assert "stream_options" not in captured["body"]


def test_openai_stream_options_usage_is_opt_in(monkeypatch) -> None:
    captured = {}

    def fake_urlopen(request, timeout):
        captured["body"] = json.loads(request.data.decode("utf-8"))
        return _ChunkedResponse(
            [b'data: {"choices":[{"delta":{"content":"{}"}}]}\n\ndata: [DONE]\n\n']
        )

    monkeypatch.setenv("KIMI_API_KEY", "secret-for-test")
    monkeypatch.setattr(teacher_module, "urlopen", fake_urlopen)
    teacher = CompatibleTeacher(
        protocol="openai",
        fallback_protocol=None,
        max_retries=0,
        stream_options_include_usage=True,
    )
    assert teacher._request_sync("openai", teacher.base_url, "s", "u", 64) == "{}"
    assert captured["body"]["stream_options"] == {"include_usage": True}


def test_openai_sse_empty_final_is_rejected(monkeypatch) -> None:
    def fake_urlopen(request, timeout):
        return _ChunkedResponse(
            [
                b'data: {"choices":[{"delta":{"reasoning_content":"private-secret",'
                b'"reasoning_details":[{"text":"hidden"}],"mystery_key":"do-not-log"},'
                b'"finish_reason":"length"}],"usage":{"completion_tokens":8192}}\n\n'
                b"data: [DONE]\n\n"
            ]
        )

    monkeypatch.setenv("KIMI_API_KEY", "secret-for-test")
    monkeypatch.setattr(teacher_module, "urlopen", fake_urlopen)
    teacher = CompatibleTeacher(
        protocol="openai", fallback_protocol=None, max_retries=0
    )
    with pytest.raises(teacher_module.TeacherError, match="no final text") as captured:
        teacher._request_sync("openai", teacher.base_url, "s", "u", 64)
    message = str(captured.value)
    assert "finish_reason=length" in message
    assert "saw_reasoning=true" in message
    assert "reasoning_chars=20" in message
    assert "completion_tokens=8192" in message
    assert "unknown_delta_keys=mystery_key" in message
    assert "private-secret" not in message
    assert "do-not-log" not in message


def test_openai_sse_wall_clock_deadline_closes_stream(monkeypatch) -> None:
    response = _ChunkedResponse(
        [b'data: {"choices":[{"delta":{"content":"late"}}]}\n\n']
    )

    def slow_read(size: int = -1) -> bytes:
        time.sleep(0.01)
        return response.chunks.pop(0) if response.chunks else b""

    response.read = slow_read  # type: ignore[method-assign]

    def fake_urlopen(request, timeout):
        return response

    monkeypatch.setenv("KIMI_API_KEY", "secret-for-test")
    monkeypatch.setattr(teacher_module, "urlopen", fake_urlopen)
    teacher = CompatibleTeacher(
        protocol="openai",
        fallback_protocol=None,
        max_retries=0,
        wall_clock_deadline_seconds=0.001,
    )
    with pytest.raises(teacher_module.ClientDeadlineExceeded, match="client_deadline"):
        teacher._request_sync("openai", teacher.base_url, "s", "u", 64)
    assert response.closed is True


def test_http_error_diagnostic_is_allowlisted_truncated_and_redacted(
    monkeypatch,
) -> None:
    key = "sk-secret-for-http-error-test"
    user = "TOP SECRET PROMPT LINE"
    error_body = json.dumps(
        {
            "error": {
                "type": "invalid_request_error",
                "code": "bad_stream_option",
                "message": f"Request {user} used {key}; " + ("x" * 500),
                "request": {"full_prompt": user},
            },
            "echo": user,
        }
    ).encode("utf-8")

    def fake_urlopen(request, timeout):
        raise HTTPError(
            request.full_url, 400, "bad request", {}, io.BytesIO(error_body)
        )

    monkeypatch.setenv("KIMI_API_KEY", key)
    monkeypatch.setattr(teacher_module, "urlopen", fake_urlopen)
    teacher = CompatibleTeacher(
        protocol="openai", fallback_protocol=None, max_retries=0, stream_openai=False
    )
    with pytest.raises(teacher_module.TeacherError) as captured:
        teacher._request_sync("openai", teacher.base_url, "system", user, 64)
    message = str(captured.value)
    assert "type=invalid_request_error" in message
    assert "code=bad_stream_option" in message
    assert key not in message
    assert user not in message
    assert len(message) < 500


def test_retry_after_429_becomes_persistable_rate_limit(monkeypatch) -> None:
    body = json.dumps(
        {
            "error": {
                "type": "rate_limit_error",
                "code": "window",
                "message": "slow down",
            }
        }
    ).encode("utf-8")

    def fake_urlopen(request, timeout):
        raise HTTPError(
            request.full_url,
            429,
            "rate limited",
            {"Retry-After": "18000"},
            io.BytesIO(body),
        )

    monkeypatch.setenv("KIMI_API_KEY", "secret-for-test")
    monkeypatch.setattr(teacher_module, "urlopen", fake_urlopen)
    teacher = CompatibleTeacher(
        protocol="openai", fallback_protocol=None, max_retries=1, stream_openai=False
    )
    with pytest.raises(teacher_module.RateLimitError) as captured:
        asyncio.run(teacher.complete(system="s", user="u"))
    assert captured.value.retry_after_seconds == 18000
    assert teacher.usage_snapshot["requests"] == 1


def test_usage_limit_403_becomes_persistable_rate_limit(monkeypatch) -> None:
    body = json.dumps(
        {
            "error": {
                "type": "permission_error",
                "message": "You've reached your usage limit. Your quota will be refreshed.",
            }
        }
    ).encode("utf-8")

    def fake_urlopen(request, timeout):
        raise HTTPError(request.full_url, 403, "forbidden", {}, io.BytesIO(body))

    monkeypatch.setenv("KIMI_API_KEY", "secret-for-test")
    monkeypatch.setattr(teacher_module, "urlopen", fake_urlopen)
    teacher = CompatibleTeacher(
        protocol="anthropic", fallback_protocol=None, max_retries=1
    )
    with pytest.raises(teacher_module.RateLimitError) as captured:
        asyncio.run(
            teacher.complete(
                system="system prompt without quota wording",
                user="bounded user prompt without provider wording",
            )
        )
    assert captured.value.retry_after_seconds is None
    assert teacher.usage_snapshot["requests"] == 1


def test_anthropic_thinking_budget_must_fit_output_cap() -> None:
    with pytest.raises(ValueError, match="greater than thinking_budget_tokens"):
        CompatibleTeacher(max_tokens=1024, thinking_budget_tokens=1024)


def test_anthropic_thinking_budget_has_public_minimum() -> None:
    with pytest.raises(ValueError, match="at least 1024"):
        CompatibleTeacher(max_tokens=4096, thinking_budget_tokens=512)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"timeout_seconds": 0}, "timeout_seconds"),
        ({"wall_clock_deadline_seconds": 0}, "wall_clock_deadline_seconds"),
        ({"max_retries": -1}, "max_retries"),
        ({"max_retries": 3}, "max_retries"),
    ],
)
def test_timeout_and_retry_policy_validation(kwargs, message) -> None:
    with pytest.raises(ValueError, match=message):
        CompatibleTeacher(**kwargs)


def test_sse_incomplete_read_retries_twice_with_request_local_provenance(
    monkeypatch,
) -> None:
    calls = 0
    delays: list[float] = []

    class InterruptedResponse:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def read(self, size: int = -1) -> bytes:
            raise IncompleteRead(b"", 1)

    def fake_urlopen(request, timeout):
        nonlocal calls
        calls += 1
        if calls <= 2:
            return InterruptedResponse()
        return _ChunkedResponse(
            [b'data: {"choices":[{"delta":{"content":"{}"}}]}\n\ndata: [DONE]\n\n']
        )

    async def fake_sleep(delay: float) -> None:
        delays.append(delay)

    monkeypatch.setenv("KIMI_API_KEY", "secret-for-test")
    monkeypatch.setattr(teacher_module, "urlopen", fake_urlopen)
    monkeypatch.setattr(teacher_module.asyncio, "sleep", fake_sleep)
    monkeypatch.setattr(teacher_module.random, "random", lambda: 0.0)
    teacher = CompatibleTeacher(
        protocol="openai",
        fallback_protocol=None,
        max_retries=2,
    )

    async def run():
        text = await teacher.complete(system="same-system", user="same-user")
        return text, teacher.provider_provenance

    text, provenance = asyncio.run(run())
    assert text == "{}"
    assert calls == 3
    assert delays == [1.0, 2.0]
    assert teacher.usage_snapshot["requests"] == 3
    assert provenance["attempts"] == {
        "wire_attempts": 3,
        "retry_count": 2,
        "max_retries": 2,
        "retry_reasons": [
            "sse_stream_read_interrupted",
            "sse_stream_read_interrupted",
        ],
    }


def test_url_error_retry_replays_identical_request_body(monkeypatch) -> None:
    bodies: list[bytes] = []

    def fake_urlopen(request, timeout):
        bodies.append(bytes(request.data))
        if len(bodies) == 1:
            raise URLError("transient route failure")
        return _Response(
            {
                "choices": [{"message": {"content": "{}"}}],
                "usage": {"completion_tokens": 1},
            }
        )

    monkeypatch.setenv("KIMI_API_KEY", "secret-for-test")
    monkeypatch.setattr(teacher_module, "urlopen", fake_urlopen)
    monkeypatch.setattr(teacher_module, "_retry_delay_seconds", lambda *args: 0.0)
    teacher = CompatibleTeacher(
        protocol="openai",
        fallback_protocol=None,
        stream_openai=False,
        max_retries=2,
    )

    async def run():
        text = await teacher.complete(system="same-system", user="same-user")
        return text, teacher.provider_provenance

    text, provenance = asyncio.run(run())
    assert text == "{}"
    assert len(bodies) == 2
    assert bodies[0] == bodies[1]
    assert provenance["attempts"]["retry_reasons"] == ["url_error"]


def test_http_499_is_retried_but_only_within_the_same_request(monkeypatch) -> None:
    calls = 0

    def fake_urlopen(request, timeout):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise HTTPError(
                request.full_url,
                499,
                "client closed request",
                {},
                io.BytesIO(b"{}"),
            )
        return _Response(
            {
                "choices": [{"message": {"content": "{}"}}],
                "usage": {"completion_tokens": 1},
            }
        )

    monkeypatch.setenv("KIMI_API_KEY", "secret-for-test")
    monkeypatch.setattr(teacher_module, "urlopen", fake_urlopen)
    monkeypatch.setattr(teacher_module, "_retry_delay_seconds", lambda *args: 0.0)
    teacher = CompatibleTeacher(
        protocol="openai",
        fallback_protocol=None,
        stream_openai=False,
        max_retries=2,
    )

    async def run():
        text = await teacher.complete(system="system", user="user")
        return text, teacher.provider_provenance

    text, provenance = asyncio.run(run())
    assert text == "{}"
    assert calls == 2
    assert provenance["attempts"]["retry_reasons"] == ["http_499"]


@pytest.mark.parametrize("status", [400, 409, 429, 501])
def test_schema_rate_limit_and_nontransient_http_errors_are_not_retried(
    monkeypatch, status: int
) -> None:
    calls = 0
    sleeps = 0

    def fake_urlopen(request, timeout):
        nonlocal calls
        calls += 1
        raise HTTPError(
            request.full_url,
            status,
            "terminal",
            {},
            io.BytesIO(b"{}"),
        )

    async def fail_sleep(delay: float) -> None:
        nonlocal sleeps
        sleeps += 1

    monkeypatch.setenv("KIMI_API_KEY", "secret-for-test")
    monkeypatch.setattr(teacher_module, "urlopen", fake_urlopen)
    monkeypatch.setattr(teacher_module.asyncio, "sleep", fail_sleep)
    teacher = CompatibleTeacher(
        protocol="openai",
        fallback_protocol=None,
        stream_openai=False,
        max_retries=2,
    )

    with pytest.raises(teacher_module.TeacherError):
        asyncio.run(teacher.complete(system="system", user="user"))
    assert calls == 1
    assert sleeps == 0


def test_response_validation_failure_is_not_retried(monkeypatch) -> None:
    calls = 0

    def invalid_request(*args, **kwargs):
        nonlocal calls
        calls += 1
        raise teacher_module.TeacherError("unexpected teacher response schema")

    teacher = CompatibleTeacher(max_retries=2)
    monkeypatch.setattr(teacher, "_request_sync", invalid_request)
    with pytest.raises(teacher_module.TeacherError, match="unexpected teacher"):
        asyncio.run(teacher.complete(system="system", user="user"))
    assert calls == 1


def test_thinking_can_be_disabled(monkeypatch) -> None:
    captured = {}

    def fake_urlopen(request, timeout):
        captured["body"] = json.loads(request.data.decode("utf-8"))
        return _Response(
            {"content": [{"type": "text", "text": "{}"}], "usage": {"output_tokens": 1}}
        )

    monkeypatch.setenv("KIMI_API_KEY", "secret-for-test")
    monkeypatch.setattr(teacher_module, "urlopen", fake_urlopen)
    teacher = CompatibleTeacher(thinking_enabled=False, max_retries=0)
    assert teacher._request_sync("anthropic", teacher.base_url, "s", "u", 32) == "{}"
    assert "thinking" not in captured["body"]
    assert captured["body"]["temperature"] == 0.2


def test_openai_thinking_off_sends_temperature(monkeypatch) -> None:
    captured = {}

    def fake_urlopen(request, timeout):
        captured["body"] = json.loads(request.data.decode("utf-8"))
        return _Response(
            {
                "choices": [{"message": {"content": "{}"}}],
                "usage": {"completion_tokens": 1},
            }
        )

    monkeypatch.setenv("KIMI_API_KEY", "secret-for-test")
    monkeypatch.setattr(teacher_module, "urlopen", fake_urlopen)
    teacher = CompatibleTeacher(
        protocol="openai",
        fallback_protocol=None,
        thinking_enabled=False,
        stream_openai=False,
        max_retries=0,
    )
    assert teacher._request_sync("openai", teacher.base_url, "s", "u", 32) == "{}"
    assert captured["body"]["temperature"] == 0.2
    assert "reasoning_effort" not in captured["body"]


def test_probe_latches_openai_fallback(monkeypatch) -> None:
    teacher = CompatibleTeacher(max_retries=0)

    def fake_request(protocol, base_url, system, user, max_tokens):
        if protocol == "anthropic":
            raise teacher_module._ProtocolError(
                "anthropic", 400, detail="code=unsupported_protocol"
            )
        return '{"ok":true}'

    monkeypatch.setattr(teacher, "_request_sync", fake_request)
    assert asyncio.run(teacher.probe()) == '{"ok":true}'
    assert teacher.protocol == "openai"
    assert teacher.base_url == "https://api.kimi.com/coding/v1"
    assert teacher.fallback_protocol is None


def test_openai_thinking_probe_leaves_room_for_public_final_text(monkeypatch) -> None:
    teacher = CompatibleTeacher(
        protocol="openai",
        fallback_protocol=None,
        thinking_enabled=True,
        max_tokens=32768,
        max_retries=0,
    )
    captured = {}

    async def fake_with_retries(protocol, base_url, system, user, max_tokens):
        captured.update(
            {
                "protocol": protocol,
                "base_url": base_url,
                "max_tokens": max_tokens,
            }
        )
        return '{"ok":true}'

    monkeypatch.setattr(teacher, "_with_retries", fake_with_retries)
    assert asyncio.run(teacher.probe()) == '{"ok":true}'
    assert captured == {
        "protocol": "openai",
        "base_url": teacher.base_url,
        "max_tokens": 4096,
    }


def test_openai_responses_nonstream_extracts_only_public_text_and_usage(
    monkeypatch,
) -> None:
    captured = {}
    body = {
        "id": "resp_nonstream",
        "status": "completed",
        "output": [
            {
                "type": "reasoning",
                "summary": [{"type": "summary_text", "text": "private-plan"}],
            },
            {
                "type": "message",
                "role": "assistant",
                "content": [
                    {"type": "output_text", "text": '{"ok":true}'},
                    {"type": "refusal", "refusal": "ignored-not-final-text"},
                ],
            },
        ],
        "usage": {
            "input_tokens": 5,
            "output_tokens": 7,
            "total_tokens": 12,
            "output_tokens_details": {"reasoning_tokens": 99},
        },
    }

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["headers"] = {
            key.casefold(): value for key, value in request.header_items()
        }
        captured["body"] = json.loads(request.data.decode("utf-8"))
        return _Response(body)

    monkeypatch.setenv("ARK_TEST_KEY", "secret-for-test")
    monkeypatch.setattr(teacher_module, "urlopen", fake_urlopen)
    teacher = CompatibleTeacher(
        base_url="https://ark.cn-beijing.volces.com/api/coding/v3",
        model="ark-model-id",
        protocol="openai_responses",
        fallback_protocol=None,
        api_key_env="ARK_TEST_KEY",
        stream_openai=False,
        max_retries=0,
    )

    async def run():
        text = await teacher.complete(system="public-system", user="public-user")
        return text, teacher.provider_provenance

    text, provenance = asyncio.run(run())
    assert text == '{"ok":true}'
    assert captured["url"] == (
        "https://ark.cn-beijing.volces.com/api/coding/v3/responses"
    )
    assert captured["headers"]["authorization"] == "Bearer secret-for-test"
    assert captured["body"] == {
        "model": "ark-model-id",
        "instructions": "public-system",
        "input": "public-user",
        "max_output_tokens": 4096,
        "reasoning": {"effort": "medium"},
        "stream": False,
        "store": False,
    }
    completion = provenance["completion"]
    assert completion == {
        "response_id": "resp_nonstream",
        "usage": {"input_tokens": 5, "output_tokens": 7, "total_tokens": 12},
    }
    assert teacher.usage_snapshot == {"requests": 1, "output_tokens": 7}
    assert "private-plan" not in json.dumps(provenance)
    assert "reasoning_tokens" not in json.dumps(provenance)

    record = DistilledRecord.from_teacher_payload(
        payload={
            "decision_trace": [
                {"check": "contract", "evidence": "fixture", "action": "finish"}
            ],
            "output": {
                "summary": "Return one bounded result.",
                "steps": [{"id": "p1", "goal": "Finish", "deliverable": "Result"}],
                "constraints": [],
            },
        },
        task_type="plan",
        seed=SeedDemand("seed-1", "title", "Build a bounded result"),
        sop=ExpertSOP("plan-v1", "plan", "fixture", "public SOP", "a" * 64),
        teacher_model=teacher.model,
        teacher_base_url=teacher.base_url,
        teacher_protocol=teacher.protocol,
        generation_params=teacher.generation_params,
        template_sha256="b" * 64,
        provider_provenance=provenance,
    ).to_dict()
    assert record["provenance"]["teacher"]["provider"]["completion"] == completion
    assert "tool_trace" not in record["provenance"]["teacher"]["provider"]["completion"]


def test_openai_responses_stream_reassembles_public_text_and_usage(
    monkeypatch,
) -> None:
    captured = {}
    events = [
        {
            "type": "response.reasoning_summary_text.delta",
            "delta": "private-stream-plan",
        },
        {"type": "response.output_text.delta", "item_id": "msg_1", "delta": '{"ok":'},
        {"type": "response.output_text.delta", "item_id": "msg_1", "delta": "true}"},
        {
            "type": "response.completed",
            "response": {
                "id": "resp_stream",
                "status": "completed",
                "usage": {"input_tokens": 8, "output_tokens": 9, "total_tokens": 17},
            },
        },
    ]
    wire = "".join(f"data: {json.dumps(event)}\n\n" for event in events)
    wire += "data: [DONE]\n\n"
    encoded = wire.encode("utf-8")

    def fake_urlopen(request, timeout):
        captured["body"] = json.loads(request.data.decode("utf-8"))
        return _ChunkedResponse([encoded[:31], encoded[31:113], encoded[113:]])

    monkeypatch.setenv("ARK_TEST_KEY", "secret-for-test")
    monkeypatch.setattr(teacher_module, "urlopen", fake_urlopen)
    teacher = CompatibleTeacher(
        base_url="https://ark.cn-beijing.volces.com/api/coding/v3",
        model="ark-model-id",
        protocol="openai_responses",
        fallback_protocol=None,
        api_key_env="ARK_TEST_KEY",
        stream_openai=True,
        max_retries=0,
    )

    async def run():
        text = await teacher.complete(system="s", user="u")
        return text, teacher.provider_provenance

    text, provenance = asyncio.run(run())
    assert text == '{"ok":true}'
    assert captured["body"]["stream"] is True
    assert provenance["completion"]["response_id"] == "resp_stream"
    assert provenance["completion"]["usage"]["output_tokens"] == 9
    assert "tool_trace" not in provenance["completion"]
    assert "private-stream-plan" not in json.dumps(provenance)
    assert teacher.usage_snapshot["output_tokens"] == 9


@pytest.mark.parametrize("status", ["failed", "incomplete", "cancelled", "error"])
def test_openai_responses_nonstream_rejects_failure_terminal_status_without_leak(
    monkeypatch, status: str
) -> None:
    secret = f"private-{status}-provider-body"

    def fake_urlopen(request, timeout):
        return _Response(
            {
                "id": "resp_failure",
                "status": status,
                "output_text": secret,
                "output": [
                    {
                        "type": "reasoning",
                        "summary": [{"type": "summary_text", "text": secret}],
                    }
                ],
                "error": {
                    "code": "provider_terminal_error",
                    "message": secret,
                    "reasoning": secret,
                },
            }
        )

    monkeypatch.setenv("ARK_TEST_KEY", "secret-for-test")
    monkeypatch.setattr(teacher_module, "urlopen", fake_urlopen)
    teacher = CompatibleTeacher(
        base_url="https://ark.cn-beijing.volces.com/api/coding/v3",
        model="ark-model-id",
        protocol="openai_responses",
        fallback_protocol=None,
        api_key_env="ARK_TEST_KEY",
        stream_openai=False,
        max_retries=0,
    )
    with pytest.raises(teacher_module.TeacherError) as captured:
        teacher._request_sync(
            "openai_responses", teacher.base_url, "system", "user", 64
        )
    error = str(captured.value)
    assert f"status={status}" in error
    assert "code=provider_terminal_error" in error
    assert secret not in error


@pytest.mark.parametrize(
    ("event", "expected_status", "expected_code"),
    [
        (
            {
                "type": "response.failed",
                "response": {
                    "status": "failed",
                    "error": {
                        "code": "server_error",
                        "message": "private-failed-body",
                    },
                },
            },
            "failed",
            "server_error",
        ),
        (
            {
                "type": "response.incomplete",
                "response": {
                    "status": "incomplete",
                    "error": {
                        "code": "output_limit",
                        "message": "private-incomplete-body",
                    },
                    "incomplete_details": {"reason": "private-incomplete-reason"},
                },
            },
            "incomplete",
            "output_limit",
        ),
        (
            {
                "type": "response.error",
                "code": "sk-private-error-code",
                "message": "private-error-body",
                "reasoning": "private-error-reasoning",
            },
            "error",
            "unknown",
        ),
        (
            {
                "type": "response.completed",
                "response": {
                    "status": "incomplete",
                    "error": {
                        "code": "truncated_output",
                        "message": "private-inconsistent-terminal-body",
                    },
                },
            },
            "incomplete",
            "truncated_output",
        ),
    ],
)
def test_openai_responses_stream_rejects_failure_events_without_leak(
    event: dict, expected_status: str, expected_code: str
) -> None:
    events = [
        {
            "type": "response.output_text.delta",
            "item_id": "msg_partial",
            "delta": "private-partial-text",
        },
        event,
    ]
    wire = "".join(f"data: {json.dumps(item)}\n\n" for item in events)
    wire += "data: [DONE]\n\n"
    with pytest.raises(teacher_module.TeacherError) as captured:
        teacher_module._openai_responses_stream_content(
            _ChunkedResponse([wire.encode("utf-8")])
        )
    error = str(captured.value)
    assert f"status={expected_status}" in error
    assert f"code={expected_code}" in error
    assert "private" not in error


def test_openai_responses_stream_rejects_eof_without_completed_event() -> None:
    wire = (
        "data: "
        + json.dumps(
            {
                "type": "response.output_text.delta",
                "item_id": "msg_partial",
                "delta": "private-partial-before-eof",
            }
        )
        + "\n\n"
    )
    with pytest.raises(
        teacher_module.TeacherError, match=r"ended before response\.completed"
    ) as captured:
        teacher_module._openai_responses_stream_content(
            _ChunkedResponse([wire.encode("utf-8")])
        )
    assert "private-partial-before-eof" not in str(captured.value)


def test_openai_responses_rejects_tool_trace_when_request_declares_no_tools(
    monkeypatch,
) -> None:
    secret = "private-tool-payload"

    def fake_urlopen(request, timeout):
        return _Response(
            {
                "id": "resp_unrequested_tool",
                "status": "completed",
                "output": [
                    {
                        "type": "function_call",
                        "id": "fc_1",
                        "call_id": "call_1",
                        "name": "shell",
                        "arguments": json.dumps({"command": secret}),
                    },
                    {
                        "type": "message",
                        "content": [{"type": "output_text", "text": "partial"}],
                    },
                ],
            }
        )

    monkeypatch.setenv("ARK_TEST_KEY", "secret-for-test")
    monkeypatch.setattr(teacher_module, "urlopen", fake_urlopen)
    teacher = CompatibleTeacher(
        base_url="https://ark.cn-beijing.volces.com/api/coding/v3",
        model="ark-model-id",
        protocol="openai_responses",
        fallback_protocol=None,
        api_key_env="ARK_TEST_KEY",
        stream_openai=False,
        max_retries=0,
    )
    with pytest.raises(
        teacher_module.TeacherError, match="tool output without declared tools"
    ) as captured:
        teacher._request_sync(
            "openai_responses", teacher.base_url, "system", "user", 64
        )
    assert secret not in str(captured.value)
    assert "completion" not in teacher.provider_provenance


def test_openai_responses_stream_rejects_tool_trace_without_declared_tools() -> None:
    secret = "private-stream-tool-payload"
    events = [
        {
            "type": "response.output_item.added",
            "item": {
                "type": "function_call",
                "id": "fc_stream",
                "call_id": "call_stream",
                "name": "shell",
                "arguments": json.dumps({"command": secret}),
            },
        },
        {
            "type": "response.completed",
            "response": {"id": "resp_stream_tool", "status": "completed"},
        },
    ]
    wire = "".join(f"data: {json.dumps(item)}\n\n" for item in events)
    with pytest.raises(
        teacher_module.TeacherError, match="tool output without declared tools"
    ) as captured:
        teacher_module._openai_responses_stream_content(
            _ChunkedResponse([wire.encode("utf-8")])
        )
    assert secret not in str(captured.value)


@pytest.mark.parametrize(
    "response_id",
    [
        "resp_sk-private-response-identifier",
        "<script>alert(1)</script>",
        "resp_javascript:alert",
        "r" * 129,
    ],
)
def test_openai_responses_omits_unsafe_response_identifiers(
    monkeypatch, response_id: str
) -> None:
    def fake_urlopen(request, timeout):
        return _Response(
            {
                "id": response_id,
                "status": "completed",
                "output_text": "{}",
                "usage": {"output_tokens": 1},
            }
        )

    monkeypatch.setenv("ARK_TEST_KEY", "secret-for-test")
    monkeypatch.setattr(teacher_module, "urlopen", fake_urlopen)
    teacher = CompatibleTeacher(
        base_url="https://ark.cn-beijing.volces.com/api/coding/v3",
        model="ark-model-id",
        protocol="openai_responses",
        fallback_protocol=None,
        api_key_env="ARK_TEST_KEY",
        stream_openai=False,
        max_retries=0,
    )

    async def run():
        await teacher.complete(system="s", user="u")
        return teacher.provider_provenance

    provenance = asyncio.run(run())
    assert "response_id" not in provenance["completion"]
    assert response_id not in json.dumps(provenance)


def test_openai_responses_reasoning_only_body_is_rejected_without_leak(
    monkeypatch,
) -> None:
    def fake_urlopen(request, timeout):
        return _Response(
            {
                "id": "resp_hidden_only",
                "output": [
                    {
                        "type": "reasoning",
                        "summary": [
                            {"type": "summary_text", "text": "private-only-plan"}
                        ],
                    }
                ],
                "usage": {"output_tokens": 11},
            }
        )

    monkeypatch.setenv("ARK_TEST_KEY", "secret-for-test")
    monkeypatch.setattr(teacher_module, "urlopen", fake_urlopen)
    teacher = CompatibleTeacher(
        base_url="https://ark.cn-beijing.volces.com/api/coding/v3",
        model="ark-model-id",
        protocol="openai_responses",
        fallback_protocol=None,
        api_key_env="ARK_TEST_KEY",
        stream_openai=False,
        max_retries=0,
    )
    with pytest.raises(teacher_module.TeacherError, match="no final text") as captured:
        teacher._request_sync(
            "openai_responses", teacher.base_url, "system", "user", 64
        )
    assert "private-only-plan" not in str(captured.value)


def test_openai_responses_request_metadata_is_task_local_under_concurrency(
    monkeypatch,
) -> None:
    def fake_urlopen(request, timeout):
        payload = json.loads(request.data.decode("utf-8"))
        marker = payload["input"]
        return _Response(
            {
                "id": f"resp_{marker}",
                "output_text": json.dumps({"marker": marker}),
                "output": [],
                "usage": {"output_tokens": 2},
            }
        )

    monkeypatch.setenv("ARK_TEST_KEY", "secret-for-test")
    monkeypatch.setattr(teacher_module, "urlopen", fake_urlopen)
    teacher = CompatibleTeacher(
        base_url="https://ark.cn-beijing.volces.com/api/coding/v3",
        model="ark-model-id",
        protocol="openai_responses",
        fallback_protocol=None,
        api_key_env="ARK_TEST_KEY",
        stream_openai=False,
        max_retries=0,
    )

    async def worker(marker: str):
        text = await teacher.complete(system="s", user=marker)
        return text, teacher.provider_provenance["completion"]["response_id"]

    async def run_workers():
        return await asyncio.gather(worker("one"), worker("two"))

    results = asyncio.run(run_workers())
    assert sorted(results) == [
        ('{"marker": "one"}', "resp_one"),
        ('{"marker": "two"}', "resp_two"),
    ]


def test_generic_400_does_not_forward_prompt_to_fallback(monkeypatch) -> None:
    teacher = CompatibleTeacher(
        protocol="anthropic",
        fallback_protocol="openai",
        max_retries=0,
    )
    calls: list[tuple[str, str, str]] = []

    def fake_request(protocol, base_url, system, user, max_tokens):
        calls.append((protocol, system, user))
        raise teacher_module._ProtocolError(
            protocol,
            400,
            detail="code=invalid_request_error; message=request was invalid",
        )

    monkeypatch.setattr(teacher, "_request_sync", fake_request)
    with pytest.raises(teacher_module.TeacherError, match="HTTP 400"):
        asyncio.run(
            teacher.complete(
                system="sensitive-system-prompt",
                user="sensitive-user-prompt",
            )
        )
    assert calls == [("anthropic", "sensitive-system-prompt", "sensitive-user-prompt")]
    assert teacher.protocol == "anthropic"
    assert teacher.fallback_protocol == "openai"


def test_complete_does_not_replay_prompt_for_explicit_compatibility_error(
    monkeypatch,
) -> None:
    teacher = CompatibleTeacher(
        base_url="https://primary.example/v1",
        protocol="openai",
        fallback_protocol="openai_responses",
        fallback_base_url="https://fallback.example/api/v3",
        max_retries=0,
    )
    calls: list[tuple[str, str, str]] = []

    def fake_request(protocol, base_url, system, user, max_tokens):
        calls.append((protocol, system, user))
        raise teacher_module._ProtocolError(
            protocol, 404, detail="code=endpoint_not_found"
        )

    monkeypatch.setattr(teacher, "_request_sync", fake_request)
    with pytest.raises(teacher_module.TeacherError, match="HTTP 404"):
        asyncio.run(
            teacher.complete(
                system="private-real-system",
                user="private-real-user",
            )
        )
    assert calls == [("openai", "private-real-system", "private-real-user")]
    assert teacher.protocol == "openai"
    assert teacher.fallback_protocol == "openai_responses"


def test_probe_only_fallback_latches_actual_protocol_and_uses_synthetic_prompt(
    monkeypatch,
) -> None:
    fallback_base = "https://fallback.example/api/v3"
    teacher = CompatibleTeacher(
        base_url="https://primary.example/v1",
        protocol="openai",
        fallback_protocol="openai_responses",
        fallback_base_url=fallback_base,
        max_retries=0,
    )
    calls: list[tuple[str, str, str]] = []

    def fake_request(protocol, base_url, system, user, max_tokens):
        calls.append((protocol, system, user))
        if protocol == "openai":
            raise teacher_module._ProtocolError(
                protocol, 404, detail="code=endpoint_not_found"
            )
        return teacher_module._CompletionText(
            '{"ok":true}', {"response_id": "resp_fallback"}
        )

    monkeypatch.setattr(teacher, "_request_sync", fake_request)

    async def run_probe():
        text = await teacher.probe()
        return text, teacher.provider_provenance

    text, provenance = asyncio.run(run_probe())
    assert text == '{"ok":true}'
    assert calls == [
        (
            "openai",
            "Return a minimal JSON health response.",
            'Return exactly {"ok":true}.',
        ),
        (
            "openai_responses",
            "Return a minimal JSON health response.",
            'Return exactly {"ok":true}.',
        ),
    ]
    assert teacher.protocol == "openai_responses"
    assert teacher.base_url == fallback_base
    assert teacher.fallback_protocol is None
    assert provenance["protocol"] == "openai_responses"
    assert provenance["base_url"] == fallback_base
    assert provenance["completion"] == {"response_id": "resp_fallback"}


def test_provider_provenance_route_is_task_local_across_protocols(monkeypatch) -> None:
    teacher = CompatibleTeacher(max_retries=0)

    def fake_request(protocol, base_url, system, user, max_tokens):
        time.sleep(0.005 if user == "one" else 0.001)
        return teacher_module._CompletionText(user, {"response_id": f"resp_{user}"})

    monkeypatch.setattr(teacher, "_request_sync", fake_request)

    async def worker(protocol, base_url, marker):
        text = await teacher._with_retries(
            protocol, base_url, "system", marker, max_tokens=16
        )
        return text, teacher.provider_provenance

    async def run_workers():
        return await asyncio.gather(
            worker("openai", "https://one.example/v1", "one"),
            worker(
                "openai_responses",
                "https://two.example/api/v3",
                "two",
            ),
        )

    first, second = asyncio.run(run_workers())
    assert first[0] == "one"
    assert first[1]["base_url"] == "https://one.example/v1"
    assert first[1]["protocol"] == "openai"
    assert first[1]["completion"] == {"response_id": "resp_one"}
    assert second[0] == "two"
    assert second[1]["base_url"] == "https://two.example/api/v3"
    assert second[1]["protocol"] == "openai_responses"
    assert second[1]["completion"] == {"response_id": "resp_two"}


@pytest.mark.parametrize(
    ("code", "expected_type"),
    [
        ("insufficient_quota", teacher_module.ProviderQuotaExhausted),
        ("rate_limit_exceeded", teacher_module.RateLimitError),
        ("deadline_exceeded", teacher_module.ClientDeadlineExceeded),
        ("server_error", teacher_module.TeacherError),
    ],
)
def test_openai_responses_app_failure_maps_to_scheduler_exception(
    code: str, expected_type: type[Exception]
) -> None:
    private_message = f"private-provider-prose-{code}"
    with pytest.raises(expected_type) as captured:
        teacher_module._openai_responses_body_content(
            {
                "status": "failed",
                "error": {"code": code, "message": private_message},
            },
            deadline_seconds=12.5,
        )
    assert type(captured.value) is expected_type
    assert private_message not in str(captured.value)


def test_openai_responses_top_level_error_event_maps_rate_limit() -> None:
    wire = (
        "data: "
        + json.dumps(
            {
                "type": "error",
                "error": {
                    "type": "rate_limit_error",
                    "message": "private-provider-prose",
                },
            }
        )
        + "\n\n"
    )
    with pytest.raises(teacher_module.RateLimitError) as captured:
        teacher_module._openai_responses_stream_content(
            _ChunkedResponse([wire.encode("utf-8")])
        )
    assert type(captured.value) is teacher_module.RateLimitError
    assert "private-provider-prose" not in str(captured.value)


def test_openai_responses_completed_closes_without_done_or_another_read() -> None:
    events = [
        {
            "type": "response.output_text.delta",
            "item_id": "msg_1",
            "delta": "{}",
        },
        {
            "type": "response.completed",
            "response": {
                "id": "resp_completed",
                "status": "completed",
                "usage": {"output_tokens": 1},
            },
        },
    ]
    wire = "".join(f"data: {json.dumps(event)}\n\n" for event in events)
    response = _ChunkedResponse(
        [wire.encode("utf-8"), b'data: {"type":"must_not_be_read"}\n\n']
    )
    content, output_tokens, completion = (
        teacher_module._openai_responses_stream_content(response)
    )
    assert content == "{}"
    assert output_tokens == 1
    assert completion["response_id"] == "resp_completed"
    assert response.closed is True
    assert len(response.chunks) == 1


@pytest.mark.parametrize(
    "post_completion_event",
    [
        {
            "type": "response.output_text.delta",
            "item_id": "msg_1",
            "delta": "late",
        },
        {
            "type": "response.completed",
            "response": {"id": "resp_second", "status": "completed"},
        },
    ],
)
def test_openai_responses_rejects_buffered_event_after_completed(
    post_completion_event: dict,
) -> None:
    events = [
        {
            "type": "response.completed",
            "response": {"status": "completed", "output_text": "{}"},
        },
        post_completion_event,
    ]
    wire = "".join(f"data: {json.dumps(event)}\n\n" for event in events)
    with pytest.raises(
        teacher_module.TeacherError, match=r"event after response\.completed"
    ):
        teacher_module._openai_responses_stream_content(
            _ChunkedResponse([wire.encode("utf-8")])
        )


class _DeadlineCloseReadErrorResponse:
    def __init__(self, error_type: type[Exception]) -> None:
        self.error_type = error_type
        self.closed = threading.Event()

    def read(self, size: int = -1) -> bytes:
        if not self.closed.wait(timeout=1):
            raise AssertionError("deadline did not close the response")
        raise self.error_type("reader closed")

    def close(self) -> None:
        self.closed.set()


@pytest.mark.parametrize("error_type", [OSError, ValueError])
def test_responses_read_error_after_deadline_maps_to_client_deadline(
    error_type: type[Exception],
) -> None:
    response = _DeadlineCloseReadErrorResponse(error_type)
    with pytest.raises(teacher_module.ClientDeadlineExceeded) as captured:
        teacher_module._openai_responses_stream_content(
            response,
            deadline_at=time.monotonic() + 0.01,
            deadline_seconds=0.01,
        )
    assert captured.value.seconds == 0.01


def test_current_credential_is_omitted_from_response_metadata_and_output(
    monkeypatch,
) -> None:
    # Deliberately has no scanner-recognized prefix: this exercises the
    # request-scoped credential gate rather than a generic regex.
    credential = "CurrentCredential0123456789ABCDEF"
    for response_id in (credential, f"resp_{credential}_suffix"):
        _, _, completion = teacher_module._openai_responses_body_content(
            {
                "id": response_id,
                "status": "completed",
                "output_text": "{}",
                "usage": {"output_tokens": 1},
            },
            current_credential=credential,
        )
        assert "response_id" not in completion

    for error_code in (credential, f"error_{credential}_suffix"):
        with pytest.raises(teacher_module.TeacherError) as captured:
            teacher_module._openai_responses_body_content(
                {
                    "status": "failed",
                    "error": {"code": error_code, "message": "private"},
                },
                current_credential=credential,
            )
        assert credential not in str(captured.value)
        assert "code=unknown" in str(captured.value)

    def fake_urlopen(request, timeout):
        return _Response(
            {
                "status": "completed",
                "output_text": f"prefix-{credential}-suffix",
            }
        )

    monkeypatch.setenv("ARK_TEST_KEY", credential)
    monkeypatch.setattr(teacher_module, "urlopen", fake_urlopen)
    teacher = CompatibleTeacher(
        base_url="https://ark.example/api/v3",
        protocol="openai_responses",
        fallback_protocol=None,
        api_key_env="ARK_TEST_KEY",
        stream_openai=False,
        max_retries=0,
    )
    with pytest.raises(
        teacher_module.TeacherError,
        match="teacher response contained current credential",
    ) as output_error:
        teacher._request_sync(
            "openai_responses", teacher.base_url, "system", "user", 16
        )
    assert credential not in str(output_error.value)

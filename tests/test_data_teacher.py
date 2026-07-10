from __future__ import annotations

import asyncio
import io
import json
import sys
import time
from pathlib import Path
from urllib.error import HTTPError

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import anchor_mvp.data.teacher as teacher_module  # noqa: E402
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
        captured["headers"] = {key.casefold(): value for key, value in request.header_items()}
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
        "anthropic", teacher.base_url, "system", "user", teacher.thinking_budget_tokens + 1
    )

    assert result == '{"ok":true}'
    assert captured["url"] == "https://api.kimi.com/coding/v1/messages"
    assert captured["timeout"] == 600.0
    assert captured["headers"]["x-api-key"] == "secret-for-test"
    assert captured["headers"]["anthropic-version"] == "2023-06-01"
    assert captured["headers"]["content-type"] == "application/json"
    assert captured["headers"]["user-agent"] == "anchor-mvp/0.1"
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
        captured["headers"] = {key.casefold(): value for key, value in request.header_items()}
        captured["body"] = json.loads(request.data.decode("utf-8"))
        return _Response({"choices": [{"message": {"content": "{}"}}], "usage": {"completion_tokens": 1}})

    monkeypatch.setenv("KIMI_API_KEY", "secret-for-test")
    monkeypatch.setattr(teacher_module, "urlopen", fake_urlopen)
    teacher = CompatibleTeacher(
        protocol="openai", fallback_protocol=None, max_retries=0, stream_openai=False
    )
    assert teacher._request_sync("openai", "https://api.kimi.com/coding/v1", "s", "u", 32) == "{}"
    assert captured["url"] == "https://api.kimi.com/coding/v1/chat/completions"
    assert captured["headers"]["authorization"] == "Bearer secret-for-test"
    assert captured["body"]["reasoning_effort"] == "medium"
    assert "temperature" not in captured["body"]
    assert "stream" not in captured["body"]


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
        {"choices": [{"delta": {"reasoning_content": "private-four", "content": '好"}'}}]},
        {"choices": [], "usage": {"completion_tokens": 11}},
    ]
    wire = "".join(f"data: {json.dumps(event, ensure_ascii=False)}\r\n\r\n" for event in events)
    wire += "data: [DONE]\r\n\r\n"
    encoded = wire.encode("utf-8")
    chinese_boundary = encoded.index("你".encode("utf-8")) + 1
    chunks = [encoded[:13], encoded[13:chinese_boundary], encoded[chinese_boundary : chinese_boundary + 2], encoded[chinese_boundary + 2 :]]

    def fake_urlopen(request, timeout):
        captured["body"] = json.loads(request.data.decode("utf-8"))
        return _ChunkedResponse(chunks)

    monkeypatch.setenv("KIMI_API_KEY", "secret-for-test")
    monkeypatch.setattr(teacher_module, "urlopen", fake_urlopen)
    teacher = CompatibleTeacher(protocol="openai", fallback_protocol=None, max_retries=0)
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
        return _ChunkedResponse([b'data: {"choices":[{"delta":{"content":"{}"}}]}\n\ndata: [DONE]\n\n'])

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
                b'data: [DONE]\n\n'
            ]
        )

    monkeypatch.setenv("KIMI_API_KEY", "secret-for-test")
    monkeypatch.setattr(teacher_module, "urlopen", fake_urlopen)
    teacher = CompatibleTeacher(protocol="openai", fallback_protocol=None, max_retries=0)
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
    response = _ChunkedResponse([b'data: {"choices":[{"delta":{"content":"late"}}]}\n\n'])

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


def test_http_error_diagnostic_is_allowlisted_truncated_and_redacted(monkeypatch) -> None:
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
        raise HTTPError(request.full_url, 400, "bad request", {}, io.BytesIO(error_body))

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
        {"error": {"type": "rate_limit_error", "code": "window", "message": "slow down"}}
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
    teacher = CompatibleTeacher(protocol="anthropic", fallback_protocol=None, max_retries=1)
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
    ],
)
def test_timeout_and_retry_policy_validation(kwargs, message) -> None:
    with pytest.raises(ValueError, match=message):
        CompatibleTeacher(**kwargs)


def test_thinking_can_be_disabled(monkeypatch) -> None:
    captured = {}

    def fake_urlopen(request, timeout):
        captured["body"] = json.loads(request.data.decode("utf-8"))
        return _Response({"content": [{"type": "text", "text": "{}"}], "usage": {"output_tokens": 1}})

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
        return _Response({"choices": [{"message": {"content": "{}"}}], "usage": {"completion_tokens": 1}})

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
            raise teacher_module._ProtocolError("anthropic", 400)
        return '{"ok":true}'

    monkeypatch.setattr(teacher, "_request_sync", fake_request)
    assert asyncio.run(teacher.probe()) == '{"ok":true}'
    assert teacher.protocol == "openai"
    assert teacher.base_url == "https://api.kimi.com/coding/v1"
    assert teacher.fallback_protocol is None

import asyncio
import json

from anchor_mvp.serving import ClientConfig, CompletionRequest, Message, OpenAICompatibleClient


class _Response:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return None

    def read(self):
        return json.dumps(self.payload).encode("utf-8")


def test_client_uses_model_field_as_adapter_selector(monkeypatch):
    captured = {}

    def fake_urlopen(request, timeout):
        captured["body"] = json.loads(request.data.decode("utf-8"))
        return _Response(
            {
                "model": "lora-frontend-gen",
                "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3},
            }
        )

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    client = OpenAICompatibleClient(ClientConfig(max_attempts=1))
    response = asyncio.run(
        client.complete(
            CompletionRequest(
                model="lora-frontend-gen",
                messages=(Message("user", "hello"),),
            )
        )
    )

    assert captured["body"]["model"] == "lora-frontend-gen"
    assert response.content == "ok"
    assert response.usage.total_tokens == 3


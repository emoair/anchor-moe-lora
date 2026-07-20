from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path
import threading
from typing import Any

import pytest

from anchor_mvp.serving.windows_serial_server import (
    AdapterBinding,
    FormalAdapterCatalog,
    FormalSerialService,
    GenerationResult,
    RequestContractError,
    WindowsSerialServerError,
    _generation_stop_token_ids,
    _install_unpadded_decode_fast_path,
    _is_unpadded_text_batch,
    _openai_agent_message,
    _resolve_processor_manifest_binding,
    create_app,
    parse_agent_chat_request,
    parse_chat_request,
    verify_base_artifact,
    verify_processor_artifact,
)


ROOT = Path(__file__).resolve().parents[1]
SERVER_LAUNCHER = ROOT / "scripts/serve/start_formal_af_serial_transformers.ps1"
BENCH_LAUNCHER = (
    ROOT / "scripts/benchmark/run_formal_partial_v1_af_windows_native.ps1"
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _catalog(tmp_path: Path) -> FormalAdapterCatalog:
    run_root = tmp_path / "registries" / "run-1"
    groups: dict[str, Any] = {}
    runtime: dict[str, Any] = {}
    locks: dict[str, Any] = {}
    for group in "ABCDEF":
        registry = run_root / group / "group_registry.json"
        groups[group] = {
            "registry_path": registry.relative_to(tmp_path).as_posix()
        }
        if group == "A":
            _write_json(registry, {"adapters": []})
            locks[group] = {"registry_sha256": _sha(registry)}
            runtime[group] = {
                stage: {
                    "model_id": "gemma4-12b-base-q4",
                    "adapter_artifact": None,
                    "adapter_dir": None,
                    "adapter_sha256": None,
                }
                for stage in ("planner", "tool_policy", "frontend", "review", "security")
            }
            continue
        adapter = tmp_path / "adapters" / group
        adapter.mkdir(parents=True)
        artifact_digest = group.lower() * 64
        artifact = f"adapter-{group.lower()}"
        final_files = {}
        for label, filename in {
            "adapter_config": "adapter_config.json",
            "adapter_model": "adapter_model.safetensors",
            "checkpoint_metadata": "checkpoint_metadata.json",
        }.items():
            file_path = adapter / filename
            file_path.write_bytes(label.encode("ascii"))
            final_files[label] = {
                "path": file_path.relative_to(tmp_path).as_posix(),
                "bytes": file_path.stat().st_size,
                "sha256": _sha(file_path),
            }
        _write_json(
            registry,
            {
                "adapters": [
                    {
                        "artifact_name": artifact,
                        "adapter_sha256": artifact_digest,
                        "final_files": final_files,
                    }
                ]
            },
        )
        locks[group] = {"registry_sha256": _sha(registry)}
        runtime[group] = {
            stage: {
                "model_id": f"formal-{group.lower()}",
                "adapter_artifact": artifact,
                "adapter_dir": adapter.relative_to(tmp_path).as_posix(),
                "adapter_sha256": artifact_digest,
            }
            for stage in ("planner", "tool_policy", "frontend", "review", "security")
        }
    run_manifest = run_root / "run_manifest.json"
    _write_json(run_manifest, {"groups": groups})

    def verify(_: Path, __: Path, *, group: str) -> dict[str, str]:
        return {"registry_sha256": locks[group]["registry_sha256"]}

    return FormalAdapterCatalog(
        tmp_path,
        {
            "runtime_bindings": runtime,
            "registry_locks": locks,
            "serial_runtime_contract": {
                "base_model_id": "gemma4-12b-base-q4",
                "maximum_active_loras": 1,
            },
        },
        run_manifest,
        registry_verifier=verify,
    )


def test_catalog_accepts_only_frozen_registry_name_path_and_digest(tmp_path: Path) -> None:
    catalog = _catalog(tmp_path)
    binding = catalog.validate_load("formal-e", tmp_path / "adapters" / "E")

    assert binding.group == "E"
    assert binding.adapter_sha256 == "e" * 64
    with pytest.raises(RequestContractError, match="path does not match"):
        catalog.validate_load("formal-e", tmp_path / "adapters" / "F")
    with pytest.raises(RequestContractError, match="unregistered"):
        catalog.validate_load("unregistered", tmp_path / "adapters" / "E")


def test_catalog_fails_closed_if_registry_digest_changes(tmp_path: Path) -> None:
    catalog = _catalog(tmp_path)
    catalog._verify_registry = lambda *_args, **_kwargs: {"registry_sha256": "0" * 64}

    with pytest.raises(RequestContractError, match="registry digest changed"):
        catalog.rehash_load("formal-c", tmp_path / "adapters" / "C")


class _FakeCatalog:
    base_model_id = "gemma4-12b-base-q4"
    adapter_model_ids = frozenset({"formal-c"})

    def validate_load(self, name: str, path: str) -> AdapterBinding:
        if name != "formal-c" or path != "C:/frozen/c":
            raise RequestContractError("not frozen")
        return AdapterBinding(name, "C", "planner-r16", Path(path), "c" * 64, "d" * 64)

    def rehash_load(self, name: str, path: str) -> AdapterBinding:
        return self.validate_load(name, path)


class _FakeEngine:
    def __init__(self) -> None:
        self._active: str | None = None
        self.events: list[tuple[str, str]] = []

    @property
    def active_adapter(self) -> str | None:
        return self._active

    @property
    def loaded(self) -> bool:
        return True

    def load_adapter(self, binding: AdapterBinding) -> None:
        assert self._active is None
        self._active = binding.model_id
        self.events.append(("load", binding.model_id))

    def unload_adapter(self, name: str) -> None:
        assert self._active == name
        self._active = None
        self.events.append(("unload", name))

    def generate(self, request: Any) -> GenerationResult:
        expected = self._active or "gemma4-12b-base-q4"
        if request.model != expected:
            raise RequestContractError("active mismatch", status=409)
        self.events.append(("generate", request.model))
        return GenerationResult("ok", 7, 1, "stop")

    def generate_stream(self, request: Any, emit: Any) -> GenerationResult:
        expected = self._active or "gemma4-12b-base-q4"
        if request.model != expected:
            raise RequestContractError("active mismatch", status=409)
        self.events.append(("stream", request.model))
        emit("ready", 7)
        emit("content", "o")
        emit("content", "k")
        return GenerationResult("ok", 7, 2, "stop")

    def close(self) -> None:
        self._active = None


def test_service_exposes_base_only_and_serializes_adapter_protocol() -> None:
    engine = _FakeEngine()
    service = FormalSerialService(_FakeCatalog(), engine, token_cap=1024)  # type: ignore[arg-type]

    assert [item["id"] for item in service.model_catalog()["data"]] == [
        "gemma4-12b-base-q4"
    ]
    service.load_adapter({"lora_name": "formal-c", "lora_path": "C:/frozen/c"})
    response = service.complete(
        {
            "model": "formal-c",
            "messages": [{"role": "user", "content": "public synthetic prompt"}],
            "max_tokens": 8,
            "temperature": 0,
        }
    )
    service.unload_adapter({"lora_name": "formal-c"})

    assert response["choices"][0]["message"]["content"] == "ok"
    assert response["usage"] == {
        "prompt_tokens": 7,
        "completion_tokens": 1,
        "total_tokens": 8,
    }
    assert engine.events == [
        ("load", "formal-c"),
        ("generate", "formal-c"),
        ("unload", "formal-c"),
    ]
    assert service.probe()["maximum_active_loras"] == 1


def test_aiohttp_surface_matches_formal_run_protocol() -> None:
    from aiohttp.test_utils import TestClient, TestServer

    async def exercise() -> None:
        service = FormalSerialService(_FakeCatalog(), _FakeEngine(), token_cap=1024)  # type: ignore[arg-type]
        client = TestClient(TestServer(create_app(service, api_key="local-test-key")))
        await client.start_server()
        try:
            unauthorized = await client.get("/v1/models")
            assert unauthorized.status == 401
            headers = {"Authorization": "Bearer local-test-key"}
            models = await client.get("/v1/models", headers=headers)
            assert models.status == 200
            assert [item["id"] for item in (await models.json())["data"]] == [
                "gemma4-12b-base-q4"
            ]
            probed = await client.post(
                "/v1/probe_lora_adapter",
                headers=headers,
                json={"lora_name": "formal-c", "lora_path": "C:/frozen/c"},
            )
            assert (await probed.json())["loaded"] is False
            loaded = await client.post(
                "/v1/load_lora_adapter",
                headers=headers,
                json={"lora_name": "formal-c", "lora_path": "C:/frozen/c"},
            )
            assert loaded.status == 200
            completion = await client.post(
                "/v1/chat/completions",
                headers=headers,
                json={
                    "model": "formal-c",
                    "messages": [{"role": "user", "content": "synthetic"}],
                    "temperature": 0,
                    "max_tokens": 4,
                },
            )
            assert completion.status == 200
            assert (await completion.json())["object"] == "chat.completion"
            unloaded = await client.post(
                "/v1/unload_lora_adapter",
                headers=headers,
                json={"lora_name": "formal-c"},
            )
            assert unloaded.status == 200
        finally:
            await client.close()

    asyncio.run(exercise())


def test_aiohttp_stream_forwards_content_before_generation_finishes() -> None:
    from aiohttp.test_utils import TestClient, TestServer

    class BlockingStreamEngine(_FakeEngine):
        def __init__(self) -> None:
            super().__init__()
            self.first_emitted = threading.Event()
            self.release = threading.Event()

        def generate_stream(self, request: Any, emit: Any) -> GenerationResult:
            emit("ready", 7)
            emit("content", "hel")
            self.first_emitted.set()
            if not self.release.wait(timeout=5):
                raise RuntimeError("test stream release timed out")
            emit("content", "lo")
            return GenerationResult("hello", 7, 2, "stop")

    async def exercise() -> None:
        engine = BlockingStreamEngine()
        service = FormalSerialService(_FakeCatalog(), engine, token_cap=1024)  # type: ignore[arg-type]
        client = TestClient(TestServer(create_app(service)))
        await client.start_server()
        try:
            response = await client.post(
                "/v1/chat/completions",
                json={
                    "model": "gemma4-12b-base-q4",
                    "messages": [{"role": "user", "content": "synthetic"}],
                    "stream": True,
                    "temperature": 0,
                    "max_tokens": 8,
                },
            )
            assert response.status == 200
            prefix = bytearray()
            while b'"content":"hel"' not in prefix:
                prefix.extend(
                    await asyncio.wait_for(response.content.readline(), timeout=1)
                )
            assert engine.first_emitted.is_set()
            assert not engine.release.is_set()
            engine.release.set()
            tail = await asyncio.wait_for(response.read(), timeout=2)
            complete = bytes(prefix) + tail
            assert b'"content":"lo"' in complete
            assert b'"finish_reason":"stop"' in complete
            assert b'"usage"' in complete
            assert complete.endswith(b"data: [DONE]\n\n")
        finally:
            engine.release.set()
            await client.close()

    asyncio.run(exercise())


def test_agent_parser_preserves_tools_calls_and_tool_results() -> None:
    payload = {
        "model": "base",
        "messages": [
            {"role": "system", "content": "Use the declared tools when useful."},
            {"role": "user", "content": "Inspect the fixture."},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "read_file",
                            "arguments": '{"path":"README.md"}',
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_1",
                "content": "synthetic fixture result",
            },
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "read_file",
                    "description": "Read one sandbox file.",
                    "parameters": {
                        "type": "object",
                        "properties": {"path": {"type": "string"}},
                        "required": ["path"],
                    },
                },
            }
        ],
        "tool_choice": "auto",
        "max_completion_tokens": 64,
        "temperature": 0.2,
        "top_p": 0.9,
        "stream": True,
    }

    parsed = parse_agent_chat_request(
        payload, allowed_models=frozenset({"base"}), token_cap=128
    )

    assert parsed.agent_mode is True
    assert parsed.stream is True
    assert parsed.messages[2]["tool_calls"][0]["function"]["name"] == "read_file"
    assert parsed.messages[3]["tool_call_id"] == "call_1"
    assert parsed.tools[0]["function"]["parameters"]["required"] == ["path"]


def test_formal_parser_remains_frozen_and_rejects_tools() -> None:
    with pytest.raises(RequestContractError, match="unsupported formal request fields"):
        parse_chat_request(
            {
                "model": "base",
                "messages": [{"role": "user", "content": "synthetic"}],
                "tools": [],
            },
            allowed_models=frozenset({"base"}),
            token_cap=128,
        )


def test_openai_agent_message_serializes_processor_tool_arguments() -> None:
    message = _openai_agent_message(
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "type": "function",
                    "function": {
                        "name": "read_file",
                        "arguments": {"path": "README.md"},
                    },
                }
            ],
        }
    )

    assert message["content"] is None
    assert message["tool_calls"][0]["function"] == {
        "name": "read_file",
        "arguments": '{"path":"README.md"}',
    }
    assert message["tool_calls"][0]["id"].startswith("call_anchor_0_")


def test_agent_endpoint_emits_openai_tool_call_nonstream_and_sse() -> None:
    from aiohttp.test_utils import TestClient, TestServer

    class AgentEngine(_FakeEngine):
        def generate(self, request: Any) -> GenerationResult:
            assert request.agent_mode is True
            assert request.tools[0]["function"]["name"] == "read_file"
            message = {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_synthetic",
                        "type": "function",
                        "function": {
                            "name": "read_file",
                            "arguments": '{"path":"README.md"}',
                        },
                    }
                ],
            }
            return GenerationResult("", 23, 9, "tool_calls", message=message)

    async def exercise() -> None:
        service = FormalSerialService(_FakeCatalog(), AgentEngine(), token_cap=128)  # type: ignore[arg-type]
        client = TestClient(TestServer(create_app(service)))
        await client.start_server()
        request = {
            "model": "gemma4-12b-base-q4",
            "messages": [{"role": "user", "content": "Inspect the fixture."}],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "read_file",
                        "description": "Read one sandbox file.",
                        "parameters": {"type": "object", "properties": {}},
                    },
                }
            ],
            "max_tokens": 32,
        }
        try:
            models = await client.get("/agent/v1/models")
            assert [item["id"] for item in (await models.json())["data"]] == [
                "gemma4-12b-base-q4"
            ]
            nonstream = await client.post(
                "/agent/v1/chat/completions", json=request
            )
            body = await nonstream.json()
            assert nonstream.status == 200
            assert body["choices"][0]["finish_reason"] == "tool_calls"
            assert body["choices"][0]["message"]["tool_calls"][0]["function"][
                "name"
            ] == "read_file"

            streamed = await client.post(
                "/agent/v1/chat/completions", json={**request, "stream": True}
            )
            stream_body = await streamed.read()
            assert streamed.status == 200
            assert b'"tool_calls"' in stream_body
            assert b'"finish_reason":"tool_calls"' in stream_body
            assert stream_body.endswith(b"data: [DONE]\n\n")
        finally:
            await client.close()

    asyncio.run(exercise())


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("temperature", 0.1, "temperature=0"),
        ("top_p", 0.9, "top_p=1"),
        ("max_tokens", 1025, "between 1 and 1024"),
    ],
)
def test_request_parser_rejects_formal_sampling_drift(
    field: str, value: Any, message: str
) -> None:
    payload = {
        "model": "base",
        "messages": [{"role": "user", "content": "synthetic"}],
        field: value,
    }
    with pytest.raises(RequestContractError, match=message):
        parse_chat_request(payload, allowed_models=frozenset({"base"}), token_cap=1024)


def test_request_parser_accepts_boolean_stream_and_rejects_non_boolean() -> None:
    payload = {
        "model": "base",
        "messages": [{"role": "user", "content": "synthetic"}],
        "stream": True,
    }
    parsed = parse_chat_request(
        payload, allowed_models=frozenset({"base"}), token_cap=1024
    )
    assert parsed.stream is True

    for invalid in (1, "true", None):
        payload["stream"] = invalid
        with pytest.raises(RequestContractError, match="stream must be a boolean"):
            parse_chat_request(
                payload, allowed_models=frozenset({"base"}), token_cap=1024
            )


def test_generation_stop_tokens_include_artifact_declared_end_of_turn() -> None:
    class Tokenizer:
        eos_token_id = 1
        eot_token = "<turn|>"
        unk_token = "<unk>"
        unk_token_id = 3
        init_kwargs = {
            "eot_token": "<turn|>",
            "model_specific_special_tokens": {"eot_token": "<turn|>"},
        }

        @staticmethod
        def convert_tokens_to_ids(token: str) -> int:
            return {"<turn|>": 106}.get(token, 3)

    class Config:
        eos_token_id = 1

    class Model:
        generation_config = Config()
        config = Config()

    assert _generation_stop_token_ids(Tokenizer(), Model()) == [1, 106]


def test_generation_stop_tokens_ignore_unknown_eot_metadata() -> None:
    class Tokenizer:
        eos_token_id = 1
        eot_token = "<not-in-vocab>"
        unk_token = "<unk>"
        unk_token_id = 3
        init_kwargs: dict[str, Any] = {}

        @staticmethod
        def convert_tokens_to_ids(_token: str) -> int:
            return 3

    class Model:
        generation_config = None
        config = None

    assert _generation_stop_token_ids(Tokenizer(), Model()) == [1]


def test_synthetic_processor_and_nf4_artifact_hash_gates(tmp_path: Path) -> None:
    processor = tmp_path / "processor"
    processor.mkdir()
    chat = processor / "chat_template.jinja"
    chat.write_text("synthetic template", encoding="utf-8")
    files = [
        {
            "path": chat.name,
            "bytes": chat.stat().st_size,
            "sha256": _sha(chat),
        }
    ]
    processor_manifest = tmp_path / "processor.json"
    _write_json(
        processor_manifest,
        {
            "schema_version": "anchor.formal-af-processor.v1",
            "processor_path": "processor",
            "files": files,
            "tree_sha256": hashlib.sha256(
                json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest(),
        },
    )
    assert verify_processor_artifact(tmp_path, processor_manifest) == processor

    base = tmp_path / "base"
    base.mkdir()
    shard = base / "model.safetensors"
    shard.write_bytes(b"synthetic-nf4")
    quant_manifest = base / "anchor_quantization_manifest.json"
    _write_json(
        quant_manifest,
        {
            "schema_version": "anchor.bnb-nf4-export.v1",
            "quantization": {
                "type": "nf4",
                "double_quant": True,
                "compute_dtype": "bfloat16",
                "storage_dtype": "bfloat16",
            },
            "weights": [
                {
                    "path": shard.name,
                    "bytes": shard.stat().st_size,
                    "sha256": _sha(shard),
                }
            ],
        },
    )
    _write_json(
        base / "config.json",
        {
            "quantization_config": {
                "quant_method": "bitsandbytes",
                "load_in_4bit": True,
                "load_in_8bit": False,
                "bnb_4bit_quant_type": "nf4",
                "bnb_4bit_use_double_quant": True,
                "bnb_4bit_compute_dtype": "bfloat16",
            }
        },
    )
    run = tmp_path / "run.json"
    _write_json(
        run,
        {
            "base_artifact": {
                "manifest_path": "base/anchor_quantization_manifest.json",
                "manifest_sha256": _sha(quant_manifest),
            }
        },
    )
    assert verify_base_artifact(tmp_path, base, run) == base
    shard.write_bytes(b"drift")
    with pytest.raises(WindowsSerialServerError, match="base shard changed"):
        verify_base_artifact(tmp_path, base, run)


def test_segmented_server_requires_the_processor_manifest_bound_by_preflight(
    tmp_path: Path,
) -> None:
    bound = tmp_path / "processor.json"
    alternate = tmp_path / "alternate.json"
    _write_json(bound, {"tree_sha256": "b" * 64})
    _write_json(alternate, {"tree_sha256": "b" * 64})
    manifest_sha = _sha(bound)
    config = {
        "processor_binding": {
            "manifest_path": "processor.json",
            "manifest_sha256": manifest_sha,
            "tree_sha256": "b" * 64,
        }
    }
    result = {
        "processor_manifest_sha256": manifest_sha,
        "processor_tree_sha256": "b" * 64,
    }

    assert (
        _resolve_processor_manifest_binding(tmp_path, bound, config, result) == bound
    )
    with pytest.raises(WindowsSerialServerError, match="formal config binding"):
        _resolve_processor_manifest_binding(tmp_path, alternate, config, result)
    with pytest.raises(WindowsSerialServerError, match="formal preflight"):
        _resolve_processor_manifest_binding(
            tmp_path,
            bound,
            config,
            {**result, "processor_tree_sha256": "c" * 64},
        )


def test_windows_launchers_keep_native_paths_and_explicit_heldout_gate() -> None:
    server = SERVER_LAUNCHER.read_text(encoding="utf-8")
    benchmark = BENCH_LAUNCHER.read_text(encoding="utf-8")

    assert "anchor_mvp.serving.windows_serial_server" in server
    assert '[string]$FormalConfig = "configs/benchmark/formal_partial_v1_af.json"' in server
    assert '[string]$BaseModel = "models/google-gemma-4-12B-bnb-nf4"' in server
    assert (
        '[string]$ProcessorManifest = '
        '"configs/serving/formal_af_windows_processor.json"' in server
    )
    assert '"--formal-config", $FormalConfig' in server
    assert '"--base-model", $BaseModel' in server
    assert '"--processor-manifest", $ProcessorManifest' in server
    assert '"--host", "127.0.0.1"' in server
    assert '"--api-key-env", "ANCHOR_VLLM_API_KEY"' in server
    assert "[switch]$PreflightOnly" in server
    assert "[switch]$PrintCommand" in server
    assert "[switch]$DisableUnpaddedDecodeFastPath" in server
    assert "wsl.exe" not in server
    assert "[switch]$Execute" in benchmark
    assert "[switch]$AuthorizeHeldoutAccess" in benchmark
    assert "[switch]$Resume" in benchmark
    assert 'if ($Resume) { $Arguments += "--resume" }' in benchmark
    assert '"--serial-runtime-lora"' in benchmark
    assert '"--authorize-heldout-access"' in benchmark
    assert "--server-project-root" not in benchmark
    assert "wsl.exe" not in benchmark


class _Scalar:
    def __init__(self, value: bool) -> None:
        self.value = value

    def item(self) -> bool:
        return self.value


class _SyntheticMask:
    ndim = 2

    def __init__(self, shape: tuple[int, int], *, all_value: bool, any_value: bool) -> None:
        self.shape = shape
        self._all = all_value
        self._any = any_value

    def all(self) -> _Scalar:
        return _Scalar(self._all)

    def any(self) -> _Scalar:
        return _Scalar(self._any)


def test_unpadded_text_fast_path_requires_batch_one_all_one_text_mask() -> None:
    ids = _SyntheticMask((1, 7), all_value=True, any_value=True)
    padding = _SyntheticMask((1, 7), all_value=True, any_value=True)
    text_types = _SyntheticMask((1, 7), all_value=False, any_value=False)

    assert _is_unpadded_text_batch(
        {
            "input_ids": ids,
            "attention_mask": padding,
            "mm_token_type_ids": text_types,
        }
    )
    assert not _is_unpadded_text_batch(
        {
            "input_ids": ids,
            "attention_mask": _SyntheticMask(
                (1, 7), all_value=False, any_value=True
            ),
            "mm_token_type_ids": text_types,
        }
    )
    assert not _is_unpadded_text_batch(
        {
            "input_ids": ids,
            "attention_mask": padding,
            "mm_token_type_ids": _SyntheticMask(
                (1, 7), all_value=False, any_value=True
            ),
        }
    )


def test_unpadded_decode_hook_drops_only_the_flagged_attention_mask() -> None:
    class SyntheticGenerationModel:
        def _update_model_kwargs_for_generation(
            self, _outputs: Any, model_kwargs: dict[str, Any]
        ) -> dict[str, Any]:
            return dict(model_kwargs)

    model = SyntheticGenerationModel()
    _install_unpadded_decode_fast_path(model)
    _install_unpadded_decode_fast_path(model)

    retained = model._update_model_kwargs_for_generation(
        None, {"attention_mask": "all-ones", "position_ids": "positions"}
    )
    setattr(model, "_anchor_drop_unpadded_decode_attention_mask", True)
    dropped = model._update_model_kwargs_for_generation(
        None, {"attention_mask": "all-ones", "position_ids": "positions"}
    )

    assert retained == {
        "attention_mask": "all-ones",
        "position_ids": "positions",
    }
    assert dropped == {"position_ids": "positions"}

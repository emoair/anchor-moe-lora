from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from anchor_mvp.benchmark.serial_backend import (
    SerialBackendError,
    SerialLoraBackend,
    require_loopback_http_url,
)
from anchor_mvp.serving import (
    CompletionRequest,
    CompletionResponse,
    Message,
)


class _Backend:
    def __init__(self) -> None:
        self.models: list[str] = []
        self.active = 0
        self.maximum_active = 0

    async def complete(self, request: CompletionRequest) -> CompletionResponse:
        self.active += 1
        self.maximum_active = max(self.maximum_active, self.active)
        await asyncio.sleep(0)
        self.models.append(request.model)
        self.active -= 1
        return CompletionResponse(content="ok", model=request.model)


class _Admin:
    def __init__(self) -> None:
        self.events: list[tuple[str, str, str | None]] = []
        self.active: str | None = None

    async def load(self, name: str, path: str) -> None:
        assert self.active is None
        self.active = name
        self.events.append(("load", name, path))

    async def unload(self, name: str) -> None:
        assert self.active == name
        self.events.append(("unload", name, None))
        self.active = None


def _runtime(root: Path) -> dict:
    planner = root / "adapters" / "planner-r8"
    frontend = root / "adapters" / "frontend-r16"
    planner.mkdir(parents=True)
    frontend.mkdir(parents=True)
    base = {
        stage: {
            "model_id": "gemma4-12b-base-q4",
            "adapter_dir": None,
            "adapter_sha256": None,
        }
        for stage in ("planner", "tool_policy", "frontend", "review", "security")
    }
    groups = {"A": base}
    for group in "BCDEF":
        groups[group] = {}
        for stage in ("planner", "tool_policy", "frontend", "review", "security"):
            frontend_stage = stage == "frontend"
            groups[group][stage] = {
                "model_id": f"fpv1-{group.lower()}-{stage}",
                "adapter_dir": (
                    "adapters/frontend-r16" if frontend_stage else "adapters/planner-r8"
                ),
                "adapter_sha256": "b" * 64 if frontend_stage else "a" * 64,
            }
    return groups


def _request(model: str) -> CompletionRequest:
    return CompletionRequest(model=model, messages=(Message("user", "public input"),))


def test_serial_backend_keeps_at_most_one_adapter_and_maps_base(tmp_path: Path) -> None:
    inner, admin = _Backend(), _Admin()
    backend = SerialLoraBackend(
        inner,
        admin,
        _runtime(tmp_path),
        project_root=tmp_path,
        server_project_root="/mnt/d/project",
    )

    async def exercise() -> None:
        await backend.complete(_request("gemma4-12b-base-q4"))
        await backend.complete(_request("fpv1-e-planner"))
        await backend.complete(_request("fpv1-e-planner"))
        await backend.complete(_request("fpv1-e-frontend"))
        await backend.complete(_request("gemma4-12b-base-q4"))
        await backend.close()

    asyncio.run(exercise())

    assert inner.models == [
        "gemma4-12b-base-q4",
        "fpv1-e-planner",
        "fpv1-e-planner",
        "fpv1-e-frontend",
        "gemma4-12b-base-q4",
    ]
    assert admin.events == [
        ("load", "fpv1-e-planner", "/mnt/d/project/adapters/planner-r8"),
        ("unload", "fpv1-e-planner", None),
        ("load", "fpv1-e-frontend", "/mnt/d/project/adapters/frontend-r16"),
        ("unload", "fpv1-e-frontend", None),
    ]
    assert admin.active is None
    assert backend.stats == {
        "mode": "serial_runtime_lora",
        "maximum_active_loras": 1,
        "active_adapter": None,
        "adapter_loads": 2,
        "adapter_unloads": 2,
        "requests": 5,
    }


def test_probe_loads_and_unloads_before_case_access(tmp_path: Path) -> None:
    inner, admin = _Backend(), _Admin()
    backend = SerialLoraBackend(inner, admin, _runtime(tmp_path), project_root=tmp_path)

    asyncio.run(backend.probe())

    assert inner.models == []
    assert [event[0] for event in admin.events] == ["load", "unload"]
    assert backend.stats["active_adapter"] is None


def test_prepare_record_excludes_previous_arm_cleanup_from_next_arm(tmp_path: Path) -> None:
    inner, admin = _Backend(), _Admin()
    backend = SerialLoraBackend(inner, admin, _runtime(tmp_path), project_root=tmp_path)

    async def exercise() -> None:
        await backend.complete(_request("fpv1-f-security"))
        assert backend.stats["active_adapter"] == "fpv1-f-security"
        await backend.prepare_record()
        assert backend.stats["active_adapter"] is None
        await backend.complete(_request("gemma4-12b-base-q4"))

    asyncio.run(exercise())

    assert admin.events[-1] == ("unload", "fpv1-f-security", None)
    assert inner.models[-1] == "gemma4-12b-base-q4"


def test_concurrent_requests_are_serialized(tmp_path: Path) -> None:
    inner, admin = _Backend(), _Admin()
    backend = SerialLoraBackend(inner, admin, _runtime(tmp_path), project_root=tmp_path)

    async def exercise() -> None:
        await asyncio.gather(
            backend.complete(_request("fpv1-c-planner")),
            backend.complete(_request("fpv1-f-frontend")),
        )
        await backend.close()

    asyncio.run(exercise())
    assert inner.maximum_active == 1
    assert admin.active is None


def test_unsafe_or_missing_adapter_binding_is_rejected(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    runtime["F"]["frontend"]["adapter_dir"] = "../heldout/private-adapter"

    with pytest.raises(SerialBackendError, match="unsafe directory"):
        SerialLoraBackend(_Backend(), _Admin(), runtime, project_root=tmp_path)


def test_unregistered_model_fails_without_admin_transition(tmp_path: Path) -> None:
    admin = _Admin()
    backend = SerialLoraBackend(
        _Backend(), admin, _runtime(tmp_path), project_root=tmp_path
    )

    with pytest.raises(SerialBackendError, match="unregistered formal model"):
        asyncio.run(backend.complete(_request("not-registered")))
    assert admin.events == []


def test_runtime_admin_must_remain_loopback_only() -> None:
    assert require_loopback_http_url(
        "http://127.0.0.1:8000/", label="admin"
    ) == "http://127.0.0.1:8000"
    assert require_loopback_http_url("http://[::1]:8000", label="admin") == (
        "http://[::1]:8000"
    )
    with pytest.raises(SerialBackendError, match="loopback"):
        require_loopback_http_url("http://0.0.0.0:8000", label="admin")
    with pytest.raises(SerialBackendError, match="loopback"):
        require_loopback_http_url("https://example.com", label="admin")


def test_server_project_root_must_be_absolute_and_normalized(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    for server_root in ("relative/project", "/mnt/d/project/../other"):
        with pytest.raises(SerialBackendError, match="absolute normalized"):
            SerialLoraBackend(
                _Backend(),
                _Admin(),
                runtime,
                project_root=tmp_path,
                server_project_root=server_root,
            )

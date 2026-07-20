"""One-active-LoRA OpenAI-compatible backend for the formal A--F run.

The frozen NF4 base remains resident in the serving process.  Before each
request this wrapper unloads the previous adapter and, when needed, loads the
single PEFT adapter bound to the requested formal model id.  Calls are guarded
by one asyncio lock, so concurrent benchmark code cannot make two adapters
resident at once.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, replace
from pathlib import Path, PurePosixPath
import re
from typing import Any, Mapping, Protocol
from urllib.parse import urlsplit

from ..serving import CompletionBackend, CompletionRequest, CompletionResponse


class SerialBackendError(RuntimeError):
    """The frozen serial runtime map or an adapter transition is invalid."""


FORMAL_STAGES = {"planner", "tool_policy", "frontend", "review", "security"}
_HEX64 = re.compile(r"[0-9a-f]{64}")


def require_loopback_http_url(value: str, *, label: str) -> str:
    """Reject a runtime-LoRA admin surface that is not local-only."""

    parsed = urlsplit(value)
    if parsed.scheme not in {"http", "https"} or parsed.hostname not in {
        "127.0.0.1",
        "localhost",
        "::1",
    }:
        raise SerialBackendError(f"{label} must use a loopback HTTP(S) URL")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise SerialBackendError(f"{label} contains forbidden URL components")
    return value.rstrip("/")


class AdapterAdmin(Protocol):
    async def load(self, name: str, path: str) -> None: ...

    async def unload(self, name: str) -> None: ...


@dataclass(frozen=True)
class SerialBinding:
    model_id: str
    adapter_dir: str | None
    adapter_sha256: str | None


class SerialLoraBackend:
    """Translate frozen model ids into serialized runtime-LoRA transitions."""

    def __init__(
        self,
        backend: CompletionBackend,
        admin: AdapterAdmin,
        runtime_bindings: Mapping[str, Mapping[str, Mapping[str, Any]]],
        *,
        project_root: str | Path,
        server_project_root: str | None = None,
    ) -> None:
        self.backend = backend
        self.admin = admin
        self.project_root = Path(project_root).resolve()
        self.server_project_root = (
            PurePosixPath(server_project_root) if server_project_root else None
        )
        if self.server_project_root is not None and (
            not self.server_project_root.is_absolute()
            or ".." in self.server_project_root.parts
        ):
            raise SerialBackendError(
                "server_project_root must be an absolute normalized POSIX path"
            )
        self._bindings, self.base_model_id = self._freeze_bindings(runtime_bindings)
        self._lock = asyncio.Lock()
        self._active_adapter: str | None = None
        self._load_count = 0
        self._unload_count = 0
        self._request_count = 0

    @property
    def stats(self) -> dict[str, Any]:
        return {
            "mode": "serial_runtime_lora",
            "maximum_active_loras": 1,
            "active_adapter": self._active_adapter,
            "adapter_loads": self._load_count,
            "adapter_unloads": self._unload_count,
            "requests": self._request_count,
        }

    def _freeze_bindings(
        self,
        runtime: Mapping[str, Mapping[str, Mapping[str, Any]]],
    ) -> tuple[dict[str, SerialBinding], str]:
        if set(runtime) != set("ABCDEF"):
            raise SerialBackendError("runtime bindings must contain exactly A through F")
        bindings: dict[str, SerialBinding] = {}
        base_ids: set[str] = set()
        for group, stages in runtime.items():
            if not isinstance(stages, Mapping) or set(stages) != FORMAL_STAGES:
                raise SerialBackendError(
                    f"group {group} must bind exactly the five formal stages"
                )
            for stage, raw in stages.items():
                if not isinstance(raw, Mapping):
                    raise SerialBackendError(f"group {group} stage {stage} is invalid")
                model_id = str(raw.get("model_id", ""))
                if not model_id:
                    raise SerialBackendError(f"group {group} stage {stage} has no model id")
                relative = raw.get("adapter_dir")
                digest = raw.get("adapter_sha256")
                if group == "A":
                    if relative is not None or digest is not None:
                        raise SerialBackendError("group A must not bind an adapter")
                    base_ids.add(model_id)
                    candidate = SerialBinding(model_id, None, None)
                else:
                    candidate = self._adapter_binding(model_id, relative, digest)
                previous = bindings.get(model_id)
                if previous is not None and previous != candidate:
                    raise SerialBackendError(
                        f"model id {model_id!r} maps to multiple adapter artifacts"
                    )
                bindings[model_id] = candidate
        if len(base_ids) != 1:
            raise SerialBackendError("group A must use one frozen base model id")
        return bindings, base_ids.pop()

    def _adapter_binding(
        self, model_id: str, relative: Any, digest: Any
    ) -> SerialBinding:
        if not isinstance(relative, str) or not relative:
            raise SerialBackendError(f"adapter {model_id!r} has no directory")
        portable = PurePosixPath(relative.replace("\\", "/"))
        if portable.is_absolute() or ".." in portable.parts:
            raise SerialBackendError(f"adapter {model_id!r} has an unsafe directory")
        local = (self.project_root / Path(*portable.parts)).resolve()
        try:
            local.relative_to(self.project_root)
        except ValueError as exc:  # pragma: no cover - guarded by PurePosixPath
            raise SerialBackendError(
                f"adapter {model_id!r} escapes the project root"
            ) from exc
        if not local.is_dir():
            raise SerialBackendError(f"adapter directory does not exist: {local}")
        if not isinstance(digest, str) or not _HEX64.fullmatch(digest):
            raise SerialBackendError(f"adapter {model_id!r} has no frozen digest")
        server_path = (
            str(self.server_project_root.joinpath(portable))
            if self.server_project_root is not None
            else str(local)
        )
        return SerialBinding(model_id, server_path, digest)

    async def probe(self) -> None:
        """Prove the local admin endpoint before any held-out case is opened."""

        candidate = next(
            (item for item in self._bindings.values() if item.adapter_dir is not None),
            None,
        )
        if candidate is None:
            raise SerialBackendError("no adapter exists for the runtime-LoRA probe")
        async with self._lock:
            await self._activate(candidate)
            await self._deactivate()

    async def complete(self, request: CompletionRequest) -> CompletionResponse:
        binding = self._bindings.get(request.model)
        if binding is None:
            raise SerialBackendError(f"unregistered formal model id: {request.model}")
        async with self._lock:
            await self._activate(binding)
            self._request_count += 1
            transport_model = (
                self.base_model_id if binding.adapter_dir is None else binding.model_id
            )
            return await self.backend.complete(replace(request, model=transport_model))

    async def close(self) -> None:
        async with self._lock:
            await self._deactivate()

    async def prepare_record(self) -> None:
        """Remove prior-arm state before the next record's latency clock starts."""

        await self.close()

    async def _activate(self, binding: SerialBinding) -> None:
        target = binding.model_id if binding.adapter_dir is not None else None
        if self._active_adapter == target:
            return
        await self._deactivate()
        if target is None:
            return
        assert binding.adapter_dir is not None
        try:
            await self.admin.load(target, binding.adapter_dir)
        except Exception as exc:
            self._active_adapter = None
            raise SerialBackendError(f"failed to load adapter {target!r}: {exc}") from exc
        self._active_adapter = target
        self._load_count += 1

    async def _deactivate(self) -> None:
        if self._active_adapter is None:
            return
        active = self._active_adapter
        try:
            await self.admin.unload(active)
        except Exception as exc:
            raise SerialBackendError(f"failed to unload adapter {active!r}: {exc}") from exc
        self._active_adapter = None
        self._unload_count += 1

from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path
import socket
import subprocess
import sys
import threading
import time
from types import SimpleNamespace
from typing import Any, Mapping

import pytest

from anchor_mvp.tooling.swebench_execution_v3 import (
    DISTILLATION_VALIDATION_STATE_SCHEMA,
    ExecutionContractError,
    distillation_tool_evidence,
    distillation_validation_state_sha256,
    sha256_file,
    verify_distillation_execution_receipt,
)
from anchor_mvp.tooling.policy import ToolPolicy
from anchor_mvp.tooling.swebench_runtime_v3 import (
    IMAGE_CACHE_BINDING_SCHEMA,
    OfficialEvalExecution,
    OfficialGrade,
    OfficialHarnessTask,
    OfficialImageAcquisitionRequest,
    OfficialImageCacheBinding,
    SWEbenchV3RuntimeAdapter,
    V3WorkspaceHandle,
    WslPodmanV3Transport,
    _HOST_UNIX_ROUTE_RELAY,
    _SUPERVISOR_IMAGE_CACHE,
    issue_distillation_execution_receipt_after_cleanup,
)


ROOT = Path(__file__).resolve().parents[1]
LOCK = ROOT / "configs" / "tooling" / "swebench_execution_v3.lock.json"
TASK_ID = "swe-full-v1:" + "a" * 64
CHECKPOINT_ID = "b" * 64
BASE_COMMIT = "c" * 40
IMAGE_KEY = "swebench/sweb.eval.x86_64.example_1776_example-1:latest"
IMAGE_DIGEST = "sha256:" + "d" * 64
PATCH = b"diff --git a/a.js b/a.js\nindex e69de29..2e65efe 100644\n--- a/a.js\n+++ b/a.js\n@@ -0,0 +1 @@\n+x = 1\n"
KEY = b"test-supervisor-key-material-32bytes-minimum"


def test_distillation_receipt_is_written_after_bound_cleanup_evidence(
    tmp_path: Path,
) -> None:
    command = "anchor-validate test"
    command_sha = hashlib.sha256(command.encode()).hexdigest()
    invocation_sha = hashlib.sha256(b"invocation").hexdigest()
    builder = {
        "tool_calls": [
            {
                "sequence": 1,
                "tool": "bash",
                "command": command,
                "command_sha256": command_sha,
                "invocation_sha256": invocation_sha,
                "execution_scope": "isolated-instance-container",
            }
        ],
        "tool_results": [
            {
                "sequence": 1,
                "tool": "bash",
                "status": "completed",
                "exit_code": 0,
                "command_sha256": command_sha,
                "invocation_sha256": invocation_sha,
                "output_sha256": hashlib.sha256(b"test output").hexdigest(),
                "visible_to_model": True,
                "provenance": "public-repo-validation-model-visible",
                "execution_scope": "isolated-instance-container",
            }
        ],
    }
    patch_sha = hashlib.sha256(PATCH).hexdigest()
    validator_version_sha = hashlib.sha256(b"validator-version").hexdigest()
    changed_files = [
        {"path": "a.js", "sha256": hashlib.sha256(b"x = 1\n").hexdigest()}
    ]
    builder["validation_state"] = {
        "schema_version": DISTILLATION_VALIDATION_STATE_SCHEMA,
        "final_patch_sha256": patch_sha,
        "changed_files": changed_files,
        "changed_files_sha256": hashlib.sha256(
            json.dumps(
                changed_files,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest(),
        "terminal_validation_output_sha256": builder["tool_results"][0][
            "output_sha256"
        ],
        "terminal_command_sha256": command_sha,
        "validator_version_sha256": validator_version_sha,
        "validator_result": {
            "schema_version": "anchor.train-sandbox-validation.v1",
            "validator_version": "1.0.1",
            "mode": "test",
            "success": True,
            "not_official_swebench_pass": True,
            "validation_level": "native_test",
            "changed_paths": ["a.js"],
            "changed_paths_sha256": hashlib.sha256(
                json.dumps(
                    ["a.js"],
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode()
            ).hexdigest(),
            "final_state_sha256": hashlib.sha256(b"final-state").hexdigest(),
            "validators": ["node-check", "npm-test"],
        },
    }
    transcript_sha, validation_sha = distillation_tool_evidence(builder)
    validation_state_sha = distillation_validation_state_sha256(
        builder,
        final_patch_sha256=patch_sha,
        validator_version_sha256=validator_version_sha,
    )
    task_digest = hashlib.sha256(TASK_ID.encode()).hexdigest()
    bindings = {
        "checkpoint_id": CHECKPOINT_ID,
        "config_sha256": hashlib.sha256(b"config").hexdigest(),
        "execution_lock_sha256": hashlib.sha256(b"lock").hexdigest(),
        "source_bank_manifest_sha256": hashlib.sha256(
            b"bank-manifest"
        ).hexdigest(),
        "candidate_task_artifact_sha256": hashlib.sha256(
            b"task-artifact"
        ).hexdigest(),
        "candidate_work_order_artifacts_sha256": hashlib.sha256(
            b"order-artifacts"
        ).hexdigest(),
        "task_id_sha256": task_digest,
        "instance_id_sha256": hashlib.sha256(b"example__example-1").hexdigest(),
        "repo_sha256": hashlib.sha256(b"example/example").hexdigest(),
        "base_commit": BASE_COMMIT,
        "image_digest": IMAGE_DIGEST,
        "image_id_sha256": hashlib.sha256(b"image-id").hexdigest(),
        "final_patch_sha256": patch_sha,
        "tool_transcript_sha256": transcript_sha,
        "validation_evidence_sha256": validation_sha,
        "validation_state_sha256": validation_state_sha,
        "validator_version_sha256": validator_version_sha,
        "lineage_sha256": hashlib.sha256(b"lineage").hexdigest(),
    }
    receipt_path = issue_distillation_execution_receipt_after_cleanup(
        private_root=tmp_path / "system-private",
        bindings=bindings,
        final_patch=PATCH,
        builder_output=builder,
        trusted_receipt_key=KEY,
        issued_at="2026-07-18T00:00:00Z",
    )
    assert receipt_path == (
        tmp_path
        / "system-private"
        / task_digest
        / "distillation-execution-receipt.json"
    )
    assert receipt_path.with_name("final.patch").read_bytes() == PATCH
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert verify_distillation_execution_receipt(
        receipt,
        trusted_receipt_key=KEY,
        expected_bindings=bindings,
    )
    assert "official" not in receipt["status"].casefold()


def _task() -> dict[str, Any]:
    return {
        "source": {
            "instance_id": "example__example-1",
            "repo": "example/example",
            "base_commit": BASE_COMMIT,
        }
    }


class FakeHarness:
    def __init__(self, *, resolved: bool = True) -> None:
        self.resolved = resolved
        self.spec = SimpleNamespace(
            eval_script="printf official-eval",
            base_image_key="sweb.base.example.x86_64:latest",
            env_image_key="sweb.env.example.x86_64.hash:latest",
            platform="linux/x86_64",
            base_dockerfile="FROM docker.io/library/ubuntu:22.04\n",
            env_dockerfile="FROM sweb.base.example.x86_64:latest\nCOPY setup_env.sh /root/\n",
            instance_dockerfile="FROM sweb.env.example.x86_64.hash:latest\nCOPY setup_repo.sh /root/\n",
            setup_env_script="#!/bin/bash\ntrue\n",
            install_repo_script="#!/bin/bash\ntrue\n",
        )

    def resolve(
        self,
        *,
        instance_id: str,
        expected_repo: str,
        expected_base_commit: str,
    ) -> OfficialHarnessTask:
        assert instance_id == "example__example-1"
        assert expected_repo == "example/example"
        assert expected_base_commit == BASE_COMMIT
        return OfficialHarnessTask(
            instance={"instance_id": instance_id},
            test_spec=self.spec,
            image_key=IMAGE_KEY,
        )

    def grade(
        self,
        *,
        task: OfficialHarnessTask,
        patch: bytes,
        test_output: bytes,
        private_directory: Path,
    ) -> OfficialGrade:
        assert task.test_spec is self.spec
        assert patch == PATCH
        assert test_output == b"official output"
        private_directory.mkdir(parents=True, exist_ok=True)
        report = b'{"private":true}\n'
        (private_directory / "official-report.json").write_bytes(report)
        return OfficialGrade(
            resolved=self.resolved,
            report_hash=hashlib.sha256(report).hexdigest(),
        )


class FakeTransport:
    def __init__(self, *, eval_exit_code: int = 0) -> None:
        self.eval_exit_code = eval_exit_code
        self.patch = PATCH
        self.applied: list[bytes] = []
        self.cleanup_count = 0
        self.eval_calls = 0
        self.acquire_calls = 0
        self.verify_calls = 0
        self.cache_valid = True

    @staticmethod
    def _binding(
        request: OfficialImageAcquisitionRequest,
    ) -> OfficialImageCacheBinding:
        unsigned = {
            "execution_lock_sha256": request.execution_lock_sha256,
            "dataset_revision": request.dataset_revision,
            "task_id_sha256": request.task_id_sha256,
            "instance_id_sha256": request.instance_id_sha256,
            "base_commit": request.base_commit,
            "image_key": request.image_key,
            "image_digest": IMAGE_DIGEST,
            "image_ref": request.image_key.rsplit(":", 1)[0] + "@" + IMAGE_DIGEST,
            "recipe_sha256": request.recipe_sha256(),
            "acquisition_mode": "pull",
        }
        binding_sha256 = hashlib.sha256(
            json.dumps(
                unsigned,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        return OfficialImageCacheBinding.from_mapping(
            {
                "schema_version": IMAGE_CACHE_BINDING_SCHEMA,
                **unsigned,
                "binding_sha256": binding_sha256,
                "ledger_content_sha256": "e" * 64,
            }
        )

    def acquire_official_image(
        self, request: OfficialImageAcquisitionRequest
    ) -> OfficialImageCacheBinding:
        self.acquire_calls += 1
        if not self.cache_valid:
            raise ExecutionContractError("fake_cache_invalid")
        return self._binding(request)

    def verify_cached_official_image(
        self, request: OfficialImageAcquisitionRequest
    ) -> OfficialImageCacheBinding:
        self.verify_calls += 1
        if not self.cache_valid:
            raise ExecutionContractError("fake_cache_invalid")
        return self._binding(request)

    def inspect_image_digest(self, image_key: str) -> str:
        assert image_key == IMAGE_KEY
        return IMAGE_DIGEST

    def materialize_testbed(self, **kwargs: Any) -> tuple[str, str, Path, str]:
        assert kwargs["image_ref"] == IMAGE_KEY.rsplit(":", 1)[0] + "@" + IMAGE_DIGEST
        return (
            "/var/lib/anchor/swebench-v3/live/materialized",
            "/var/lib/anchor/swebench-v3/live/materialized/testbed",
            Path("/supervisor/private/materialized/testbed"),
            "materialization-1",
        )

    def capture_binary_diff(self, handle: V3WorkspaceHandle) -> bytes:
        assert handle.canonical_testbed == "/testbed"
        return self.patch

    def workspace_inventory(self, handle: V3WorkspaceHandle) -> Mapping[str, Any]:
        del handle
        return {"files": ["a.py"]}

    def run_model(self, **kwargs: Any) -> Any:
        raise AssertionError("not used by receipt tests")

    def apply_binary_diff(self, handle: V3WorkspaceHandle, patch: bytes) -> None:
        del handle
        self.applied.append(patch)

    def run_official_eval(self, **kwargs: Any) -> OfficialEvalExecution:
        self.eval_calls += 1
        handle = kwargs["handle"]
        patch = kwargs["patch"]
        return OfficialEvalExecution(
            exit_code=self.eval_exit_code,
            timed_out=False,
            duration_ms=12.5,
            stdout=b"official output",
            stderr=b"",
            fresh_container=True,
            network_mode="none",
            image_ref=handle.image_ref,
            patch_sha256=hashlib.sha256(patch).hexdigest(),
        )

    def cleanup(self, handle: V3WorkspaceHandle) -> None:
        del handle
        self.cleanup_count += 1


def _adapter(
    tmp_path: Path,
    *,
    resolved: bool = True,
    eval_exit_code: int = 0,
    key: bytes = KEY,
) -> tuple[SWEbenchV3RuntimeAdapter, FakeTransport]:
    transport = FakeTransport(eval_exit_code=eval_exit_code)
    adapter = SWEbenchV3RuntimeAdapter(
        project_root=ROOT,
        lock_path=LOCK,
        expected_lock_sha256=sha256_file(LOCK),
        private_root=tmp_path / "system-private",
        official_eval_timeout_seconds=30,
        harness=FakeHarness(resolved=resolved),
        transport=transport,
        receipt_key=key,
    )
    return adapter, transport


def test_runtime_uses_official_image_testbed_and_restores_exact_binary_diff(
    tmp_path: Path,
) -> None:
    adapter, transport = _adapter(tmp_path)
    handle = adapter.prepare_task(TASK_ID, _task())
    assert handle.canonical_testbed == "/testbed"
    assert handle.image_ref.endswith("@" + IMAGE_DIGEST)
    assert handle.sandbox_contract()["network_mode"] == "none"
    adapter.restore_binary_diff(handle, PATCH)
    assert transport.applied == [PATCH]


@pytest.mark.parametrize(
    ("resolved", "expected_outcome"),
    [(True, "completed"), (False, "failed")],
)
def test_signed_receipt_is_terminal_only_with_exact_bindings(
    tmp_path: Path, resolved: bool, expected_outcome: str
) -> None:
    adapter, transport = _adapter(tmp_path, resolved=resolved)
    handle = adapter.prepare_task(TASK_ID, _task())
    finalized = adapter.finalize(
        handle=handle,
        expected_cumulative_diff=PATCH,
        checkpoint_id=CHECKPOINT_ID,
        revision=2,
    )
    assert finalized["gold_eligible"] is resolved
    assert transport.eval_calls == 1
    assert (
        adapter.finalization_outcome(
            TASK_ID,
            _task(),
            checkpoint_id=CHECKPOINT_ID,
            revision=2,
        )
        == expected_outcome
    )
    receipt_path = (
        tmp_path
        / "system-private"
        / hashlib.sha256(TASK_ID.encode()).hexdigest()
        / "official-eval-receipt.json"
    )
    binding_path = receipt_path.with_name("representative-runtime-binding.json")
    binding = json.loads(binding_path.read_text(encoding="utf-8"))
    assert binding["content_free"] is True
    assert binding["task_id_sha256"] == hashlib.sha256(TASK_ID.encode()).hexdigest()
    assert binding["image_key_sha256"] == hashlib.sha256(IMAGE_KEY.encode()).hexdigest()
    content_sha256 = binding.pop("content_sha256")
    assert content_sha256 == hashlib.sha256(
        json.dumps(
            binding, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode()
    ).hexdigest()
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["revision"] = 3
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    assert (
        adapter.finalization_outcome(
            TASK_ID,
            _task(),
            checkpoint_id=CHECKPOINT_ID,
            revision=2,
        )
        is None
    )


def test_key_rotation_invalidates_old_receipt_and_forces_resume_revalidation(
    tmp_path: Path,
) -> None:
    adapter, _ = _adapter(tmp_path)
    handle = adapter.prepare_task(TASK_ID, _task())
    adapter.finalize(
        handle=handle,
        expected_cumulative_diff=PATCH,
        checkpoint_id=CHECKPOINT_ID,
        revision=1,
    )
    rotated, _ = _adapter(tmp_path, key=b"rotated-supervisor-key-material-32bytes")
    assert (
        rotated.finalization_outcome(
            TASK_ID,
            _task(),
            checkpoint_id=CHECKPOINT_ID,
            revision=1,
        )
        is None
    )


def test_nonzero_official_eval_exit_cannot_be_forged_as_success(tmp_path: Path) -> None:
    adapter, _ = _adapter(tmp_path, eval_exit_code=9)
    handle = adapter.prepare_task(TASK_ID, _task())
    with pytest.raises(ExecutionContractError, match="official_eval_isolation_failed"):
        adapter.finalize(
            handle=handle,
            expected_cumulative_diff=PATCH,
            checkpoint_id=CHECKPOINT_ID,
            revision=1,
        )
    assert (
        adapter.finalization_outcome(
            TASK_ID,
            _task(),
            checkpoint_id=CHECKPOINT_ID,
            revision=1,
        )
        is None
    )


def test_final_diff_mismatch_never_invokes_official_evaluator(tmp_path: Path) -> None:
    adapter, transport = _adapter(tmp_path)
    handle = adapter.prepare_task(TASK_ID, _task())
    with pytest.raises(ExecutionContractError, match="final_diff_binding_failed"):
        adapter.finalize(
            handle=handle,
            expected_cumulative_diff=b"different",
            checkpoint_id=CHECKPOINT_ID,
            revision=1,
        )
    assert transport.eval_calls == 0


def test_cache_drift_before_hidden_eval_fails_closed(tmp_path: Path) -> None:
    adapter, transport = _adapter(tmp_path)
    handle = adapter.prepare_task(TASK_ID, _task())
    transport.cache_valid = False
    with pytest.raises(ExecutionContractError, match="fake_cache_invalid"):
        adapter.finalize(
            handle=handle,
            expected_cumulative_diff=PATCH,
            checkpoint_id=CHECKPOINT_ID,
            revision=1,
        )
    assert transport.eval_calls == 0


def test_image_request_and_binding_are_content_hash_bound() -> None:
    harness = FakeHarness()
    request = OfficialImageAcquisitionRequest.from_test_spec(
        execution_lock_sha256="f" * 64,
        dataset_revision="1" * 40,
        task_id=TASK_ID,
        instance_id="example__example-1",
        base_commit=BASE_COMMIT,
        image_key=IMAGE_KEY,
        test_spec=harness.spec,
    )
    assert request.recipe_sha256() == request.recipe_sha256()
    binding = FakeTransport._binding(request)
    assert binding.matches(request)
    tampered = {
        "schema_version": IMAGE_CACHE_BINDING_SCHEMA,
        **{
            name: getattr(binding, name)
            for name in binding.__dataclass_fields__
        },
        "image_digest": "sha256:" + "0" * 64,
    }
    with pytest.raises(ExecutionContractError, match="cache_binding_invalid"):
        OfficialImageCacheBinding.from_mapping(tampered)


def test_supervisor_acquisition_has_race_safe_ledger_and_is_only_networked_path() -> None:
    assert "fcntl.LOCK_EX" in _SUPERVISOR_IMAGE_CACHE
    assert "image_lock" in _SUPERVISOR_IMAGE_CACHE
    assert "task_lock" not in _SUPERVISOR_IMAGE_CACHE
    assert _SUPERVISOR_IMAGE_CACHE.count("load_ledger(ledger_path,request)") >= 2
    assert "['podman','pull'" in _SUPERVISOR_IMAGE_CACHE
    assert "--policy=always" in _SUPERVISOR_IMAGE_CACHE
    assert "['podman','build'" in _SUPERVISOR_IMAGE_CACHE
    assert "--network=host" in _SUPERVISOR_IMAGE_CACHE
    assert "image_cache_private_root_insecure" in _SUPERVISOR_IMAGE_CACHE
    assert "image_cache_ledger_permissions_invalid" in _SUPERVISOR_IMAGE_CACHE
    resume_source = inspect.getsource(WslPodmanV3Transport._verify_cached_request)
    assert "podman','pull" not in resume_source
    assert "podman','build" not in resume_source
    materialize = inspect.getsource(WslPodmanV3Transport.materialize_testbed)
    evaluator = inspect.getsource(WslPodmanV3Transport.run_official_eval)
    assert "--pull=never" in materialize and "--network none" in materialize
    assert "--pull=never" in evaluator and "--network=none" in evaluator


def test_private_native_writer_is_root_owned_scope_and_atomic() -> None:
    source = inspect.getsource(WslPodmanV3Transport._write_native)
    assert "p.relative_to(root)" in source
    assert "s.st_uid==0" in source
    assert "stat.S_IMODE(s.st_mode)==0o700" in source
    assert "os.O_EXCL" in source
    assert "os.fsync" in source
    assert "os.replace" in source


def test_model_config_bytes_stage_only_below_supervisor_private_root(
    tmp_path: Path,
) -> None:
    adapter, _ = _adapter(tmp_path)
    handle = adapter.prepare_task(TASK_ID, _task())
    transport = WslPodmanV3Transport(
        wsl_distro="Ubuntu-22.04",
        native_root="/var/lib/anchor/swebench-v3",
    )
    copied: list[tuple[Path, str]] = []
    written: list[tuple[str, bytes]] = []
    transport._copy_windows_file_to_native = lambda source, destination: copied.append(  # type: ignore[method-assign]
        (source, destination)
    )
    transport._write_native = lambda path, value: written.append((path, value))  # type: ignore[method-assign]

    def stop_after_staging(**kwargs: Any) -> str:
        del kwargs
        raise ExecutionContractError("stop_after_private_stage")

    transport._start_route_relay = stop_after_staging  # type: ignore[method-assign]
    config_bytes = b'{"permission":{"*":"deny"}}'
    with pytest.raises(ExecutionContractError, match="stop_after_private_stage"):
        transport.run_model(
            handle=handle,
            linux_opencode=Path("D:/private-build/opencode"),
            config_bytes=config_bytes,
            provider_id="anchor-test",
            model_id="model-test",
            variant="max",
            prompt="public task",
            policy=ToolPolicy(
                allowed_tools=("edit", "bash"),
                allowed_commands=("anchor-validate test",),
            ),
            route_host="127.0.0.1",
            route_port=12345,
            sample_id="sample-1",
        )
    assert copied == [
        (
            Path("D:/private-build/opencode"),
            handle.native_root + "/model-private/opencode",
        )
    ]
    assert written == [
        (handle.native_root + "/model-private/opencode.json", config_bytes)
    ]


def test_fixed_target_unix_relay_never_forwards_second_or_unapproved_request(
    tmp_path: Path,
) -> None:
    if not hasattr(socket, "AF_UNIX"):
        pytest.skip("AF_UNIX unavailable")
    upstream = socket.socket()
    upstream.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    upstream.bind(("127.0.0.1", 0))
    upstream.listen(4)
    upstream.settimeout(2)
    received: list[bytes] = []

    def serve() -> None:
        while len(received) < 2:
            try:
                connection, _ = upstream.accept()
            except TimeoutError:
                return
            with connection:
                chunks: list[bytes] = []
                while True:
                    chunk = connection.recv(4096)
                    if not chunk:
                        break
                    chunks.append(chunk)
                received.append(b"".join(chunks))
                connection.sendall(
                    b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\nConnection: close\r\n\r\n{}"
                )

    server = threading.Thread(target=serve, daemon=True)
    server.start()
    route_path = tmp_path / "route.sock"
    relay = subprocess.Popen(
        [
            sys.executable,
            "-c",
            _HOST_UNIX_ROUTE_RELAY,
            str(route_path),
            "127.0.0.1",
            str(upstream.getsockname()[1]),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        deadline = time.monotonic() + 3
        while not route_path.exists() and relay.poll() is None and time.monotonic() < deadline:
            time.sleep(0.02)
        if not route_path.exists():
            pytest.skip("local AF_UNIX subprocess unavailable")

        def request(payload: bytes) -> bytes:
            client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            client.settimeout(2)
            try:
                client.connect(str(route_path))
                client.sendall(payload)
                chunks: list[bytes] = []
                while True:
                    try:
                        chunk = client.recv(4096)
                    except (ConnectionResetError, TimeoutError):
                        break
                    if not chunk:
                        break
                    chunks.append(chunk)
                return b"".join(chunks)
            finally:
                client.close()

        request(
            b"GET /anchor/health HTTP/1.1\r\nHost: 127.0.0.1:18080\r\n\r\n"
            b"GET /v1/forbidden HTTP/1.1\r\nHost: 127.0.0.1:18080\r\n\r\n"
        )
        request(b"GET /v1/admin HTTP/1.1\r\nHost: 127.0.0.1:18080\r\n\r\n")
        response = request(
            b"GET /v1/models HTTP/1.1\r\nHost: 127.0.0.1:18080\r\n"
            b"Connection: keep-alive\r\n\r\n"
        )
        assert b"200 OK" in response
        server.join(timeout=3)
        assert received
        assert all(b"/v1/forbidden" not in item for item in received)
        assert all(b"/v1/admin" not in item for item in received)
        assert all(item.count(b" HTTP/1.1") == 1 for item in received)
        assert all(b"Connection: close" in item for item in received)
    finally:
        relay.terminate()
        relay.wait(timeout=5)
        upstream.close()

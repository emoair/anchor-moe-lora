from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest

from anchor_mvp.tooling.route_diagnostics import (
    ROUTE_FAILURE_DIAGNOSTIC_NAME,
    ROUTE_FAILURE_DIAGNOSTIC_SCHEMA,
    RouteDiagnosticSource,
    build_route_failure_diagnostic,
    validate_route_failure_diagnostic,
    write_route_failure_diagnostic,
)


ROOT = Path(__file__).resolve().parents[1]
COORDINATOR = ROOT / "scripts" / "tooling" / "run_swebench_ccswitch.py"
SPEC = importlib.util.spec_from_file_location(
    "route_diagnostic_coordinator",
    COORDINATOR,
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


FIXED_NOW = datetime(2026, 7, 18, 8, 9, 10, 123000, tzinfo=timezone.utc)


def test_route_failure_summary_never_copies_stderr_content(tmp_path: Path) -> None:
    secret = "sk-test-SHOULD-NOT-SURVIVE"
    token = "url-token-SHOULD-NOT-SURVIVE"
    body = "private request body SHOULD-NOT-SURVIVE"
    stderr = tmp_path / "route.stderr.log"
    raw = (
        f"Unauthorized: {secret}\n"
        f"https://provider.invalid/v1?token={token}\n"
        f"body={body}\n"
    ).encode()
    stderr.write_bytes(raw)

    diagnostic = build_route_failure_diagnostic(
        startup_error_code="ccswitch_route_exited",
        sources=[
            RouteDiagnosticSource(
                route_alias="glm52_max",
                exit_code=17,
                stderr_path=stderr,
            )
        ],
        now=FIXED_NOW,
    )

    validate_route_failure_diagnostic(
        diagnostic,
        expected_startup_error_code="ccswitch_route_exited",
        expected_classified_error_code=(
            "ccswitch_route_exited_authentication_failure"
        ),
    )
    serialized = json.dumps(diagnostic, sort_keys=True)
    assert secret not in serialized
    assert token not in serialized
    assert body not in serialized
    assert "https://" not in serialized
    assert diagnostic["schema_version"] == ROUTE_FAILURE_DIAGNOSTIC_SCHEMA
    assert diagnostic["content_free"] is True
    assert diagnostic["observed_at"] == "2026-07-18T08:09:10.123Z"
    assert diagnostic["routes"] == [
        {
            "route_alias": "glm52_max",
            "exit_code": 17,
            "stderr_class": "authentication_failure",
            "stderr_bytes": len(raw),
            "observed_at": "2026-07-18T08:09:10.123Z",
            "stderr_modified_at": diagnostic["routes"][0]["stderr_modified_at"],
        }
    ]


def test_route_failure_writer_classifies_only_bounded_tail_and_is_atomic(
    tmp_path: Path,
) -> None:
    stderr = tmp_path / "route.stderr.log"
    raw = b"private-prefix\n" + (b"x" * (70 * 1024)) + b"\nEADDRINUSE\n"
    stderr.write_bytes(raw)
    output = tmp_path / ROUTE_FAILURE_DIAGNOSTIC_NAME

    diagnostic = write_route_failure_diagnostic(
        output,
        startup_error_code="ccswitch_route_health_timeout",
        sources=[
            RouteDiagnosticSource(
                route_alias="kimi_k3_max",
                exit_code=None,
                stderr_path=stderr,
            )
        ],
        now=FIXED_NOW,
    )

    assert output.is_file()
    assert not output.with_name(output.name + ".tmp").exists()
    assert json.loads(output.read_text(encoding="utf-8")) == diagnostic
    assert diagnostic["routes"][0]["stderr_class"] == "listen_address_conflict"
    assert diagnostic["routes"][0]["stderr_bytes"] == len(raw)
    assert diagnostic["classified_error_code"] == (
        "ccswitch_route_health_timeout_listen_address_conflict"
    )
    assert "private-prefix" not in output.read_text(encoding="utf-8")


@pytest.mark.parametrize(
    "stderr_line",
    [
        (
            "listen tcp 192.0.2.44:43821: bind: The requested address is not "
            "valid in its context."
        ),
        "Error: listen EADDRNOTAVAIL: address not available 192.0.2.44:43821",
        "listen tcp 192.0.2.44:43821: bind: cannot assign requested address",
    ],
)
def test_unavailable_listen_address_is_content_free(
    tmp_path: Path,
    stderr_line: str,
) -> None:
    stderr = tmp_path / "route.stderr.log"
    stderr.write_text(stderr_line, encoding="utf-8")

    diagnostic = build_route_failure_diagnostic(
        startup_error_code="ccswitch_route_exited",
        sources=[
            RouteDiagnosticSource(
                route_alias="glm52_max",
                exit_code=1,
                stderr_path=stderr,
            )
        ],
        now=FIXED_NOW,
    )

    validate_route_failure_diagnostic(
        diagnostic,
        expected_startup_error_code="ccswitch_route_exited",
        expected_classified_error_code=(
            "ccswitch_route_exited_listen_address_unavailable"
        ),
    )
    serialized = json.dumps(diagnostic, sort_keys=True)
    assert diagnostic["routes"][0]["stderr_class"] == (
        "listen_address_unavailable"
    )
    assert stderr_line not in serialized
    assert "192.0.2.44" not in serialized
    assert "43821" not in serialized


def test_windows_socket_10049_in_139_bytes_is_listen_address_unavailable(
    tmp_path: Path,
) -> None:
    # The observed localized Windows launcher failure exposed no stable English
    # phrase.  Error 10049 is WSAEADDRNOTAVAIL and is therefore the portable,
    # content-free signal.  Keep this fixture at the observed byte count while
    # using synthetic text rather than copying the raw launcher message.
    prefix = b"local route startup failed: windows socket error 10049\n"
    raw = prefix + (b"x" * (139 - len(prefix)))
    assert len(raw) == 139
    stderr = tmp_path / "route.stderr.log"
    stderr.write_bytes(raw)

    diagnostic = build_route_failure_diagnostic(
        startup_error_code="ccswitch_route_exited",
        sources=[
            RouteDiagnosticSource(
                route_alias="glm52_max",
                exit_code=1,
                stderr_path=stderr,
            )
        ],
        now=FIXED_NOW,
    )

    validate_route_failure_diagnostic(
        diagnostic,
        expected_startup_error_code="ccswitch_route_exited",
        expected_classified_error_code=(
            "ccswitch_route_exited_listen_address_unavailable"
        ),
    )
    serialized = json.dumps(diagnostic, sort_keys=True)
    assert diagnostic["routes"][0]["stderr_class"] == (
        "listen_address_unavailable"
    )
    assert diagnostic["routes"][0]["stderr_bytes"] == 139
    assert raw.decode("ascii") not in serialized


@pytest.mark.parametrize(
    ("stderr_line", "expected_class"),
    [
        (
            "A parameter cannot be found that matches parameter name 'NetworkMode'",
            "launcher_argument_invalid",
        ),
        (
            "CC Switch route manifest/profile validation failed",
            "profile_or_manifest_invalid",
        ),
        ("Patched CC Switch binary hash mismatch", "component_integrity_failure"),
        ("Credential env 'PRIVATE_NAME' is not set", "credential_missing"),
        (
            "Proxy URL must be an absolute https URL with token=private",
            "network_configuration_invalid",
        ),
        ("Model discovery was ambiguous", "model_discovery_failure"),
    ],
)
def test_known_launcher_failures_reduce_to_finite_classes(
    tmp_path: Path,
    stderr_line: str,
    expected_class: str,
) -> None:
    stderr = tmp_path / "route.stderr.log"
    stderr.write_text(stderr_line, encoding="utf-8")

    diagnostic = build_route_failure_diagnostic(
        startup_error_code="ccswitch_route_exited",
        sources=[
            RouteDiagnosticSource(
                route_alias="glm52_max",
                exit_code=1,
                stderr_path=stderr,
            )
        ],
        now=FIXED_NOW,
    )

    assert diagnostic["routes"][0]["stderr_class"] == expected_class
    assert diagnostic["classified_error_code"] == (
        f"ccswitch_route_exited_{expected_class}"
    )
    assert stderr_line not in json.dumps(diagnostic, sort_keys=True)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value.update({"stderr": "raw secret"}),
        lambda value: value.update({"classified_error_code": "ccswitch_route_exited"}),
        lambda value: value["routes"][0].update({"stderr_class": "raw secret"}),
        lambda value: value["routes"][0].update({"stderr_bytes": -1}),
        lambda value: value["routes"][0].update(
            {"observed_at": "2026-07-18T08:09:10Z"}
        ),
    ],
)
def test_route_failure_validator_rejects_noncanonical_or_content_bearing_shape(
    tmp_path: Path,
    mutation: object,
) -> None:
    stderr = tmp_path / "route.stderr.log"
    stderr.write_text("Unauthorized", encoding="utf-8")
    diagnostic = build_route_failure_diagnostic(
        startup_error_code="ccswitch_route_exited",
        sources=[
            RouteDiagnosticSource(
                route_alias="glm52_max",
                exit_code=1,
                stderr_path=stderr,
            )
        ],
        now=FIXED_NOW,
    )
    tampered = deepcopy(diagnostic)
    assert callable(mutation)
    mutation(tampered)

    with pytest.raises(ValueError):
        validate_route_failure_diagnostic(tampered)


class _ExitedProcess:
    def __init__(self, returncode: int) -> None:
        self.returncode = returncode

    def poll(self) -> int:
        return self.returncode


def _bare_backend(tmp_path: Path, *, startup_error_code: str | None) -> object:
    route_root = tmp_path / "route-runtime"
    route_root.mkdir(parents=True)
    first_stderr = route_root / "glm52_max.stderr.log"
    second_stderr = route_root / "kimi_k3_max.stderr.log"
    first_stderr.write_text("", encoding="utf-8")
    second_stderr.write_text(
        "Unauthorized sk-test-secret https://provider.invalid/?token=hidden",
        encoding="utf-8",
    )

    backend = object.__new__(MODULE.LiveBackend)
    backend.config = SimpleNamespace(
        routes={"glm52_max": object(), "kimi_k3_max": object()},
        runtime=SimpleNamespace(output_dir=tmp_path, retain_router_state=False),
    )
    backend._processes = [_ExitedProcess(0), _ExitedProcess(19)]
    backend._route_log_handles = [
        first_stderr.open("ab"),
        second_stderr.open("ab"),
    ]
    backend._route_startup_error_code = startup_error_code
    backend._route_failure_public_code = None
    backend._route_diagnostic_path = tmp_path / ROUTE_FAILURE_DIAGNOSTIC_NAME
    return backend


def test_failed_close_keeps_only_summary_and_removes_route_runtime(
    tmp_path: Path,
) -> None:
    backend = _bare_backend(tmp_path, startup_error_code="ccswitch_route_exited")

    backend.close()

    diagnostic_path = tmp_path / ROUTE_FAILURE_DIAGNOSTIC_NAME
    assert diagnostic_path.is_file()
    assert not (tmp_path / "route-runtime").exists()
    diagnostic = json.loads(diagnostic_path.read_text(encoding="utf-8"))
    validate_route_failure_diagnostic(
        diagnostic,
        expected_startup_error_code="ccswitch_route_exited",
        expected_classified_error_code=backend._route_failure_public_code,
    )
    assert diagnostic["routes"][0]["exit_code"] == 0
    assert diagnostic["routes"][1]["exit_code"] == 19
    assert "sk-test-secret" not in diagnostic_path.read_text(encoding="utf-8")
    assert "provider.invalid" not in diagnostic_path.read_text(encoding="utf-8")


def test_normal_close_removes_route_runtime_without_creating_summary(
    tmp_path: Path,
) -> None:
    backend = _bare_backend(tmp_path, startup_error_code=None)

    backend.close()

    assert not (tmp_path / "route-runtime").exists()
    assert not (tmp_path / ROUTE_FAILURE_DIAGNOSTIC_NAME).exists()
    assert backend._route_failure_public_code is None

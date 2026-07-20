"""Content-free diagnostics for short-lived local route processes.

The raw router logs may contain credentials, request bodies, or provider URLs and
must remain temporary.  This module deliberately reduces stderr to a finite
classification plus byte/timestamp metadata before those logs are removed.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
from typing import Any, Iterable, Mapping


ROUTE_FAILURE_DIAGNOSTIC_SCHEMA = (
    "anchor.ccswitch-route-failure-diagnostic.v1"
)
ROUTE_FAILURE_DIAGNOSTIC_NAME = "route-failure-diagnostic.json"
ROUTE_STARTUP_ERROR_CODES = frozenset(
    {
        "ccswitch_route_exited",
        "ccswitch_route_health_timeout",
        "ccswitch_route_not_visible_from_wsl",
    }
)
_MAX_CLASSIFICATION_BYTES = 64 * 1024
_SAFE_ROUTE_ALIAS = re.compile(r"^[a-z0-9_]{1,64}$")
_SAFE_ERROR_CODE = re.compile(r"^[a-z0-9_]{1,80}$")
_UTC_MILLISECOND_TIMESTAMP = re.compile(
    r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z$"
)
_STDERR_CLASSES = frozenset(
    {
        "authentication_failure",
        "component_integrity_failure",
        "connection_failure",
        "credential_missing",
        "dns_resolution_failure",
        "filesystem_permission_denied",
        "launcher_argument_invalid",
        "listen_address_conflict",
        "listen_address_unavailable",
        "model_discovery_failure",
        "network_configuration_invalid",
        "profile_or_manifest_invalid",
        "provider_rate_limited",
        "runtime_dependency_missing",
        "runtime_exception",
        "stderr_empty",
        "stderr_missing",
        "stderr_unclassified",
        "stderr_unreadable",
        "tls_failure",
    }
)


@dataclass(frozen=True)
class RouteDiagnosticSource:
    """Trusted process metadata and the temporary stderr file to classify."""

    route_alias: str
    exit_code: int | None
    stderr_path: Path


def _timestamp(value: datetime) -> str:
    normalized = value.astimezone(timezone.utc)
    return normalized.isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _stderr_tail(path: Path, size: int) -> bytes:
    with path.open("rb") as handle:
        if size > _MAX_CLASSIFICATION_BYTES:
            handle.seek(size - _MAX_CLASSIFICATION_BYTES, os.SEEK_SET)
        return handle.read(_MAX_CLASSIFICATION_BYTES)


def _classify_stderr_bytes(value: bytes) -> str:
    if not value:
        return "stderr_empty"
    text = value.decode("utf-8", errors="replace").casefold()
    classifications: tuple[tuple[str, tuple[str, ...]], ...] = (
        (
            "launcher_argument_invalid",
            (
                "parameter cannot be found that matches parameter name",
                "namedparameternotfound",
                "cannot bind parameter",
            ),
        ),
        (
            "listen_address_conflict",
            (
                "address already in use",
                "eaddrinuse",
                "only one usage of each socket address",
                "failed to bind",
            ),
        ),
        (
            "listen_address_unavailable",
            (
                "10049",
                "eaddrnotavail",
                "wsaeaddrnotavail",
                "addrnotavailable",
                "cannot assign requested address",
                "requested address is not valid in its context",
                "address not available",
            ),
        ),
        (
            "profile_or_manifest_invalid",
            (
                "invalid profile",
                "invalid manifest",
                "profile validation",
                "manifest validation",
                "manifest/profile validation",
                "failed to parse profile",
                "failed to parse manifest",
            ),
        ),
        (
            "component_integrity_failure",
            (
                "binary hash mismatch",
                "binary checksum mismatch",
                "component hash mismatch",
                "component checksum mismatch",
            ),
        ),
        (
            "runtime_dependency_missing",
            (
                "command not found",
                "is not recognized as an internal or external command",
                "no such file or directory",
                "cannot find module",
                "module not found",
                "binary is missing",
                "not a valid win32 application",
            ),
        ),
        (
            "credential_missing",
            (
                "credential env",
                "credential environment variable",
                "requires env",
                "required environment variable is not set",
            ),
        ),
        (
            "network_configuration_invalid",
            (
                "network mode proxy requires",
                "proxy url environment variable name is invalid",
                "proxy url must be an absolute",
                "direct mode is not physically direct",
                "physical-route preflight",
            ),
        ),
        (
            "model_discovery_failure",
            (
                "model discovery was ambiguous",
                "model discovery failed",
                "manual fallback id",
            ),
        ),
        (
            "filesystem_permission_denied",
            ("permission denied", "access is denied", "eacces", "eperm"),
        ),
        (
            "authentication_failure",
            (
                "unauthorized",
                "authentication failed",
                "invalid api key",
                "invalid token",
                "status code: 401",
                "http 401",
                "status code: 403",
                "http 403",
                "forbidden",
            ),
        ),
        (
            "provider_rate_limited",
            ("rate limit", "too many requests", "status code: 429", "http 429"),
        ),
        (
            "dns_resolution_failure",
            (
                "getaddrinfo",
                "name or service not known",
                "temporary failure in name resolution",
                "dns lookup failed",
            ),
        ),
        (
            "tls_failure",
            (
                "certificate verify failed",
                "certificate validation",
                "tls handshake",
                "ssl handshake",
                "unknown certificate authority",
            ),
        ),
        (
            "connection_failure",
            (
                "connection refused",
                "connection reset",
                "network is unreachable",
                "connect timeout",
                "connection timed out",
            ),
        ),
        (
            "runtime_exception",
            ("uncaught exception", "unhandled exception", "panic:", "fatal error"),
        ),
    )
    for classification, markers in classifications:
        if any(marker in text for marker in markers):
            return classification
    return "stderr_unclassified"


def _route_entry(
    source: RouteDiagnosticSource,
    *,
    observed_at: str,
) -> dict[str, Any]:
    if not _SAFE_ROUTE_ALIAS.fullmatch(source.route_alias):
        raise ValueError("unsafe route alias")
    if source.exit_code is not None and (
        isinstance(source.exit_code, bool) or not isinstance(source.exit_code, int)
    ):
        raise ValueError("invalid route exit code")

    stderr_bytes = 0
    stderr_modified_at: str | None = None
    try:
        stat = source.stderr_path.stat()
    except FileNotFoundError:
        stderr_class = "stderr_missing"
    except OSError:
        stderr_class = "stderr_unreadable"
    else:
        stderr_bytes = stat.st_size
        stderr_modified_at = _timestamp(
            datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc)
        )
        try:
            stderr_class = _classify_stderr_bytes(
                _stderr_tail(source.stderr_path, stderr_bytes)
            )
        except OSError:
            stderr_class = "stderr_unreadable"

    return {
        "route_alias": source.route_alias,
        "exit_code": source.exit_code,
        "stderr_class": stderr_class,
        "stderr_bytes": stderr_bytes,
        "observed_at": observed_at,
        "stderr_modified_at": stderr_modified_at,
    }


def _classified_error_code(
    startup_error_code: str,
    routes: list[Mapping[str, Any]],
) -> str:
    # The latest attempted route is the best health-timeout candidate.  An
    # actually exited process is stronger evidence and wins when present.
    primary = routes[-1] if routes else None
    for route in reversed(routes):
        if route["exit_code"] is not None:
            primary = route
            break
    suffix = str(primary["stderr_class"]) if primary is not None else "stderr_missing"
    candidate = f"{startup_error_code}_{suffix}"
    if _SAFE_ERROR_CODE.fullmatch(candidate):
        return candidate
    return "ccswitch_route_startup_failed_diagnostic_unavailable"


def _require_timestamp(value: object, label: str) -> str:
    if not isinstance(value, str) or not _UTC_MILLISECOND_TIMESTAMP.fullmatch(value):
        raise ValueError(f"invalid {label}")
    try:
        datetime.strptime(value, "%Y-%m-%dT%H:%M:%S.%fZ")
    except ValueError as exc:
        raise ValueError(f"invalid {label}") from exc
    return value


def validate_route_failure_diagnostic(
    value: object,
    *,
    expected_startup_error_code: str | None = None,
    expected_classified_error_code: str | None = None,
) -> None:
    """Fail closed unless ``value`` has the exact content-free schema."""

    root_keys = {
        "schema_version",
        "content_free",
        "startup_error_code",
        "classified_error_code",
        "observed_at",
        "routes",
    }
    if not isinstance(value, Mapping) or set(value) != root_keys:
        raise ValueError("invalid route diagnostic root")
    if value["schema_version"] != ROUTE_FAILURE_DIAGNOSTIC_SCHEMA:
        raise ValueError("invalid route diagnostic schema")
    if value["content_free"] is not True:
        raise ValueError("route diagnostic is not content-free")

    startup_error_code = value["startup_error_code"]
    if startup_error_code not in ROUTE_STARTUP_ERROR_CODES:
        raise ValueError("invalid route diagnostic startup error code")
    if (
        expected_startup_error_code is not None
        and startup_error_code != expected_startup_error_code
    ):
        raise ValueError("route diagnostic startup error mismatch")
    classified_error_code = value["classified_error_code"]
    if not isinstance(classified_error_code, str) or not _SAFE_ERROR_CODE.fullmatch(
        classified_error_code
    ):
        raise ValueError("invalid classified route error code")
    if (
        expected_classified_error_code is not None
        and classified_error_code != expected_classified_error_code
    ):
        raise ValueError("classified route error mismatch")

    observed_at = _require_timestamp(value["observed_at"], "observed_at")
    raw_routes = value["routes"]
    if not isinstance(raw_routes, list) or not 1 <= len(raw_routes) <= 64:
        raise ValueError("invalid route diagnostic routes")
    route_keys = {
        "route_alias",
        "exit_code",
        "stderr_class",
        "stderr_bytes",
        "observed_at",
        "stderr_modified_at",
    }
    aliases: set[str] = set()
    routes: list[Mapping[str, Any]] = []
    for route in raw_routes:
        if not isinstance(route, Mapping) or set(route) != route_keys:
            raise ValueError("invalid route diagnostic entry")
        alias = route["route_alias"]
        if (
            not isinstance(alias, str)
            or not _SAFE_ROUTE_ALIAS.fullmatch(alias)
            or alias in aliases
        ):
            raise ValueError("invalid route diagnostic alias")
        aliases.add(alias)
        exit_code = route["exit_code"]
        if exit_code is not None and (
            isinstance(exit_code, bool) or not isinstance(exit_code, int)
        ):
            raise ValueError("invalid route diagnostic exit code")
        stderr_class = route["stderr_class"]
        if stderr_class not in _STDERR_CLASSES:
            raise ValueError("invalid route stderr classification")
        stderr_bytes = route["stderr_bytes"]
        if (
            isinstance(stderr_bytes, bool)
            or not isinstance(stderr_bytes, int)
            or stderr_bytes < 0
        ):
            raise ValueError("invalid route stderr size")
        if route["observed_at"] != observed_at:
            raise ValueError("route diagnostic observation time mismatch")
        stderr_modified_at = route["stderr_modified_at"]
        if stderr_modified_at is not None:
            _require_timestamp(stderr_modified_at, "stderr_modified_at")
        if stderr_class == "stderr_missing" and (
            stderr_bytes != 0 or stderr_modified_at is not None
        ):
            raise ValueError("invalid missing stderr metadata")
        if stderr_class == "stderr_empty" and (
            stderr_bytes != 0 or stderr_modified_at is None
        ):
            raise ValueError("invalid empty stderr metadata")
        if stderr_class not in {
            "stderr_empty",
            "stderr_missing",
            "stderr_unreadable",
        } and (stderr_bytes == 0 or stderr_modified_at is None):
            raise ValueError("invalid classified stderr metadata")
        routes.append(route)

    expected_code = _classified_error_code(str(startup_error_code), routes)
    if classified_error_code != expected_code:
        raise ValueError("route diagnostic classification mismatch")


def build_route_failure_diagnostic(
    *,
    startup_error_code: str,
    sources: Iterable[RouteDiagnosticSource],
    now: datetime | None = None,
) -> dict[str, Any]:
    """Build a finite, content-free route failure summary.

    No stderr text, file path, URL, port, hash, environment value, or command
    argument is copied into the returned mapping.
    """

    if startup_error_code not in ROUTE_STARTUP_ERROR_CODES:
        raise ValueError("unsafe startup error code")
    observed_at = _timestamp(now or datetime.now(timezone.utc))
    routes = [
        _route_entry(source, observed_at=observed_at)
        for source in sources
    ]
    classified_error_code = _classified_error_code(startup_error_code, routes)
    diagnostic = {
        "schema_version": ROUTE_FAILURE_DIAGNOSTIC_SCHEMA,
        "content_free": True,
        "startup_error_code": startup_error_code,
        "classified_error_code": classified_error_code,
        "observed_at": observed_at,
        "routes": routes,
    }
    validate_route_failure_diagnostic(diagnostic)
    return diagnostic


def write_route_failure_diagnostic(
    path: Path,
    *,
    startup_error_code: str,
    sources: Iterable[RouteDiagnosticSource],
    now: datetime | None = None,
) -> dict[str, Any]:
    """Atomically persist a content-free summary and return its mapping."""

    diagnostic = build_route_failure_diagnostic(
        startup_error_code=startup_error_code,
        sources=sources,
        now=now,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(diagnostic, ensure_ascii=True, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    temporary.replace(path)
    return diagnostic


__all__ = [
    "ROUTE_FAILURE_DIAGNOSTIC_NAME",
    "ROUTE_FAILURE_DIAGNOSTIC_SCHEMA",
    "ROUTE_STARTUP_ERROR_CODES",
    "RouteDiagnosticSource",
    "build_route_failure_diagnostic",
    "validate_route_failure_diagnostic",
    "write_route_failure_diagnostic",
]

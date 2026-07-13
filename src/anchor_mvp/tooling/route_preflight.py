from __future__ import annotations

from contextlib import AbstractContextManager, contextmanager, nullcontext
from dataclasses import dataclass, field
import ipaddress
import json
import os
import re
import shlex
import socket
import subprocess
import sys
import tempfile
from typing import Callable, Iterator, Literal, Sequence, TextIO, cast


KIMI_API_HOST = "api.kimi.com"
SYNTHETIC_TUN_NETWORK = ipaddress.IPv4Network("198.18.0.0/15")
RouteMode = Literal["prompt", "current", "direct", "abort"]
ResolvedRouteMode = Literal["current", "direct", "abort"]

_DISTRO_RE = re.compile(r"^[A-Za-z0-9._-]+$")
_DEVICE_RE = re.compile(r"^[A-Za-z0-9_.:@-]+$")
_SAFE_ROUTE_TOKEN_RE = re.compile(r"^[A-Za-z0-9_.:/@-]+$")
_VIRTUAL_DEVICE_PREFIXES = (
    "br-",
    "clash",
    "docker",
    "dummy",
    "mihomo",
    "podman",
    "ppp",
    "tap",
    "tailscale",
    "tun",
    "veth",
    "virbr",
    "vpn",
    "warp",
    "wg",
    "zt",
)
_SECRET_ENV_NAMES = (
    "ANCHOR_TEACHER_API_KEY",
    "ANTHROPIC_API_KEY",
    "ARK_CODING_API_KEY",
    "KIMI_API_KEY",
    "KIMI_CODE_API_KEY",
    "OPENAI_API_KEY",
    "TEACHER_API_KEY",
)


class RoutePreflightError(RuntimeError):
    """Fail-closed route inspection or temporary-route failure."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


@dataclass(frozen=True)
class RoutePath:
    destination: ipaddress.IPv4Address
    gateway: ipaddress.IPv4Address | None
    device: str
    source: ipaddress.IPv4Address | None
    virtual_reasons: tuple[str, ...] = ()

    @property
    def is_virtual(self) -> bool:
        return bool(self.virtual_reasons)


@dataclass(frozen=True)
class DefaultRoute:
    gateway: ipaddress.IPv4Address | None
    device: str
    source: ipaddress.IPv4Address | None
    metric: int
    virtual_reasons: tuple[str, ...] = ()

    @property
    def is_virtual(self) -> bool:
        return bool(self.virtual_reasons)


@dataclass(frozen=True)
class KimiRouteAudit:
    host: str
    distro: str
    ipv4_addresses: tuple[ipaddress.IPv4Address, ...]
    current_routes: tuple[RoutePath, ...]
    default_routes: tuple[DefaultRoute, ...]
    physical_default: DefaultRoute | None


CommandRunner = Callable[[Sequence[str], float], subprocess.CompletedProcess[str]]
Resolver = Callable[[str], tuple[ipaddress.IPv4Address, ...]]


def _sanitized_environment() -> dict[str, str]:
    # Windows environment names are case-insensitive, but ``os.environ`` keeps
    # the spelling supplied by the parent process.  WSL needs both SystemRoot
    # and ComSpec; dropping an upper-case ``SYSTEMROOT`` makes wsl.exe fail with
    # an RPC error before the route command starts.
    allowed = {
        "comspec",
        "localappdata",
        "path",
        "pathext",
        "systemroot",
        "temp",
        "tmp",
        "userprofile",
        "windir",
    }
    blocked = {name.casefold() for name in _SECRET_ENV_NAMES}
    environment = {
        name: value
        for name, value in os.environ.items()
        if name.casefold() in allowed and name.casefold() not in blocked
    }
    return environment


def _default_command_runner(
    command: Sequence[str], timeout_seconds: float
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(command),
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        check=False,
        timeout=timeout_seconds,
        env=_sanitized_environment(),
    )


@contextmanager
def _exclusive_route_lock(distro: str) -> Iterator[None]:
    path = os.path.join(tempfile.gettempdir(), f"anchor-kimi-route-{distro}.lock")
    descriptor = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
    with os.fdopen(descriptor, "r+b") as handle:
        if os.fstat(handle.fileno()).st_size == 0:
            handle.write(b"\0")
            handle.flush()
        handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(  # type: ignore[attr-defined]
                    handle.fileno(),
                    fcntl.LOCK_EX | fcntl.LOCK_NB,  # type: ignore[attr-defined]
                )
        except OSError as error:
            raise RoutePreflightError("route_lock_busy") from error
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(  # type: ignore[attr-defined]
                    handle.fileno(),
                    fcntl.LOCK_UN,  # type: ignore[attr-defined]
                )


def resolve_kimi_ipv4(host: str = KIMI_API_HOST) -> tuple[ipaddress.IPv4Address, ...]:
    try:
        values = socket.getaddrinfo(
            host,
            443,
            family=socket.AF_INET,
            type=socket.SOCK_STREAM,
        )
    except socket.gaierror as error:
        raise RoutePreflightError("dns_ipv4_resolution_failed") from error
    addresses = {
        ipaddress.IPv4Address(value[4][0])
        for value in values
        if value[0] == socket.AF_INET
    }
    if not addresses:
        raise RoutePreflightError("dns_ipv4_resolution_empty")
    return tuple(sorted(addresses, key=int))


def _ipv4(value: object, *, code: str) -> ipaddress.IPv4Address | None:
    if value in (None, ""):
        return None
    try:
        return ipaddress.IPv4Address(str(value))
    except ipaddress.AddressValueError as error:
        raise RoutePreflightError(code) from error


def _device(value: object) -> str:
    text = str(value or "")
    if not text or not _DEVICE_RE.fullmatch(text):
        raise RoutePreflightError("route_device_invalid")
    return text


def _is_obvious_virtual_device(device: str) -> bool:
    normalized = device.casefold()
    return normalized in {"lo", "loopback", "loopback0"} or normalized.startswith(
        _VIRTUAL_DEVICE_PREFIXES
    )


def _virtual_reasons(
    *,
    destination: ipaddress.IPv4Address | None,
    gateway: ipaddress.IPv4Address | None,
    source: ipaddress.IPv4Address | None,
    device: str,
) -> tuple[str, ...]:
    reasons: list[str] = []
    if destination is not None and destination in SYNTHETIC_TUN_NETWORK:
        reasons.append("destination_in_198.18.0.0/15")
    if gateway is not None and gateway in SYNTHETIC_TUN_NETWORK:
        reasons.append("gateway_in_198.18.0.0/15")
    if source is not None and source in SYNTHETIC_TUN_NETWORK:
        reasons.append("source_in_198.18.0.0/15")
    if gateway is not None and (gateway.is_loopback or gateway.is_link_local):
        reasons.append("gateway_is_loopback_or_link_local")
    if _is_obvious_virtual_device(device):
        reasons.append("virtual_device")
    return tuple(reasons)


def _json_rows(stdout: str, *, code: str) -> list[dict[str, object]]:
    try:
        value = json.loads(stdout)
    except json.JSONDecodeError as error:
        raise RoutePreflightError(code) from error
    if not isinstance(value, list) or any(not isinstance(item, dict) for item in value):
        raise RoutePreflightError(code)
    return value


class WslKimiRoutePreflight:
    """Inspect and temporarily override one provider's IPv4 routes in one WSL distro."""

    def __init__(
        self,
        distro: str,
        *,
        host: str = KIMI_API_HOST,
        resolver: Resolver = resolve_kimi_ipv4,
        command_runner: CommandRunner = _default_command_runner,
        timeout_seconds: float = 10.0,
    ) -> None:
        if not _DISTRO_RE.fullmatch(distro):
            raise RoutePreflightError("wsl_distro_invalid")
        if timeout_seconds <= 0:
            raise RoutePreflightError("route_timeout_invalid")
        self.distro = distro
        self.host = host
        self.resolver = resolver
        self.command_runner = command_runner
        self.timeout_seconds = timeout_seconds

    def _ip(
        self,
        *arguments: str,
        root: bool = False,
        check: bool = True,
    ) -> subprocess.CompletedProcess[str]:
        command = ["wsl.exe", "-d", self.distro]
        if root:
            command.extend(("-u", "root"))
        command.extend(("--", "ip", *arguments))
        try:
            result = self.command_runner(tuple(command), self.timeout_seconds)
        except subprocess.TimeoutExpired as error:
            raise RoutePreflightError("wsl_route_command_timeout") from error
        except OSError as error:
            raise RoutePreflightError("wsl_route_command_unavailable") from error
        if check and result.returncode != 0:
            raise RoutePreflightError("wsl_route_command_failed")
        return result

    def _route_get(self, address: ipaddress.IPv4Address) -> RoutePath:
        result = self._ip("-j", "-4", "route", "get", str(address))
        rows = _json_rows(result.stdout, code="route_get_json_invalid")
        if len(rows) != 1:
            raise RoutePreflightError("route_get_result_invalid")
        row = rows[0]
        destination = _ipv4(row.get("dst"), code="route_destination_invalid")
        if destination != address:
            raise RoutePreflightError("route_destination_mismatch")
        gateway = _ipv4(row.get("gateway"), code="route_gateway_invalid")
        source = _ipv4(row.get("prefsrc", row.get("src")), code="route_source_invalid")
        device = _device(row.get("dev"))
        return RoutePath(
            destination=address,
            gateway=gateway,
            device=device,
            source=source,
            virtual_reasons=_virtual_reasons(
                destination=address,
                gateway=gateway,
                source=source,
                device=device,
            ),
        )

    def _default_routes(self) -> tuple[DefaultRoute, ...]:
        result = self._ip("-j", "-4", "route", "show", "default")
        rows = _json_rows(result.stdout, code="default_route_json_invalid")
        routes: list[DefaultRoute] = []
        for row in rows:
            if row.get("dst") != "default":
                raise RoutePreflightError("default_route_destination_invalid")
            gateway = _ipv4(row.get("gateway"), code="default_route_gateway_invalid")
            source = _ipv4(
                row.get("prefsrc", row.get("src")),
                code="default_route_source_invalid",
            )
            device = _device(row.get("dev"))
            raw_metric = row.get("metric", 0)
            if isinstance(raw_metric, bool) or not isinstance(raw_metric, (int, str)):
                raise RoutePreflightError("default_route_metric_invalid")
            try:
                metric = int(raw_metric)
            except (TypeError, ValueError) as error:
                raise RoutePreflightError("default_route_metric_invalid") from error
            routes.append(
                DefaultRoute(
                    gateway=gateway,
                    device=device,
                    source=source,
                    metric=metric,
                    virtual_reasons=_virtual_reasons(
                        destination=None,
                        gateway=gateway,
                        source=source,
                        device=device,
                    ),
                )
            )
        return tuple(
            sorted(
                routes,
                key=lambda route: (
                    route.metric,
                    route.device,
                    str(route.gateway or ""),
                ),
            )
        )

    def inspect(self) -> KimiRouteAudit:
        addresses = tuple(sorted(set(self.resolver(self.host)), key=int))
        if not addresses:
            raise RoutePreflightError("dns_ipv4_resolution_empty")
        current = tuple(self._route_get(address) for address in addresses)
        defaults = self._default_routes()
        candidates = tuple(
            route
            for route in defaults
            if not route.is_virtual and route.gateway is not None
        )
        best_metric = candidates[0].metric if candidates else None
        best = tuple(route for route in candidates if route.metric == best_metric)
        physical = best[0] if len(best) == 1 else None
        return KimiRouteAudit(
            host=self.host,
            distro=self.distro,
            ipv4_addresses=addresses,
            current_routes=current,
            default_routes=defaults,
            physical_default=physical,
        )

    def _snapshot_exact_route(
        self, address: ipaddress.IPv4Address
    ) -> tuple[str, ...] | None:
        result = self._ip("-4", "-o", "route", "show", "exact", f"{address}/32")
        lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
        if not lines:
            return None
        if len(lines) != 1:
            raise RoutePreflightError("exact_route_snapshot_ambiguous")
        try:
            tokens = shlex.split(lines[0], posix=True)
        except ValueError as error:
            raise RoutePreflightError("exact_route_snapshot_invalid") from error
        if not tokens or any(
            not _SAFE_ROUTE_TOKEN_RE.fullmatch(item) for item in tokens
        ):
            raise RoutePreflightError("exact_route_snapshot_invalid")
        try:
            destination = ipaddress.IPv4Network(tokens[0], strict=False)
        except (ipaddress.AddressValueError, ipaddress.NetmaskValueError) as error:
            raise RoutePreflightError("exact_route_snapshot_invalid") from error
        if destination.prefixlen != 32 or destination.network_address != address:
            raise RoutePreflightError("exact_route_snapshot_mismatch")
        tokens[0] = f"{address}/32"
        return tuple(tokens)

    def _restore_routes(
        self,
        modified: Sequence[ipaddress.IPv4Address],
        snapshots: dict[ipaddress.IPv4Address, tuple[str, ...] | None],
        installed: dict[ipaddress.IPv4Address, tuple[str, ...]],
    ) -> None:
        failed = False
        conflict = False
        for address in reversed(modified):
            try:
                snapshot = snapshots[address]
                current = self._snapshot_exact_route(address)
                if current == snapshot:
                    continue
                if current != installed[address]:
                    conflict = True
                    continue
                if snapshot is None:
                    self._ip("-4", "route", "del", f"{address}/32", root=True)
                else:
                    self._ip("-4", "route", "replace", *snapshot, root=True)
            except RoutePreflightError:
                failed = True
        if conflict:
            raise RoutePreflightError("route_restore_conflict")
        if failed:
            raise RoutePreflightError("route_restore_failed")

    @contextmanager
    def temporary_direct_routes(self, audit: KimiRouteAudit) -> Iterator[None]:
        with _exclusive_route_lock(self.distro):
            with self._temporary_direct_routes_unlocked(audit):
                yield

    @contextmanager
    def _temporary_direct_routes_unlocked(
        self, audit: KimiRouteAudit
    ) -> Iterator[None]:
        if audit.distro != self.distro or audit.host != self.host:
            raise RoutePreflightError("route_audit_identity_mismatch")
        if any(address in SYNTHETIC_TUN_NETWORK for address in audit.ipv4_addresses):
            raise RoutePreflightError("direct_route_rejects_synthetic_destination")
        if any(not address.is_global for address in audit.ipv4_addresses):
            raise RoutePreflightError("direct_route_rejects_non_public_destination")
        physical = audit.physical_default
        if physical is None or physical.gateway is None:
            raise RoutePreflightError("physical_default_route_missing")

        current_defaults = self._default_routes()
        if not any(
            route.gateway == physical.gateway
            and route.device == physical.device
            and not route.is_virtual
            for route in current_defaults
        ):
            raise RoutePreflightError("physical_default_route_changed")

        snapshots = {
            address: self._snapshot_exact_route(address)
            for address in audit.ipv4_addresses
        }
        modified: list[ipaddress.IPv4Address] = []
        installed = {
            address: _direct_route_snapshot(address, physical)
            for address in audit.ipv4_addresses
        }
        try:
            for address in audit.ipv4_addresses:
                modified.append(address)
                self._ip(
                    "-4",
                    "route",
                    "replace",
                    f"{address}/32",
                    "via",
                    str(physical.gateway),
                    "dev",
                    physical.device,
                    root=True,
                )
                exact = self._snapshot_exact_route(address)
                if exact is None or not _matches_direct_snapshot(
                    exact, address, physical
                ):
                    raise RoutePreflightError("direct_route_verification_failed")
                installed[address] = exact
                verified = self._route_get(address)
                if (
                    verified.gateway != physical.gateway
                    or verified.device != physical.device
                    or verified.is_virtual
                ):
                    raise RoutePreflightError("direct_route_verification_failed")
            yield
        finally:
            if modified:
                self._restore_routes(modified, snapshots, installed)


def _direct_route_snapshot(
    address: ipaddress.IPv4Address,
    physical: DefaultRoute,
) -> tuple[str, ...]:
    if physical.gateway is None:
        raise RoutePreflightError("physical_default_route_missing")
    return (
        f"{address}/32",
        "via",
        str(physical.gateway),
        "dev",
        physical.device,
    )


def _matches_direct_snapshot(
    snapshot: tuple[str, ...] | None,
    address: ipaddress.IPv4Address,
    physical: DefaultRoute,
) -> bool:
    if snapshot is None or physical.gateway is None or snapshot[0] != f"{address}/32":
        return False
    try:
        gateway = snapshot[snapshot.index("via") + 1]
        device = snapshot[snapshot.index("dev") + 1]
    except (ValueError, IndexError):
        return False
    return gateway == str(physical.gateway) and device == physical.device


def render_route_audit(audit: KimiRouteAudit) -> str:
    addresses = ",".join(str(value) for value in audit.ipv4_addresses)
    lines = [
        "provider_route_preflight=no_api_request",
        f"host={audit.host} distro={audit.distro} ipv4={addresses}",
    ]
    for route in audit.current_routes:
        reasons = ",".join(route.virtual_reasons) or "none"
        lines.append(
            "current_route "
            f"ip={route.destination} gateway={route.gateway or 'none'} "
            f"dev={route.device} src={route.source or 'none'} "
            f"virtual={'yes' if route.is_virtual else 'no'} reasons={reasons}"
        )
    physical = audit.physical_default
    if physical is None:
        lines.append("physical_default=none")
    else:
        lines.append(
            "physical_default "
            f"gateway={physical.gateway} dev={physical.device} "
            f"src={physical.source or 'none'} metric={physical.metric}"
        )
    return "\n".join(lines)


def choose_route_mode(
    requested: RouteMode,
    audit: KimiRouteAudit,
    *,
    stdin: TextIO = sys.stdin,
    stdout: TextIO = sys.stdout,
) -> ResolvedRouteMode:
    if requested not in {"prompt", "current", "direct", "abort"}:
        raise RoutePreflightError("route_mode_invalid")
    if requested != "prompt":
        return requested
    if not stdin.isatty():
        raise RoutePreflightError("route_mode_required_for_non_tty")
    direct = audit.physical_default
    direct_text = (
        f"temporary /32 via {direct.gateway} dev {direct.device}"
        if direct is not None
        else "unavailable: no physical default candidate"
    )
    stdout.write(
        "Select provider route mode:\n"
        "  [1/current] keep current routing\n"
        f"  [2/direct] {direct_text}\n"
        "  [3/abort] stop before any API request\n"
        "route> "
    )
    stdout.flush()
    choice = stdin.readline().strip().casefold()
    selected = {
        "1": "current",
        "current": "current",
        "2": "direct",
        "direct": "direct",
        "3": "abort",
        "abort": "abort",
    }.get(choice)
    if selected is None:
        raise RoutePreflightError("route_mode_selection_invalid")
    return cast(ResolvedRouteMode, selected)


@dataclass(frozen=True)
class KimiRoutePlan:
    audit: KimiRouteAudit
    mode: ResolvedRouteMode
    _preflight: WslKimiRoutePreflight = field(repr=False, compare=False)

    def activate(self) -> AbstractContextManager[None]:
        if self.mode == "abort":
            raise RoutePreflightError("route_mode_abort")
        if self.mode == "direct":
            return self._preflight.temporary_direct_routes(self.audit)
        if self.mode == "current":
            return nullcontext()
        raise RoutePreflightError("route_mode_invalid")


def prepare_kimi_route_plan(
    *,
    distro: str | None,
    requested_mode: RouteMode,
    host: str = KIMI_API_HOST,
    stdin: TextIO = sys.stdin,
    stdout: TextIO = sys.stdout,
    preflight: WslKimiRoutePreflight | None = None,
) -> KimiRoutePlan:
    if requested_mode not in {"prompt", "current", "direct", "abort"}:
        raise RoutePreflightError("route_mode_invalid")
    if requested_mode == "abort":
        raise RoutePreflightError("route_mode_abort")
    if requested_mode == "prompt" and not stdin.isatty():
        raise RoutePreflightError("route_mode_required_for_non_tty")
    if distro is None:
        raise RoutePreflightError("wsl_distro_required_for_route_preflight")
    inspector = preflight or WslKimiRoutePreflight(distro, host=host)
    if inspector.host != host:
        raise RoutePreflightError("route_host_mismatch")
    audit = inspector.inspect()
    stdout.write(render_route_audit(audit) + "\n")
    stdout.flush()
    mode = choose_route_mode(
        requested_mode,
        audit,
        stdin=stdin,
        stdout=stdout,
    )
    if mode == "abort":
        raise RoutePreflightError("route_mode_abort")
    if mode == "direct":
        if any(address in SYNTHETIC_TUN_NETWORK for address in audit.ipv4_addresses):
            raise RoutePreflightError("direct_route_rejects_synthetic_destination")
        if any(not address.is_global for address in audit.ipv4_addresses):
            raise RoutePreflightError("direct_route_rejects_non_public_destination")
        if audit.physical_default is None:
            raise RoutePreflightError("physical_default_route_missing_or_ambiguous")
    return KimiRoutePlan(audit=audit, mode=mode, _preflight=inspector)


__all__ = [
    "DefaultRoute",
    "KIMI_API_HOST",
    "KimiRouteAudit",
    "KimiRoutePlan",
    "ResolvedRouteMode",
    "RouteMode",
    "RoutePath",
    "RoutePreflightError",
    "SYNTHETIC_TUN_NETWORK",
    "WslKimiRoutePreflight",
    "choose_route_mode",
    "prepare_kimi_route_plan",
    "render_route_audit",
    "resolve_kimi_ipv4",
]

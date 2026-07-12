from __future__ import annotations

from io import StringIO
import importlib.util
import ipaddress
import json
from pathlib import Path
import subprocess
import sys

import pytest

from anchor_mvp.tooling.route_preflight import (
    KIMI_API_HOST,
    KimiRoutePlan,
    RoutePreflightError,
    WslKimiRoutePreflight,
    _default_command_runner,
    _exclusive_route_lock,
    choose_route_mode,
    prepare_kimi_route_plan,
    resolve_kimi_ipv4,
)


ROOT = Path(__file__).resolve().parents[1]
KIMI_IP = ipaddress.IPv4Address("103.143.17.156")
TEST_DISTRO = "AnchorRouteTest"


class TTYInput(StringIO):
    def isatty(self) -> bool:
        return True


def completed(command, stdout="", returncode=0):
    return subprocess.CompletedProcess(command, returncode, stdout, "")


def test_resolver_keeps_ipv4_deduplicated_and_sorted(monkeypatch):
    monkeypatch.setattr(
        "anchor_mvp.tooling.route_preflight.socket.getaddrinfo",
        lambda *_args, **_kwargs: [
            (2, 1, 6, "", ("103.143.17.200", 443)),
            (2, 1, 6, "", ("103.143.17.156", 443)),
            (2, 1, 6, "", ("103.143.17.200", 443)),
        ],
    )

    assert resolve_kimi_ipv4() == (
        ipaddress.IPv4Address("103.143.17.156"),
        ipaddress.IPv4Address("103.143.17.200"),
    )


def test_inspect_records_each_route_and_selects_non_tun_default():
    commands = []

    def runner(command, _timeout):
        commands.append(tuple(command))
        arguments = tuple(command[command.index("ip") + 1 :])
        if arguments == ("-j", "-4", "route", "get", str(KIMI_IP)):
            return completed(
                command,
                json.dumps(
                    [
                        {
                            "dst": str(KIMI_IP),
                            "gateway": "198.18.0.2",
                            "dev": "eth0",
                            "prefsrc": "198.18.0.1",
                        }
                    ]
                ),
            )
        if arguments == ("-j", "-4", "route", "show", "default"):
            return completed(
                command,
                json.dumps(
                    [
                        {
                            "dst": "default",
                            "gateway": "198.18.0.2",
                            "dev": "eth0",
                            "metric": 1,
                        },
                        {
                            "dst": "default",
                            "gateway": "192.168.3.1",
                            "dev": "eth4",
                            "metric": 30,
                        },
                    ]
                ),
            )
        raise AssertionError(arguments)

    audit = WslKimiRoutePreflight(
        "Ubuntu-22.04",
        resolver=lambda _host: (KIMI_IP,),
        command_runner=runner,
    ).inspect()

    assert audit.host == KIMI_API_HOST
    assert audit.current_routes[0].gateway == ipaddress.IPv4Address("198.18.0.2")
    assert audit.current_routes[0].device == "eth0"
    assert audit.current_routes[0].source == ipaddress.IPv4Address("198.18.0.1")
    assert set(audit.current_routes[0].virtual_reasons) == {
        "gateway_in_198.18.0.0/15",
        "source_in_198.18.0.0/15",
    }
    assert audit.physical_default is not None
    assert audit.physical_default.gateway == ipaddress.IPv4Address("192.168.3.1")
    assert audit.physical_default.device == "eth4"
    assert len(commands) == 2


@pytest.mark.parametrize(
    "device", ["tun0", "tap1", "wg0", "tailscale0", "clash0", "mihomo", "zt0", "lo", "docker0", "podman0", "br-test", "veth42"]
)
def test_obvious_virtual_devices_are_not_physical_defaults(device):
    def runner(command, _timeout):
        arguments = tuple(command[command.index("ip") + 1 :])
        if "get" in arguments:
            return completed(
                command,
                json.dumps(
                    [{"dst": str(KIMI_IP), "gateway": "10.0.0.1", "dev": device}]
                ),
            )
        return completed(
            command,
            json.dumps(
                [{"dst": "default", "gateway": "10.0.0.1", "dev": device, "metric": 1}]
            ),
        )

    audit = WslKimiRoutePreflight(
        "Ubuntu-22.04",
        resolver=lambda _host: (KIMI_IP,),
        command_runner=runner,
    ).inspect()

    assert audit.current_routes[0].is_virtual
    assert audit.physical_default is None


def test_non_tty_prompt_fails_before_dns_or_wsl():
    class NeverInspect:
        def inspect(self):
            raise AssertionError("route inspection must not run")

    with pytest.raises(RoutePreflightError) as raised:
        prepare_kimi_route_plan(
            distro="Ubuntu-22.04",
            requested_mode="prompt",
            stdin=StringIO(),
            stdout=StringIO(),
            preflight=NeverInspect(),  # type: ignore[arg-type]
        )

    assert raised.value.code == "route_mode_required_for_non_tty"


def test_explicit_abort_stops_before_dns_or_wsl():
    class NeverInspect:
        def inspect(self):
            raise AssertionError("route inspection must not run")

    with pytest.raises(RoutePreflightError) as raised:
        prepare_kimi_route_plan(
            distro="Ubuntu-22.04",
            requested_mode="abort",
            stdin=StringIO(),
            stdout=StringIO(),
            preflight=NeverInspect(),  # type: ignore[arg-type]
        )

    assert raised.value.code == "route_mode_abort"


def _stateful_preflight(*, exact_route: str):
    commands: list[tuple[str, ...]] = []
    state = {"direct": False}

    def runner(command, _timeout):
        command = tuple(command)
        commands.append(command)
        arguments = tuple(command[command.index("ip") + 1 :])
        if arguments == ("-j", "-4", "route", "show", "default"):
            return completed(
                command,
                json.dumps(
                    [
                        {
                            "dst": "default",
                            "gateway": "198.18.0.2",
                            "dev": "eth0",
                            "metric": 1,
                        },
                        {
                            "dst": "default",
                            "gateway": "192.168.3.1",
                            "dev": "eth4",
                            "metric": 30,
                        },
                    ]
                ),
            )
        if arguments == ("-4", "-o", "route", "show", "exact", f"{KIMI_IP}/32"):
            value = (
                f"{KIMI_IP} via 192.168.3.1 dev eth4\n"
                if state["direct"]
                else exact_route
            )
            return completed(command, value)
        if arguments[:4] == ("-4", "route", "replace", f"{KIMI_IP}/32"):
            state["direct"] = arguments[4:8] == (
                "via",
                "192.168.3.1",
                "dev",
                "eth4",
            )
            return completed(command)
        if arguments == ("-4", "route", "del", f"{KIMI_IP}/32"):
            state["direct"] = False
            return completed(command)
        if arguments == ("-j", "-4", "route", "get", str(KIMI_IP)):
            if state["direct"]:
                route = {
                    "dst": str(KIMI_IP),
                    "gateway": "192.168.3.1",
                    "dev": "eth4",
                    "prefsrc": "192.168.3.68",
                }
            else:
                route = {
                    "dst": str(KIMI_IP),
                    "gateway": "198.18.0.2",
                    "dev": "eth0",
                    "prefsrc": "198.18.0.1",
                }
            return completed(command, json.dumps([route]))
        raise AssertionError(arguments)

    inspector = WslKimiRoutePreflight(
        TEST_DISTRO,
        resolver=lambda _host: (KIMI_IP,),
        command_runner=runner,
    )
    return inspector, commands, state


def test_direct_replaces_only_host_route_then_deletes_when_no_snapshot():
    inspector, commands, state = _stateful_preflight(exact_route="")
    audit = inspector.inspect()

    with inspector.temporary_direct_routes(audit):
        assert state["direct"] is True

    assert state["direct"] is False
    mutations = [command[command.index("ip") + 1 :] for command in commands if "root" in command]
    assert mutations == [
        (
            "-4",
            "route",
            "replace",
            f"{KIMI_IP}/32",
            "via",
            "192.168.3.1",
            "dev",
            "eth4",
        ),
        ("-4", "route", "del", f"{KIMI_IP}/32"),
    ]
    assert all("default" not in command for command in mutations)


def test_direct_restores_preexisting_exact_route_on_body_exception():
    original = f"{KIMI_IP} via 198.18.0.2 dev eth0 metric 7\n"
    inspector, commands, _state = _stateful_preflight(exact_route=original)
    audit = inspector.inspect()

    with pytest.raises(KeyboardInterrupt):
        with inspector.temporary_direct_routes(audit):
            raise KeyboardInterrupt

    mutations = [command[command.index("ip") + 1 :] for command in commands if "root" in command]
    assert mutations[-1] == (
        "-4",
        "route",
        "replace",
        f"{KIMI_IP}/32",
        "via",
        "198.18.0.2",
        "dev",
        "eth0",
        "metric",
        "7",
    )


def test_direct_restores_if_replace_mutates_then_times_out():
    state = {"direct": False, "deleted": False}

    def runner(command, _timeout):
        arguments = tuple(command[command.index("ip") + 1 :])
        if arguments == ("-j", "-4", "route", "show", "default"):
            return completed(
                command,
                json.dumps(
                    [{"dst": "default", "gateway": "192.168.3.1", "dev": "eth4", "metric": 30}]
                ),
            )
        if arguments == ("-j", "-4", "route", "get", str(KIMI_IP)):
            route = {
                "dst": str(KIMI_IP),
                "gateway": "192.168.3.1" if state["direct"] else "198.18.0.2",
                "dev": "eth4" if state["direct"] else "eth0",
            }
            return completed(command, json.dumps([route]))
        if arguments == ("-4", "-o", "route", "show", "exact", f"{KIMI_IP}/32"):
            value = f"{KIMI_IP} via 192.168.3.1 dev eth4\n" if state["direct"] else ""
            return completed(command, value)
        if arguments[:4] == ("-4", "route", "replace", f"{KIMI_IP}/32"):
            state["direct"] = True
            raise subprocess.TimeoutExpired(command, 10)
        if arguments == ("-4", "route", "del", f"{KIMI_IP}/32"):
            state["direct"] = False
            state["deleted"] = True
            return completed(command)
        raise AssertionError(arguments)

    inspector = WslKimiRoutePreflight(
        TEST_DISTRO,
        resolver=lambda _host: (KIMI_IP,),
        command_runner=runner,
    )
    audit = inspector.inspect()

    with pytest.raises(RoutePreflightError) as raised:
        with inspector.temporary_direct_routes(audit):
            pass

    assert raised.value.code == "wsl_route_command_timeout"
    assert state == {"direct": False, "deleted": True}


def test_direct_does_not_overwrite_externally_changed_route_during_restore():
    state = {"route": "original"}

    def exact_route() -> str:
        if state["route"] == "original":
            return ""
        if state["route"] == "direct":
            return f"{KIMI_IP} via 192.168.3.1 dev eth4\n"
        return f"{KIMI_IP} via 192.168.3.1 dev eth4 metric 99\n"

    def runner(command, _timeout):
        arguments = tuple(command[command.index("ip") + 1 :])
        if arguments == ("-j", "-4", "route", "show", "default"):
            return completed(
                command,
                json.dumps(
                    [
                        {
                            "dst": "default",
                            "gateway": "192.168.3.1",
                            "dev": "eth4",
                            "metric": 30,
                        }
                    ]
                ),
            )
        if arguments == ("-j", "-4", "route", "get", str(KIMI_IP)):
            gateway = "192.168.3.1" if state["route"] != "original" else "198.18.0.2"
            device = "eth4" if state["route"] != "original" else "eth0"
            return completed(
                command,
                json.dumps([{"dst": str(KIMI_IP), "gateway": gateway, "dev": device}]),
            )
        if arguments == ("-4", "-o", "route", "show", "exact", f"{KIMI_IP}/32"):
            return completed(command, exact_route())
        if arguments[:4] == ("-4", "route", "replace", f"{KIMI_IP}/32"):
            state["route"] = "direct"
            return completed(command)
        if arguments == ("-4", "route", "del", f"{KIMI_IP}/32"):
            raise AssertionError("conflicting external route must not be deleted")
        raise AssertionError(arguments)

    inspector = WslKimiRoutePreflight(
        TEST_DISTRO,
        resolver=lambda _host: (KIMI_IP,),
        command_runner=runner,
    )
    audit = inspector.inspect()

    with pytest.raises(RoutePreflightError) as raised:
        with inspector.temporary_direct_routes(audit):
            state["route"] = "external"

    assert raised.value.code == "route_restore_conflict"
    assert state["route"] == "external"


def test_current_mode_never_mutates_routes():
    inspector, commands, _state = _stateful_preflight(exact_route="")
    output = StringIO()
    plan = prepare_kimi_route_plan(
        distro="Ubuntu-22.04",
        requested_mode="current",
        stdin=StringIO(),
        stdout=output,
        preflight=inspector,
    )
    inspection_count = len(commands)

    with plan.activate():
        pass

    assert len(commands) == inspection_count
    assert "gateway=198.18.0.2" in output.getvalue()
    assert "KIMI_CODE_API_KEY" not in output.getvalue()


def test_direct_rejects_synthetic_dns_destination_without_mutation():
    fake_ip = ipaddress.IPv4Address("198.18.1.20")

    def runner(command, _timeout):
        arguments = tuple(command[command.index("ip") + 1 :])
        if "get" in arguments:
            return completed(
                command,
                json.dumps(
                    [{"dst": str(fake_ip), "gateway": "198.18.0.2", "dev": "eth0"}]
                ),
            )
        return completed(
            command,
            json.dumps(
                [
                    {
                        "dst": "default",
                        "gateway": "192.168.3.1",
                        "dev": "eth4",
                        "metric": 30,
                    }
                ]
            ),
        )

    inspector = WslKimiRoutePreflight(
        "Ubuntu-22.04",
        resolver=lambda _host: (fake_ip,),
        command_runner=runner,
    )

    with pytest.raises(RoutePreflightError) as raised:
        prepare_kimi_route_plan(
            distro="Ubuntu-22.04",
            requested_mode="direct",
            stdin=StringIO(),
            stdout=StringIO(),
            preflight=inspector,
        )

    assert raised.value.code == "direct_route_rejects_synthetic_destination"


def test_direct_rejects_private_dns_destination_without_mutation():
    private_ip = ipaddress.IPv4Address("10.20.30.40")

    def runner(command, _timeout):
        arguments = tuple(command[command.index("ip") + 1 :])
        if "get" in arguments:
            return completed(
                command,
                json.dumps(
                    [{"dst": str(private_ip), "gateway": "10.0.0.1", "dev": "eth0"}]
                ),
            )
        return completed(
            command,
            json.dumps(
                [
                    {
                        "dst": "default",
                        "gateway": "192.168.3.1",
                        "dev": "eth4",
                        "metric": 30,
                    }
                ]
            ),
        )

    inspector = WslKimiRoutePreflight(
        "Ubuntu-22.04",
        resolver=lambda _host: (private_ip,),
        command_runner=runner,
    )
    with pytest.raises(RoutePreflightError) as raised:
        prepare_kimi_route_plan(
            distro="Ubuntu-22.04",
            requested_mode="direct",
            stdin=StringIO(),
            stdout=StringIO(),
            preflight=inspector,
        )

    assert raised.value.code == "direct_route_rejects_non_public_destination"


def test_equal_metric_physical_defaults_are_ambiguous_for_direct():
    def runner(command, _timeout):
        arguments = tuple(command[command.index("ip") + 1 :])
        if "get" in arguments:
            return completed(
                command,
                json.dumps(
                    [{"dst": str(KIMI_IP), "gateway": "10.0.0.1", "dev": "eth0"}]
                ),
            )
        return completed(
            command,
            json.dumps(
                [
                    {"dst": "default", "gateway": "10.0.0.1", "dev": "eth0", "metric": 5},
                    {"dst": "default", "gateway": "10.1.0.1", "dev": "eth1", "metric": 5},
                ]
            ),
        )

    inspector = WslKimiRoutePreflight(
        "Ubuntu-22.04",
        resolver=lambda _host: (KIMI_IP,),
        command_runner=runner,
    )
    with pytest.raises(RoutePreflightError) as raised:
        prepare_kimi_route_plan(
            distro="Ubuntu-22.04",
            requested_mode="direct",
            stdin=StringIO(),
            stdout=StringIO(),
            preflight=inspector,
        )

    assert raised.value.code == "physical_default_route_missing_or_ambiguous"


def test_route_subprocess_environment_does_not_receive_api_key(monkeypatch):
    captured = {}
    monkeypatch.setenv("SYSTEMROOT", r"C:\Windows")
    monkeypatch.setenv("COMSPEC", r"C:\Windows\System32\cmd.exe")
    monkeypatch.setenv("KIMI_CODE_API_KEY", "never-forward-this")
    monkeypatch.setenv("HTTPS_PROXY", "http://secret-proxy.invalid")
    monkeypatch.setenv("OPENCODE_CONFIG_CONTENT", "secret-config")

    def fake_run(command, **kwargs):
        captured.update(kwargs)
        return completed(command)

    monkeypatch.setattr("anchor_mvp.tooling.route_preflight.subprocess.run", fake_run)
    _default_command_runner(("wsl.exe", "--", "ip", "route"), 1.0)

    assert "KIMI_CODE_API_KEY" not in captured["env"]
    assert "HTTPS_PROXY" not in captured["env"]
    assert "OPENCODE_CONFIG_CONTENT" not in captured["env"]
    assert "never-forward-this" not in repr(captured)
    assert captured["env"]["SYSTEMROOT"] == r"C:\Windows"
    assert captured["env"]["COMSPEC"] == r"C:\Windows\System32\cmd.exe"


def test_direct_route_lock_rejects_a_second_live_owner():
    with _exclusive_route_lock(TEST_DISTRO):
        with pytest.raises(RoutePreflightError) as raised:
            with _exclusive_route_lock(TEST_DISTRO):
                pass

    assert raised.value.code == "route_lock_busy"


def test_prompt_abort_and_unknown_programmatic_mode_fail_closed():
    inspector, commands, _state = _stateful_preflight(exact_route="")
    with pytest.raises(RoutePreflightError) as aborted:
        prepare_kimi_route_plan(
            distro=TEST_DISTRO,
            requested_mode="prompt",
            stdin=TTYInput("3\n"),
            stdout=StringIO(),
            preflight=inspector,
        )
    assert aborted.value.code == "route_mode_abort"
    assert all("root" not in command for command in commands)

    with pytest.raises(RoutePreflightError) as invalid:
        prepare_kimi_route_plan(
            distro=TEST_DISTRO,
            requested_mode="unexpected",  # type: ignore[arg-type]
            stdin=StringIO(),
            stdout=StringIO(),
            preflight=inspector,
        )
    assert invalid.value.code == "route_mode_invalid"


def test_unknown_mode_fails_closed_in_chooser_and_plan_activation():
    inspector, _commands, _state = _stateful_preflight(exact_route="")
    audit = inspector.inspect()

    with pytest.raises(RoutePreflightError) as chosen:
        choose_route_mode(
            "unexpected",  # type: ignore[arg-type]
            audit,
            stdin=StringIO(),
            stdout=StringIO(),
        )
    assert chosen.value.code == "route_mode_invalid"

    plan = KimiRoutePlan(
        audit=audit,
        mode="unexpected",  # type: ignore[arg-type]
        _preflight=inspector,
    )
    with pytest.raises(RoutePreflightError) as activated:
        plan.activate()
    assert activated.value.code == "route_mode_invalid"


def test_run_live_cli_defaults_to_prompt_and_accepts_explicit_mode(monkeypatch):
    path = ROOT / "scripts" / "tooling" / "run_live.py"
    spec = importlib.util.spec_from_file_location("anchor_run_live_route_test", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    monkeypatch.setattr(sys, "argv", [str(path)])
    assert module.parse_args().route_mode == "prompt"
    monkeypatch.setattr(sys, "argv", [str(path), "--route-mode", "direct"])
    assert module.parse_args().route_mode == "direct"


def test_run_live_wraps_batch_and_single_with_one_route_plan_each():
    source = (ROOT / "scripts" / "tooling" / "run_live.py").read_text(encoding="utf-8")

    assert source.count("route_plan = route_preflight(executor, args.route_mode)") == 2
    assert source.count("with route_plan.activate():") == 2

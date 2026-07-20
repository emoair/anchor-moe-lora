"""Run exactly one explicitly confirmed v3 task to create live probe evidence.

The normal full-bank coordinator never bypasses its execution attestation.
This narrow bootstrap command is the only exception: it requires all static
code, bank, route, receipt-key, and on-demand-image-supervisor prerequisites,
then runs one train task with concurrency one in an isolated output directory.
It may let the trusted WSL supervisor access the network to acquire/build the
official TestSpec image. Model and evaluator containers remain network-none
and pull-never. No work starts without both confirmation flags.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import importlib.util
import json
from pathlib import Path
import re
import subprocess
import sys
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[2]
COORDINATOR_PATH = ROOT / "scripts/tooling/run_swebench_ccswitch.py"
_CONTROL = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{7,127}$")


def _load_coordinator():
    spec = importlib.util.spec_from_file_location(
        "anchor_swebench_ccswitch_coordinator", COORDINATOR_PATH
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("coordinator_import_failed")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _static_probe_ready(report: Mapping[str, Any]) -> bool:
    contract = _mapping(report.get("execution_contract"))
    probe = _mapping(contract.get("current_probe"))
    bindings = _mapping(probe.get("bindings"))
    dataset = _mapping(bindings.get("dataset"))
    harness = _mapping(bindings.get("official_harness"))
    opencode = _mapping(bindings.get("opencode"))
    validator = _mapping(bindings.get("validator"))
    supervisor = _mapping(bindings.get("supervisor_private_state"))
    cache = _mapping(bindings.get("on_demand_image_cache"))
    return bool(
        report.get("component_ready") is True
        and report.get("bank_ready") is True
        and dataset.get("present_and_bound") is True
        and harness.get("clean_checkout") is True
        and harness.get("import_ok") is True
        and opencode.get("tool_contract_version")
        == "anchor.execution-tool-contract.v3"
        and isinstance(opencode.get("linux_binary_sha256"), str)
        and opencode.get("model_isolation_contract") is True
        and opencode.get("testbed_workdir_contract") is True
        and validator.get("code_bound") is True
        and validator.get("self_test") is True
        and validator.get("rejects_arbitrary_commands") is True
        and supervisor.get("receipt_key_metadata_valid") is True
        and cache.get("offline_integrity_probe") is True
        and cache.get("pull_during_model_or_eval") is False
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/data/swebench_five_stage.ccswitch.yaml",
    )
    parser.add_argument("--control-run-id", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--confirm-representative-live", action="store_true")
    parser.add_argument("--confirm-supervisor-network", action="store_true")
    args = parser.parse_args()
    if (
        not args.confirm_representative_live
        or not args.confirm_supervisor_network
        or not _CONTROL.fullmatch(args.control_run_id)
    ):
        print("representative_probe=blocked reason=explicit_confirmation_required")
        return 2
    coordinator = _load_coordinator()
    config_path = (ROOT / args.config).resolve()
    try:
        config_path.relative_to(ROOT)
        config = coordinator.CoordinatorConfig.load(config_path)
        report = coordinator.offline_preflight(config)
        if not _static_probe_ready(report):
            print("representative_probe=blocked reason=static_prerequisite_failed")
            return 2
        probe_output = ROOT / "artifacts/tooling/swebench-v3/representative-run"
        probe_runtime = replace(config.runtime, output_dir=probe_output)
        probe_config = replace(config, runtime=probe_runtime)
        result = coordinator.run_live(
            probe_config,
            control_run_id=args.control_run_id,
            resume=bool(args.resume),
            concurrency=1,
            max_tasks=1,
        )
        bindings = sorted(
            (probe_output / "system-private").glob(
                "*/representative-runtime-binding.json"
            )
        )
        if len(bindings) != 1:
            print("representative_probe=blocked reason=authenticated_runtime_binding_absent")
            return 3
        output = (
            ROOT
            / "artifacts/tooling/swebench-v3/representative-probe-attestation.json"
        )
        built = subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts/tooling/build_swebench_v3_probe_attestation.py"),
                "--lock",
                str(config.execution_contract.lock),
                "--runtime-binding",
                str(bindings[0]),
                "--output",
                str(output),
            ],
            cwd=ROOT,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            encoding="utf-8",
            timeout=120,
            check=False,
        )
        if built.returncode != 0:
            print("representative_probe=blocked reason=attestation_build_failed")
            return 3
        print(
            json.dumps(
                {
                    "representative_probe": "completed",
                    "task_count": result.get("task_count_requested"),
                    "counts": result.get("counts"),
                    "attestation": output.relative_to(ROOT).as_posix(),
                    "full_bank_live_started": False,
                },
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
            )
        )
        return 0
    except Exception:  # noqa: BLE001 - keep probe failure content-free
        print("representative_probe=blocked reason=local_probe_failure")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

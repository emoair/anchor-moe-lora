"""Command line entry point for safe dry runs and explicit adapter training."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping

from .config import ALLOWED_ADAPTERS, ALLOWED_RANKS, ConfigError, load_training_config, select_adapter
from .dependencies import dependency_report
from .manifest import build_manifest, sha256_file, write_json
from .preflight import build_preflight_report, verify_prior_smoke_gate
from .schema import DatasetValidationError, validate_jsonl


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Anchor-MoE-LoRA Gemma 4 12B QLoRA trainer")
    parser.add_argument(
        "stage",
        nargs="?",
        choices=("train", "preflight", "smoke-gate"),
        default="train",
        help="train (legacy default), read-only preflight, or one-step smoke gate",
    )
    parser.add_argument("--config", required=True, help="JSON-compatible YAML training config")
    parser.add_argument("--adapter", choices=ALLOWED_ADAPTERS)
    parser.add_argument("--rank", type=int, choices=ALLOWED_RANKS)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="validate only; this is the default")
    mode.add_argument("--execute", action="store_true", help="perform a real local training run")
    parser.add_argument(
        "--allow-model-download",
        action="store_true",
        help="permit Hugging Face downloads during --execute (off by default)",
    )
    parser.add_argument(
        "--require-data",
        action="store_true",
        help="make missing dataset files fatal during a dry run",
    )
    parser.add_argument("--manifest-out", help="override manifest output path")
    parser.add_argument(
        "--deep-base-checksum",
        action="store_true",
        help="rehash the 23 GB base file instead of trusting its verified download manifest",
    )
    return parser


def project_root(config: Mapping[str, Any]) -> Path:
    config_path = Path(str(config["_config_path"]))
    relative = config.get("paths", {}).get("project_root", "../..")
    return (config_path.parent / relative).resolve()


def _dataset_reports(config: Mapping[str, Any], *, require_data: bool) -> tuple[list[dict[str, Any]], list[Path]]:
    root = project_root(config)
    expected = config["active_adapter"].get("expected_experts")
    reports: list[dict[str, Any]] = []
    present: list[Path] = []
    for relative_path in config["active_adapter"]["datasets"]:
        path = (root / relative_path).resolve()
        if not path.is_file():
            if require_data:
                raise DatasetValidationError(f"dataset does not exist: {path}")
            reports.append({"path": str(path), "exists": False, "ok": None})
            continue
        validation = validate_jsonl(path, allowed_experts=expected)
        validation.update({"exists": True, "sha256": sha256_file(path), "bytes": path.stat().st_size})
        reports.append(validation)
        present.append(path)
    return reports, present


def _manifest_path(
    config: Mapping[str, Any], override: str | None, mode: str, stage: str
) -> Path:
    if override:
        return Path(override).expanduser().resolve()
    root = project_root(config)
    manifest_dir = root / config.get("paths", {}).get("manifest_dir", "artifacts/manifests")
    if stage == "preflight":
        name = f"preflight.{mode}.json"
    elif stage == "smoke-gate":
        name = f"smoke-gate-{config['run_name']}.{mode}.json"
    else:
        name = f"{config['run_name']}.{mode}.json"
    return manifest_dir / name


def _environment_issues(dependencies: Mapping[str, Any]) -> list[str]:
    issues: list[str] = []
    if dependencies.get("missing"):
        issues.append("missing=" + ",".join(dependencies["missing"]))
    if dependencies.get("incompatible"):
        issues.append("incompatible=" + ",".join(dependencies["incompatible"]))
    if dependencies.get("python_supported") is False:
        issues.append("python>=3.10 required")
    return issues


def _assert_one_step_profile(config: Mapping[str, Any]) -> None:
    training = config["training"]
    expected = {
        "max_steps": 1,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 1,
    }
    for key, value in expected.items():
        if training.get(key) != value:
            raise ConfigError(
                f"smoke-gate requires {key}={value}; use gemma4_12b_qlora_one_step.yaml"
            )
    if training.get("max_seq_length", 10**9) > 128:
        raise ConfigError("smoke-gate caps max_seq_length at 128")


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        base_config = load_training_config(args.config)
        if args.stage == "preflight":
            if args.execute:
                raise ConfigError("preflight is read-only; use --dry-run or omit the mode flag")
            config = dict(base_config)
            config["adapter_name"] = None
            config["run_name"] = "preflight"
        else:
            if not args.adapter:
                raise ConfigError(f"--adapter is required for {args.stage}")
            config = select_adapter(base_config, args.adapter, args.rank)
        if args.stage == "smoke-gate":
            _assert_one_step_profile(config)
        execute = bool(args.execute)
        dependencies = dependency_report(
            probe_device=True,
            require_full_training=args.stage == "train",
        )
        mode = "execute" if execute else "dry-run"
        root = project_root(config)
        preflight, heldout_cases = build_preflight_report(
            config,
            root,
            dependencies,
            deep_checksum=bool(args.deep_base_checksum),
        )

        if args.stage == "preflight":
            datasets: list[dict[str, Any]] = []
            dataset_paths: list[Path] = []
        else:
            # Always collect a manifest before turning missing data into a hard
            # execution failure; this keeps blocked-gate evidence inspectable.
            require_data = bool(args.require_data) and not execute
            datasets, dataset_paths = _dataset_reports(config, require_data=require_data)
        manifest = build_manifest(
            config,
            dependency_report=dependencies,
            datasets=datasets,
            mode=mode,
        )
        manifest["stage"] = args.stage
        manifest["preflight"] = preflight
        if args.stage == "smoke-gate":
            manifest["smoke_gate"] = {
                "executed": False,
                "ready": preflight["passed"],
                "passed": False,
            }
        manifest_path = write_json(
            _manifest_path(config, args.manifest_out, mode, args.stage), manifest
        )
        response: dict[str, Any] = {
            "ok": True if args.stage == "train" and not execute else preflight["passed"],
            "stage": args.stage,
            "mode": mode,
            "run_name": config["run_name"],
            "manifest": str(manifest_path),
            "missing_dependencies": dependencies["missing"],
            "incompatible_dependencies": dependencies.get("incompatible", []),
            "python_supported": dependencies.get("python_supported"),
            "datasets": datasets,
            "preflight_passed": preflight["passed"],
            "failed_gates": [
                name for name, gate in preflight["gates"].items() if not gate["passed"]
            ],
        }

        if args.stage == "preflight":
            print(json.dumps(response, ensure_ascii=False, indent=2))
            return 0 if preflight["passed"] else 3

        if args.stage == "smoke-gate" and not execute:
            print(json.dumps(response, ensure_ascii=False, indent=2))
            return 0 if preflight["passed"] else 3

        if execute:
            if args.dry_run:
                raise ConfigError("--execute and --dry-run cannot be combined")
            if not dependencies["ready"]:
                raise RuntimeError(
                    "training environment is not ready: " + "; ".join(_environment_issues(dependencies))
                )
            if not preflight["passed"]:
                failed = [name for name, gate in preflight["gates"].items() if not gate["passed"]]
                raise RuntimeError(
                    "scale-up preflight is blocked; real training is forbidden: "
                    + ", ".join(failed)
                )
            missing_active = [item["path"] for item in datasets if not item.get("exists")]
            if missing_active:
                raise RuntimeError(
                    "active adapter datasets are missing: " + ", ".join(missing_active)
                )
            if args.stage == "train":
                prior_smoke = verify_prior_smoke_gate(config, root, preflight)
                manifest["prior_smoke_gate"] = prior_smoke
                write_json(manifest_path, manifest)
                if not prior_smoke["passed"]:
                    raise RuntimeError(
                        "scale-up training requires a passing executed smoke-gate manifest"
                    )
            output_root = root / config.get("paths", {}).get("adapter_dir", "artifacts/adapters")
            output_dir = output_root / (
                f"smoke-gate-{config['run_name']}" if args.stage == "smoke-gate" else config["run_name"]
            )
            from .runtime import train_adapter

            selected_heldout = [
                case for case in heldout_cases if case["expert"] == config["adapter_name"]
            ]
            result = train_adapter(
                config,
                dataset_paths=dataset_paths,
                output_dir=output_dir,
                allow_model_download=args.allow_model_download,
                manifest=manifest,
                smoke_heldout_cases=selected_heldout if args.stage == "smoke-gate" else None,
            )
            response["training"] = result
            if args.stage == "smoke-gate":
                manifest["smoke_gate"] = result["smoke_gate"]
                write_json(manifest_path, manifest)
                response["ok"] = bool(result["smoke_gate"]["passed"])
                response["smoke_gate_passed"] = response["ok"]
        print(json.dumps(response, ensure_ascii=False, indent=2))
        return 0 if response["ok"] else 4
    except (ConfigError, DatasetValidationError, RuntimeError) as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False, indent=2), file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

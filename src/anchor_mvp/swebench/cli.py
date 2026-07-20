"""Command-line interface for local SWE metadata task-card imports."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
import sys
from typing import Sequence

from .batch import (
    BatchConfig,
    build_live_teacher,
    compile_manifest,
    compile_work_orders,
    load_replay_teacher,
    load_validated_cards,
    run_batch,
    write_work_orders,
)
from .importer import ImportConfig, import_metadata_cards
from .schema import SWEBenchValidationError


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="anchor-swebench",
        description=(
            "Import pinned, allowlisted SWE-style train metadata into task cards. "
            "This command never downloads datasets or container images."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    import_parser = subparsers.add_parser(
        "import",
        help="validate and optionally write metadata-only task cards",
    )
    import_parser.add_argument("--source-jsonl", type=Path, required=True)
    import_parser.add_argument("--dataset-id", required=True)
    import_parser.add_argument("--dataset-revision", required=True)
    import_parser.add_argument("--train-allowlist", type=Path, required=True)
    import_parser.add_argument("--heldout-registry", type=Path, required=True)
    import_parser.add_argument("--license-ledger", type=Path, required=True)
    import_parser.add_argument("--chain-index", type=Path)
    import_parser.add_argument("--cards-output", type=Path)
    import_parser.add_argument("--manifest-output", type=Path)
    import_parser.add_argument("--domain-id", default="python-repository")
    import_parser.add_argument("--language", default="python")
    import_parser.add_argument("--task-kind", default="issue-resolution")
    import_parser.add_argument(
        "--builder-expert-id",
        default="swe-shared-builder",
    )
    import_parser.add_argument(
        "--reviewer-expert-id",
        default="swe-shared-reviewer",
    )
    import_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate and print a content-free manifest without writing files",
    )
    compile_parser = subparsers.add_parser(
        "compile",
        help="split imported cards into five dependency-bound work orders",
    )
    compile_parser.add_argument("--cards-jsonl", type=Path, required=True)
    compile_parser.add_argument("--import-manifest", type=Path, required=True)
    compile_parser.add_argument("--work-orders-output", type=Path)
    compile_parser.add_argument(
        "--write",
        action="store_true",
        help="write work orders; without this flag compile is read-only",
    )
    batch_parser = subparsers.add_parser(
        "batch",
        help="run replay/live five-stage chains from a pinned batch config",
    )
    batch_parser.add_argument("--config", type=Path, required=True)
    batch_parser.add_argument(
        "--allow-live",
        action="store_true",
        help="explicitly permit provider requests when config mode is live",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "compile":
        try:
            cards = load_validated_cards(
                args.cards_jsonl.resolve(), args.import_manifest.resolve()
            )
            orders = compile_work_orders(cards)
            manifest = compile_manifest(cards)
            if args.write:
                if args.work_orders_output is None:
                    raise SWEBenchValidationError(
                        "--write requires --work-orders-output"
                    )
                write_work_orders(args.work_orders_output.resolve(), orders)
                manifest["files_written"] = 1
                manifest["work_orders_output"] = str(args.work_orders_output.resolve())
        except (OSError, SWEBenchValidationError, ValueError) as exc:
            print(f"anchor-swebench: {exc}", file=sys.stderr)
            return 2
        print(json.dumps(manifest, ensure_ascii=False, sort_keys=True))
        return 0
    if args.command == "batch":
        repo_root = Path(__file__).resolve().parents[3]
        try:
            config = BatchConfig.load(repo_root, args.config.resolve())
            if config.mode == "dry-run":
                cards = load_validated_cards(config.cards_jsonl, config.import_manifest)
                if config.max_cards is not None:
                    cards = cards[: config.max_cards]
                result = compile_manifest(cards)
            else:
                if config.mode == "live" and not args.allow_live:
                    raise SWEBenchValidationError(
                        "live mode requires the explicit --allow-live flag"
                    )
                teacher = (
                    load_replay_teacher(config.replay_responses_jsonl)
                    if config.mode == "replay"
                    and config.replay_responses_jsonl is not None
                    else build_live_teacher(config.provider or {})
                )
                batch = asyncio.run(run_batch(config, teacher=teacher))
                result = dict(batch.manifest)
        except (OSError, SWEBenchValidationError, ValueError, RuntimeError) as exc:
            print(f"anchor-swebench: {exc}", file=sys.stderr)
            return 2
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
        return 0
    if args.command != "import":
        parser.error("unknown command")
    config = ImportConfig(
        source_jsonl=args.source_jsonl,
        dataset_id=args.dataset_id,
        dataset_revision=args.dataset_revision,
        train_allowlist=args.train_allowlist,
        heldout_registry=args.heldout_registry,
        license_ledger=args.license_ledger,
        domain_id=args.domain_id,
        language=args.language,
        task_kind=args.task_kind,
        builder_expert_id=args.builder_expert_id,
        reviewer_expert_id=args.reviewer_expert_id,
        chain_index=args.chain_index,
        cards_output=args.cards_output,
        manifest_output=args.manifest_output,
        dry_run=args.dry_run,
    )
    try:
        result = import_metadata_cards(config)
    except (OSError, SWEBenchValidationError, ValueError) as exc:
        print(f"anchor-swebench: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(dict(result.manifest), ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

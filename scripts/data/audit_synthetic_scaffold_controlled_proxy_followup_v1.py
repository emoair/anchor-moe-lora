#!/usr/bin/env python3
"""Audit the metadata-only controlled-proxy follow-up contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from anchor_mvp.swebench.synthetic_scaffold_controlled_proxy_followup import (
    CONTRACT_PATH,
    ControlledProxyFollowupAuditError,
    audit_followup,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--contract", type=Path, default=Path(CONTRACT_PATH))
    args = parser.parse_args()
    try:
        summary = audit_followup(args.repo_root, args.contract)
    except ControlledProxyFollowupAuditError as exc:
        print(json.dumps({"error_code": exc.code, "status": "blocked"}))
        return 2
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Audit the additive controlled-proxy Q+O risk-evidence companion."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from anchor_mvp.swebench.synthetic_scaffold_controlled_proxy_risk_evidence import (
    CONTRACT_PATH,
    RiskEvidenceAuditError,
    audit_risk_evidence,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--contract", type=Path, default=Path(CONTRACT_PATH))
    args = parser.parse_args()
    try:
        summary = audit_risk_evidence(args.repo_root, args.contract)
    except RiskEvidenceAuditError as exc:
        print(json.dumps({"error_code": exc.code, "status": "blocked"}))
        return 2
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Print the content-free formal SWE-bench gate payload.

Unlike the full coordinator preflight, this reader never opens candidate task
JSONL.  It is safe for status, WebUI, and generic preflight surfaces.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from distillation_control import WorkspacePolicy  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    report = WorkspacePolicy(root).formal_preflight()
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

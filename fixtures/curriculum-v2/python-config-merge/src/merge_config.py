from __future__ import annotations

import json
import sys


def main(argv: list[str]) -> int:
    with open(argv[1], encoding="utf-8") as handle:
        print(json.dumps(json.load(handle)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

from __future__ import annotations

import sys


def main() -> int:
    for line in sys.stdin:
        print(line.rstrip())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import sys


def main(argv: list[str]) -> int:
    print(float(argv[0]) + float(argv[2]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

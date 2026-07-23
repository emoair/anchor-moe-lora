from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from anchor_mvp.research.synthetic_five_role_qonly_diagnostic_v1 import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())

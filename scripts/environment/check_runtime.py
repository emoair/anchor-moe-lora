"""Print a JSON readiness record without modifying the environment."""

from __future__ import annotations

import json

from anchor_mvp.training.dependencies import dependency_report


if __name__ == "__main__":
    print(json.dumps(dependency_report(probe_device=True), indent=2, ensure_ascii=False))

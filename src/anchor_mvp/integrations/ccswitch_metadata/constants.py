"""Trust anchors for the independently audited CC Switch metadata adapter."""

from __future__ import annotations

from typing import Final


SOURCE_REPOSITORY: Final = "https://github.com/farion1231/cc-switch"
SOURCE_TAG: Final = "v3.16.5"
SOURCE_COMMIT: Final = "8d1b3306d09a27b9d8fc29694791d8421aba5f93"
SOURCE_TAG_OBJECT: Final = "a58917a5d6d2a4ace4e7c7fd63dcee57355ef653"
RAW_BASE: Final = (
    f"https://raw.githubusercontent.com/farion1231/cc-switch/{SOURCE_COMMIT}/"
)

MAX_SNAPSHOT_BYTES: Final = 1_000_000
MAX_SOURCE_BYTES: Final = 2_000_000

# These are verification anchors, not runtime imports. The synchronizer hashes the
# response bytes and discards them; it never parses or executes upstream TS/Rust.
EXPECTED_SOURCE_FILES: Final = {
    "LICENSE": {
        "size": 1067,
        "sha256": "912b6a597d10c43b40a0909349ed95b052b17efb6502b4898e1b35dafb896755",
        "role": "license",
    },
    "src/config/opencodeProviderPresets.ts": {
        "size": 52961,
        "sha256": "3dc3103127b3a9a671f4a577480054cf5dc023a08833c207b60846eb3c601bae",
        "role": "provider_and_model_presets",
    },
    "src-tauri/src/database/schema.rs": {
        "size": 97208,
        "sha256": "c959050f1438196d968249715d0afbcb159c70b2933e92b9c105d5a07e0f5c1c",
        "role": "pricing_seed_and_database_schema",
    },
    "src-tauri/src/services/usage_stats.rs": {
        "size": 150275,
        "sha256": "edef115868a751c2c7f1eccbea0ca07fd530af7d9548d3c77656530652dedb54",
        "role": "model_alias_normalization",
    },
    "src-tauri/src/proxy/usage/calculator.rs": {
        "size": 9265,
        "sha256": "8a9f3d93894b309023d4d3795509c6f2df7b951bb1c3681e87b89f7566b8fec7",
        "role": "token_cost_formula",
    },
    "src-tauri/src/opencode_config.rs": {
        "size": 8236,
        "sha256": "8c26d3be1b93f4c3e68fb327a088baba4587d51e8c8edf435bf0d5d0404c4efe",
        "role": "opencode_config_injection",
    },
    "src-tauri/src/provider.rs": {
        "size": 56685,
        "sha256": "a6451e4d2fc1e698fe245141c0289c9b1d117446491690df5813e16c8493256c",
        "role": "opencode_provider_contract",
    },
}

ALLOWED_SOURCE_URLS: Final = frozenset(
    RAW_BASE + path for path in EXPECTED_SOURCE_FILES
)

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DOWNLOADER = ROOT / "scripts/data/download_swebench_train.ps1"


def test_direct_route_requires_up_physical_adapter_for_bound_address() -> None:
    source = DOWNLOADER.read_text(encoding="utf-8-sig")

    address_lookup = source.index("Get-NetIPAddress")
    physical_lookup = source.index("Get-NetAdapter -Physical")
    physical_claim = source.index("Route mode : direct physical binding")
    assert address_lookup < physical_lookup < physical_claim
    assert "$_.ifIndex -eq $Binding.InterfaceIndex" in source
    assert "$_.Status -eq 'Up'" in source
    assert "Virtual, TUN, TAP, VPN, and disconnected adapters are refused" in source


def test_curl_binding_is_reachable_only_after_physical_validation() -> None:
    source = DOWNLOADER.read_text(encoding="utf-8-sig")

    refusal = source.index("not an Up physical adapter")
    curl_binding = source.index("'--interface', $SourceAddress")
    assert refusal < curl_binding
    assert "--noproxy" in source

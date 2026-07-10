"""Read-only dependency and device probes used before a heavyweight run."""

from __future__ import annotations

import importlib.util
import ctypes
import os
import platform
import re
import sys
from importlib import metadata
from typing import Any


CORE_PACKAGES = {
    "torch": "torch",
    "transformers": "transformers",
    "peft": "peft",
    "accelerate": "accelerate",
    "bitsandbytes": "bitsandbytes",
    "sentencepiece": "sentencepiece",
    "protobuf": "google.protobuf",
}
FULL_TRAINING_PACKAGES = {
    "trl": "trl",
    "datasets": "datasets",
    "pyarrow": "pyarrow",
}
REPORTED_PACKAGES = {**CORE_PACKAGES, **FULL_TRAINING_PACKAGES}
MINIMUM_VERSIONS = {
    "transformers": (5, 10, 1),
    "peft": (0, 19, 0),
    "pyarrow": (21, 0, 0),
}
# PyArrow 24.0.0 reproducibly crashed this Windows/Python 3.11 runtime in
# arrow.dll before model loading. Keep the validated Arrow 21 line until the
# Windows wheel/runtime combination is re-qualified.
MAXIMUM_EXCLUSIVE_VERSIONS = {"pyarrow": (22, 0, 0)}


def _host_memory_report() -> dict[str, Any]:
    """Probe available physical memory without adding psutil to the gate."""

    try:
        if sys.platform == "win32":
            class MemoryStatusEx(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            status = MemoryStatusEx()
            status.dwLength = ctypes.sizeof(status)
            if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):  # type: ignore[attr-defined]
                raise OSError("GlobalMemoryStatusEx failed")
            total_bytes = int(status.ullTotalPhys)
            available_bytes = int(status.ullAvailPhys)
        else:
            page_size = int(os.sysconf("SC_PAGE_SIZE"))
            total_bytes = int(os.sysconf("SC_PHYS_PAGES")) * page_size
            available_bytes = int(os.sysconf("SC_AVPHYS_PAGES")) * page_size
        return {
            "probed": True,
            "total_memory_gib": round(total_bytes / 1024**3, 2),
            "available_memory_gib": round(available_bytes / 1024**3, 2),
        }
    except (AttributeError, OSError, TypeError, ValueError) as exc:
        return {"probed": True, "probe_error": f"{type(exc).__name__}: {exc}"}


def _numeric_version(value: str | None) -> tuple[int, ...] | None:
    if not value:
        return None
    match = re.match(r"^(\d+)(?:\.(\d+))?(?:\.(\d+))?", value)
    if not match:
        return None
    return tuple(int(part or 0) for part in match.groups())


def dependency_report(
    *,
    probe_device: bool = True,
    require_full_training: bool = True,
) -> dict[str, Any]:
    packages: dict[str, dict[str, Any]] = {}
    for distribution, module in REPORTED_PACKAGES.items():
        try:
            available = importlib.util.find_spec(module) is not None
        except (ImportError, ModuleNotFoundError):
            available = False
        try:
            version = metadata.version(distribution) if available else None
        except metadata.PackageNotFoundError:
            version = "unknown" if available else None
        minimum = MINIMUM_VERSIONS.get(distribution)
        maximum_exclusive = MAXIMUM_EXCLUSIVE_VERSIONS.get(distribution)
        parsed = _numeric_version(version)
        compatible = available and (
            (minimum is None or (parsed is not None and parsed >= minimum))
            and (
                maximum_exclusive is None
                or (parsed is not None and parsed < maximum_exclusive)
            )
        )
        packages[distribution] = {
            "available": available,
            "version": version,
            "minimum": ".".join(map(str, minimum)) if minimum else None,
            "maximum_exclusive": (
                ".".join(map(str, maximum_exclusive)) if maximum_exclusive else None
            ),
            "compatible": compatible,
        }

    device: dict[str, Any] = {"probed": False}
    if probe_device and packages["torch"]["available"]:
        try:
            import torch

            cuda = torch.cuda.is_available()
            device = {
                "probed": True,
                "cuda_available": cuda,
                "bf16_supported": bool(cuda and torch.cuda.is_bf16_supported()),
                "torch_cuda_version": torch.version.cuda,
            }
            if cuda:
                properties = torch.cuda.get_device_properties(0)
                free_bytes, total_bytes = torch.cuda.mem_get_info(0)
                device.update(
                    {
                        "name": properties.name,
                        "total_memory_gib": round(properties.total_memory / 1024**3, 2),
                        "free_memory_gib": round(free_bytes / 1024**3, 2),
                        "runtime_total_memory_gib": round(total_bytes / 1024**3, 2),
                        "capability": list(torch.cuda.get_device_capability(0)),
                    }
                )
        except Exception as exc:  # pragma: no cover - varies with driver state
            device = {"probed": True, "probe_error": f"{type(exc).__name__}: {exc}"}

    required = set(CORE_PACKAGES)
    if require_full_training:
        required.update(FULL_TRAINING_PACKAGES)
    missing = sorted(
        name for name in required if not packages[name]["available"]
    )
    incompatible = sorted(
        name
        for name in required
        if packages[name]["available"] and not packages[name]["compatible"]
    )
    python_supported = sys.version_info >= (3, 10)
    return {
        "python": sys.version.split()[0],
        "python_supported": python_supported,
        "minimum_python": "3.10",
        "platform": platform.platform(),
        "packages": packages,
        "required_profile": "full_training" if require_full_training else "smoke_core",
        "missing": missing,
        "incompatible": incompatible,
        "device": device,
        "host_memory": _host_memory_report(),
        "ready": not missing and not incompatible and python_supported,
    }

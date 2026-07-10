from __future__ import annotations

import asyncio
import subprocess


def query_vram_mb() -> float | None:
    """Return aggregate memory used by compute processes, or None without NVIDIA SMI."""

    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=used_memory",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (FileNotFoundError, subprocess.SubprocessError, OSError):
        return None
    if completed.returncode != 0:
        return None
    values: list[float] = []
    for line in completed.stdout.splitlines():
        try:
            values.append(float(line.strip()))
        except ValueError:
            continue
    return sum(values) if values else 0.0


class VramSampler:
    def __init__(self, interval_seconds: float = 0.2, *, enabled: bool = True) -> None:
        self.interval_seconds = interval_seconds
        self.enabled = enabled
        self.peak_mb: float | None = None
        self._task: asyncio.Task[None] | None = None
        self._stop = asyncio.Event()

    async def __aenter__(self) -> "VramSampler":
        if self.enabled:
            self._task = asyncio.create_task(self._sample_loop())
        return self

    async def __aexit__(self, *_: object) -> None:
        self._stop.set()
        if self._task:
            await self._task

    async def _sample_loop(self) -> None:
        while not self._stop.is_set():
            value = await asyncio.to_thread(query_vram_mb)
            if value is not None:
                self.peak_mb = value if self.peak_mb is None else max(self.peak_mb, value)
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=self.interval_seconds)
            except asyncio.TimeoutError:
                pass


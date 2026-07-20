"""Measure LoRA residual overlap with CUDA streams and event barriers.

This is a microbenchmark, not an inference-quality or end-to-end throughput
claim.  It deliberately computes one shared dense projection plus independent
low-rank residuals and compares serial execution with tick/tock streams.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Arguments:
    hidden_size: int
    token_count: int
    ranks: tuple[int, ...]
    warmup: int
    iterations: int
    max_streams: int
    dtype: str


def _parse_args() -> Arguments:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden-size", type=int, default=2048)
    parser.add_argument("--token-count", type=int, default=128)
    parser.add_argument("--ranks", default="8,16,32,64")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--max-streams", type=int, default=4)
    parser.add_argument(
        "--dtype", choices=("float16", "bfloat16", "float32"), default="float16"
    )
    raw = parser.parse_args()
    ranks = tuple(int(item.strip()) for item in raw.ranks.split(",") if item.strip())
    values = (
        raw.hidden_size,
        raw.token_count,
        raw.warmup,
        raw.iterations,
        raw.max_streams,
        *ranks,
    )
    if not ranks or any(value < 1 for value in values):
        parser.error("all dimensions, ranks, and iteration counts must be positive")
    return Arguments(
        hidden_size=raw.hidden_size,
        token_count=raw.token_count,
        ranks=ranks,
        warmup=raw.warmup,
        iterations=raw.iterations,
        max_streams=raw.max_streams,
        dtype=raw.dtype,
    )


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, round((len(ordered) - 1) * fraction))
    return ordered[index]


def _main() -> int:
    args = _parse_args()
    try:
        import torch
    except ImportError:
        print(json.dumps({"status": "unavailable", "reason": "torch_missing"}))
        return 2
    if not torch.cuda.is_available():
        print(json.dumps({"status": "unavailable", "reason": "cuda_missing"}))
        return 2

    device = torch.device("cuda:0")
    dtype = getattr(torch, args.dtype)
    generator = torch.Generator(device=device).manual_seed(7)
    x = torch.randn(
        args.token_count,
        args.hidden_size,
        device=device,
        dtype=dtype,
        generator=generator,
    )
    base_weight = torch.randn(
        args.hidden_size,
        args.hidden_size,
        device=device,
        dtype=dtype,
        generator=generator,
    )
    adapters = [
        (
            torch.randn(
                args.hidden_size,
                rank,
                device=device,
                dtype=dtype,
                generator=generator,
            ),
            torch.randn(
                rank,
                args.hidden_size,
                device=device,
                dtype=dtype,
                generator=generator,
            ),
        )
        for rank in args.ranks
    ]
    raw_groups: dict[int, list[tuple[Any, Any]]] = defaultdict(list)
    for rank, adapter in zip(args.ranks, adapters, strict=True):
        raw_groups[rank].append(adapter)
    grouped_adapters = [
        (
            rank,
            torch.stack([item[0] for item in group]),
            torch.stack([item[1] for item in group]),
        )
        for rank, group in sorted(raw_groups.items())
    ]
    streams = [
        torch.cuda.Stream(device=device)
        for _ in range(min(args.max_streams, len(adapters)))
    ]

    def serial_once() -> float:
        torch.cuda.synchronize(device)
        start = torch.cuda.Event(enable_timing=True)
        stop = torch.cuda.Event(enable_timing=True)
        start.record()
        base = x @ base_weight
        outputs = [base + (x @ left) @ right for left, right in adapters]
        stop.record()
        stop.synchronize()
        if not outputs:
            raise AssertionError("no adapter outputs")
        return float(start.elapsed_time(stop))

    def streamed_once() -> float:
        torch.cuda.synchronize(device)
        start = torch.cuda.Event(enable_timing=True)
        tick = torch.cuda.Event()
        done = [torch.cuda.Event() for _ in adapters]
        stop = torch.cuda.Event(enable_timing=True)
        start.record()
        base = x @ base_weight
        tick.record()
        outputs: list[Any] = [None] * len(adapters)
        for index, (left, right) in enumerate(adapters):
            stream = streams[index % len(streams)]
            with torch.cuda.stream(stream):
                stream.wait_event(tick)
                outputs[index] = base + (x @ left) @ right
                done[index].record(stream)
        current = torch.cuda.current_stream(device)
        for event in done:
            current.wait_event(event)
        stop.record(current)
        stop.synchronize()
        if any(output is None for output in outputs):
            raise AssertionError("missing adapter output")
        return float(start.elapsed_time(stop))

    def grouped_once() -> float:
        """Use one batched GEMM pair per rank bucket as a fusion baseline."""

        torch.cuda.synchronize(device)
        start = torch.cuda.Event(enable_timing=True)
        stop = torch.cuda.Event(enable_timing=True)
        start.record()
        base = x @ base_weight
        outputs = []
        for _rank, left, right in grouped_adapters:
            expanded = x.unsqueeze(0).expand(left.shape[0], -1, -1)
            outputs.append(
                base.unsqueeze(0) + torch.bmm(torch.bmm(expanded, left), right)
            )
        stop.record()
        stop.synchronize()
        if not outputs:
            raise AssertionError("no grouped adapter outputs")
        return float(start.elapsed_time(stop))

    for _ in range(args.warmup):
        serial_once()
        streamed_once()
        grouped_once()
    torch.cuda.reset_peak_memory_stats(device)
    serial = [serial_once() for _ in range(args.iterations)]
    streamed = [streamed_once() for _ in range(args.iterations)]
    grouped = [grouped_once() for _ in range(args.iterations)]
    properties = torch.cuda.get_device_properties(device)
    peer_access = []
    for peer in range(torch.cuda.device_count()):
        if peer == device.index:
            continue
        can_access = False
        probe = getattr(torch.cuda, "can_device_access_peer", None)
        if callable(probe):
            can_access = bool(probe(device.index, peer))
        peer_access.append({"peer": peer, "can_access": can_access})

    serial_median = statistics.median(serial)
    streamed_median = statistics.median(streamed)
    grouped_median = statistics.median(grouped)
    result = {
        "schema_version": "anchor.neural-swarm-cuda-probe.v1",
        "status": "completed",
        "claim_scope": "microbenchmark_only",
        "device": {
            "name": properties.name,
            "compute_capability": f"{properties.major}.{properties.minor}",
            "device_count": torch.cuda.device_count(),
            "peer_access": peer_access,
        },
        "config": {
            "hidden_size": args.hidden_size,
            "token_count": args.token_count,
            "ranks": list(args.ranks),
            "dtype": args.dtype,
            "streams": len(streams),
            "iterations": args.iterations,
        },
        "serial_ms": {
            "median": serial_median,
            "p95": _percentile(serial, 0.95),
        },
        "streamed_ms": {
            "median": streamed_median,
            "p95": _percentile(streamed, 0.95),
        },
        "rank_grouped_ms": {
            "median": grouped_median,
            "p95": _percentile(grouped, 0.95),
        },
        "median_speedup": serial_median / streamed_median,
        "rank_grouped_median_speedup": serial_median / grouped_median,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "notes": [
            "includes one shared dense projection and independent LoRA residuals",
            "does not include attention, KV reconstruction, routing, or inter-GPU traffic",
            "speedup below 1.0 means streams were slower on this workload",
            "rank-grouped path uses PyTorch batched GEMM and is not a fused custom kernel",
        ],
    }
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())

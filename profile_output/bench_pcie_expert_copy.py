#!/usr/bin/env python3
"""Benchmark production expert-cache copies on the local GPTQ payload shape."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch

from sglang.srt.layers.moe.expert_cache import _CUDA_BATCH_COPY


def make_payload(device: str, num_experts: int) -> list[torch.Tensor]:
    shapes = (
        (1024, 1024),
        (2048, 256),
        (1024, 16),
        (2048, 4),
    )
    dtypes = (torch.uint8, torch.uint8, torch.bfloat16, torch.bfloat16)
    return [
        torch.empty(
            (num_experts, *shape),
            dtype=dtype,
            device=device,
            pin_memory=device == "cpu",
        )
        for shape, dtype in zip(shapes, dtypes)
    ]


def build_copy_groups(
    host: list[torch.Tensor],
    staging: list[torch.Tensor],
    resident: list[torch.Tensor],
) -> tuple[
    list[tuple[torch.Tensor, torch.Tensor]],
    list[tuple[torch.Tensor, torch.Tensor]],
]:
    requested = (1, 7, 13, 29, 61, 101, 149, 211)
    hits = {1: 0, 13: 1, 61: 2}
    admissions = {7: 3, 29: 4, 101: 5}
    populate = []
    admit = []
    for host_tensor, target, resident_tensor in zip(host, staging, resident):
        for index, expert in enumerate(requested):
            source = (
                resident_tensor[hits[expert]]
                if expert in hits
                else host_tensor[expert]
            )
            populate.append((target[index], source))
        for expert, slot in admissions.items():
            admit.append((resident_tensor[slot], target[requested.index(expert)]))
    return populate, admit


def submit_individual(
    groups: tuple[list[tuple[torch.Tensor, torch.Tensor]], ...],
) -> None:
    for copies in groups:
        for destination, source in copies:
            destination.copy_(source, non_blocking=True)


def submit_batched(
    groups: tuple[list[tuple[torch.Tensor, torch.Tensor]], ...],
    stream: torch.cuda.Stream,
    *,
    prefer_overlap: bool,
) -> None:
    for copies in groups:
        if not _CUDA_BATCH_COPY.submit(
            copies, stream, prefer_overlap=prefer_overlap
        ):
            raise RuntimeError("production batched-copy path unexpectedly fell back")


def measure(
    mode: str,
    groups: tuple[list[tuple[torch.Tensor, torch.Tensor]], ...],
    stream: torch.cuda.Stream,
    iterations: int,
    *,
    prefer_overlap: bool = False,
) -> dict[str, float]:
    for _ in range(10):
        with torch.cuda.stream(stream):
            if mode == "individual":
                submit_individual(groups)
            else:
                submit_batched(
                    groups, stream, prefer_overlap=prefer_overlap
                )
    stream.synchronize()

    host_us = []
    gpu_us = []
    for _ in range(iterations):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        host_start = time.perf_counter_ns()
        with torch.cuda.stream(stream):
            start.record(stream)
            if mode == "individual":
                submit_individual(groups)
            else:
                submit_batched(
                    groups, stream, prefer_overlap=prefer_overlap
                )
            end.record(stream)
        host_us.append((time.perf_counter_ns() - host_start) / 1000)
        end.synchronize()
        gpu_us.append(start.elapsed_time(end) * 1000)

    return {
        "host_submit_median_us": statistics.median(host_us),
        "host_submit_p90_us": sorted(host_us)[int(iterations * 0.9)],
        "gpu_elapsed_median_us": statistics.median(gpu_us),
        "gpu_elapsed_p90_us": sorted(gpu_us)[int(iterations * 0.9)],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=200)
    args = parser.parse_args()

    host = make_payload("cpu", 256)
    staging = make_payload("cuda", 8)
    resident = make_payload("cuda", 8)
    for tensor in host:
        tensor.random_(0, 16)
    for tensor in resident:
        tensor.random_(0, 16)
    groups = build_copy_groups(host, staging, resident)
    stream = torch.cuda.Stream()

    individual = measure("individual", groups, stream, args.iterations)
    batched = measure("batched", groups, stream, args.iterations)
    batched_overlap = measure(
        "batched",
        groups,
        stream,
        args.iterations,
        prefer_overlap=True,
    )
    with torch.cuda.stream(stream):
        submit_batched(groups, stream, prefer_overlap=False)
    stream.synchronize()

    output = {
        "populate_copy_count": len(groups[0]),
        "admission_copy_count": len(groups[1]),
        "h2d_bytes": sum(
            source.nbytes
            for copies in groups
            for _, source in copies
            if source.device.type == "cpu"
        ),
        "d2d_bytes": sum(
            source.nbytes
            for copies in groups
            for _, source in copies
            if source.device.type == "cuda"
        ),
        "correct": all(
            torch.equal(destination.cpu(), source.cpu())
            for destination, source in groups[-1]
        ),
        "individual": individual,
        "batched": batched,
        "batched_prefer_overlap": batched_overlap,
    }
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Compare decode throughput for semantically related and unrelated prompts."""

import argparse
import json
import statistics
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests


PROMPTS = {
    "domain": [
        "In LLM inference systems, explain how KV cache design affects decode throughput.",
        "In LLM inference systems, explain how continuous batching affects decode throughput.",
        "In LLM inference systems, explain how MoE expert routing affects decode throughput.",
        "In LLM inference systems, compare tensor parallelism and expert parallelism.",
        "In LLM inference systems, explain how FlashAttention affects decode throughput.",
        "In LLM inference systems, explain how W4A16 quantization affects decode throughput.",
        "In LLM inference systems, explain how speculative decoding affects decode throughput.",
        "In LLM inference systems, explain how PCIe expert offload affects decode throughput.",
    ],
    "unrelated": [
        "In modern history, explain the principal causes of the French Revolution clearly.",
        "In bread making, explain how fermentation changes the flavor of sourdough clearly.",
        "In mathematics, explain why the square root of two is an irrational number clearly.",
        "In ecology, explain how nutrient cycling sustains a tropical rainforest clearly.",
        "In databases, explain how a B-tree index accelerates SQL query execution clearly.",
        "In poetry, explain how imagery supports the central theme of a sonnet clearly.",
        "In astronomy, explain why Earth's axial tilt produces seasonal changes clearly.",
        "In bicycle repair, explain how to diagnose a poorly shifting rear derailleur clearly.",
    ],
}


def run_request(base_url, prompt, barrier, max_new_tokens, timeout):
    events = []
    barrier.wait()
    request_start = time.perf_counter()
    with requests.post(
        f"{base_url}/generate",
        json={
            "text": prompt,
            "stream": True,
            "sampling_params": {
                "temperature": 0,
                "max_new_tokens": max_new_tokens,
                "ignore_eos": True,
            },
        },
        stream=True,
        timeout=timeout,
    ) as response:
        response.raise_for_status()
        final = None
        for raw_line in response.iter_lines(decode_unicode=True):
            if not raw_line or not raw_line.startswith("data:"):
                continue
            payload = raw_line[len("data:") :].strip()
            if payload == "[DONE]":
                break
            final = json.loads(payload)
            count = final.get("meta_info", {}).get("completion_tokens")
            if count is not None:
                now = time.perf_counter()
                if not events or count != events[-1][1]:
                    events.append((now, int(count)))

    if not events or final is None:
        raise RuntimeError("stream returned no completion-token events")
    return {
        "request_start": request_start,
        "events": events,
        "completion_tokens": events[-1][1],
        "text": final.get("text", ""),
    }


def run_cohort(base_url, prompts, max_new_tokens, timeout):
    barrier = threading.Barrier(len(prompts))
    with ThreadPoolExecutor(max_workers=len(prompts)) as pool:
        futures = [
            pool.submit(
                run_request, base_url, prompt, barrier, max_new_tokens, timeout
            )
            for prompt in prompts
        ]
        requests_out = [future.result() for future in futures]

    first_token_times = [item["events"][0][0] for item in requests_out]
    last_token_times = [item["events"][-1][0] for item in requests_out]
    decode_start = max(first_token_times)
    decode_end = max(last_token_times)
    tokens_after_barrier = 0
    per_request_tps = []
    request_summaries = []
    origin = min(item["request_start"] for item in requests_out)

    for prompt, item in zip(prompts, requests_out):
        baseline_count = 0
        for event_time, count in item["events"]:
            if event_time <= decode_start:
                baseline_count = count
            else:
                break
        tokens_after_barrier += item["completion_tokens"] - baseline_count

        first_time, first_count = item["events"][0]
        last_time, last_count = item["events"][-1]
        duration = last_time - first_time
        tps = (last_count - first_count) / duration if duration > 0 else None
        if tps is not None:
            per_request_tps.append(tps)
        request_summaries.append(
            {
                "prompt": prompt,
                "completion_tokens": item["completion_tokens"],
                "ttft_s": first_time - item["request_start"],
                "first_token_s": first_time - origin,
                "last_token_s": last_time - origin,
                "decode_tps": tps,
                "output_preview": item["text"][:160],
            }
        )

    duration = decode_end - decode_start
    return {
        "aggregate_decode_tps": tokens_after_barrier / duration,
        "aggregate_decode_tokens": tokens_after_barrier,
        "aggregate_decode_duration_s": duration,
        "median_request_decode_tps": statistics.median(per_request_tps),
        "min_first_token_to_barrier_s": decode_start - min(first_token_times),
        "requests": request_summaries,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--group", choices=PROMPTS, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:30116")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=900)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    prompts = PROMPTS[args.group]
    output = {
        "group": args.group,
        "base_url": args.base_url,
        "batch_size": len(prompts),
        "max_new_tokens": args.max_new_tokens,
        "warmups": [],
        "runs": [],
    }
    for index in range(args.warmups):
        result = run_cohort(
            args.base_url, prompts, args.max_new_tokens, args.timeout
        )
        output["warmups"].append(result)
        print(
            f"warmup {index + 1}: aggregate={result['aggregate_decode_tps']:.3f} "
            f"req_median={result['median_request_decode_tps']:.3f}",
            flush=True,
        )
    for index in range(args.runs):
        result = run_cohort(
            args.base_url, prompts, args.max_new_tokens, args.timeout
        )
        output["runs"].append(result)
        print(
            f"run {index + 1}: aggregate={result['aggregate_decode_tps']:.3f} "
            f"req_median={result['median_request_decode_tps']:.3f}",
            flush=True,
        )

    output["median_aggregate_decode_tps"] = statistics.median(
        run["aggregate_decode_tps"] for run in output["runs"]
    )
    output["median_request_decode_tps"] = statistics.median(
        run["median_request_decode_tps"] for run in output["runs"]
    )
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(
        f"median aggregate={output['median_aggregate_decode_tps']:.3f} "
        f"req_median={output['median_request_decode_tps']:.3f}",
        flush=True,
    )


if __name__ == "__main__":
    main()

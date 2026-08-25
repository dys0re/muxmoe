#!/usr/bin/env python3
"""Run a command behind cgroup and whole-system RAM safety limits.

The cgroup limit is the hard backstop for the complete command tree.  The
whole-system watchdog is intentionally set below the requested absolute limit
so unrelated processes cannot make the host cross it while the command runs.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
from pathlib import Path
import shlex
import shutil
import signal
import subprocess
import sys
import time
from typing import TextIO


GIB = 1024**3


def read_meminfo() -> dict[str, int]:
    values: dict[str, int] = {}
    with open("/proc/meminfo", encoding="utf-8") as meminfo:
        for line in meminfo:
            key, raw = line.split(":", 1)
            fields = raw.split()
            if fields:
                values[key] = int(fields[0]) * 1024
    return values


def read_int(path: Path | None) -> int | None:
    if path is None:
        return None
    try:
        raw = path.read_text(encoding="utf-8").strip()
        return None if raw == "max" else int(raw)
    except (FileNotFoundError, PermissionError, ValueError):
        return None


def systemctl(*args: str, check: bool = False) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["systemctl", "--user", *args],
        check=check,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def find_scope_cgroup(unit: str, timeout_s: float = 5.0) -> Path | None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        result = systemctl("show", unit, "--property=ControlGroup", "--value")
        control_group = result.stdout.strip()
        if result.returncode == 0 and control_group:
            path = Path("/sys/fs/cgroup") / control_group.lstrip("/")
            if path.exists():
                return path
        time.sleep(0.05)
    return None


def kill_scope(unit: str, launcher: subprocess.Popen[bytes]) -> None:
    result = systemctl(
        "kill",
        unit,
        "--kill-whom=all",
        "--signal=SIGKILL",
    )
    if result.returncode != 0 and launcher.poll() is None:
        try:
            os.killpg(launcher.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Guard a command with cgroup and whole-system RAM limits."
    )
    parser.add_argument(
        "--system-limit-gib",
        type=float,
        default=28.0,
        help="Absolute whole-system effective RAM limit (default: 28 GiB).",
    )
    parser.add_argument(
        "--trip-gib",
        type=float,
        default=26.5,
        help="Watchdog trip point, kept below the absolute limit (default: 26.5 GiB).",
    )
    parser.add_argument(
        "--command-max-gib",
        type=float,
        default=20.0,
        help="Maximum cgroup memory for the complete command tree (default: 20 GiB).",
    )
    parser.add_argument(
        "--command-high-gib",
        type=float,
        default=18.0,
        help="Cgroup throttling threshold (default: 18 GiB).",
    )
    parser.add_argument(
        "--command-swap-max-gib",
        type=float,
        default=1.0,
        help="Maximum cgroup swap usage (default: 1 GiB).",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=0.1,
        help="Sampling interval in seconds (default: 0.1).",
    )
    parser.add_argument(
        "--log",
        type=Path,
        required=True,
        help="CSV sample log path; a sibling .summary.json is also written.",
    )
    parser.add_argument(
        "--command-log",
        type=Path,
        help="Optional file receiving the guarded command's stdout and stderr.",
    )
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.command and args.command[0] == "--":
        args.command = args.command[1:]
    if not args.command:
        parser.error("a command is required after --")
    if not 0 < args.trip_gib < args.system_limit_gib:
        parser.error("--trip-gib must be positive and below --system-limit-gib")
    if args.system_limit_gib - args.trip_gib < 1.0:
        parser.error("leave at least 1 GiB between trip and absolute system limit")
    if args.interval <= 0 or args.interval > 1:
        parser.error("--interval must be in (0, 1]")
    return args


def write_sample(
    writer: csv.writer,
    log_file: TextIO,
    start: float,
    info: dict[str, int],
    cgroup_current: int | None,
    event: str,
) -> tuple[float, float]:
    system_used = info["MemTotal"] - info["MemAvailable"]
    swap_used = info.get("SwapTotal", 0) - info.get("SwapFree", 0)
    elapsed = time.monotonic() - start
    writer.writerow(
        [
            dt.datetime.now(dt.timezone.utc).astimezone().isoformat(),
            f"{elapsed:.3f}",
            f"{system_used / GIB:.4f}",
            f"{info['MemAvailable'] / GIB:.4f}",
            f"{swap_used / GIB:.4f}",
            "" if cgroup_current is None else f"{cgroup_current / GIB:.4f}",
            event,
        ]
    )
    log_file.flush()
    return system_used / GIB, info["MemAvailable"] / GIB


def main() -> int:
    args = parse_args()
    if shutil.which("systemd-run") is None or shutil.which("systemctl") is None:
        raise SystemExit("systemd-run/systemctl is required for the hard cgroup limit")

    initial = read_meminfo()
    baseline_used_gib = (initial["MemTotal"] - initial["MemAvailable"]) / GIB
    if baseline_used_gib >= args.trip_gib:
        raise SystemExit(
            f"refusing to start: baseline effective RAM use is "
            f"{baseline_used_gib:.2f} GiB, at/above the {args.trip_gib:.2f} GiB trip point"
        )

    # Keep the command's hard cap compatible with the current host headroom.
    safe_command_max_gib = min(
        args.command_max_gib,
        args.trip_gib - baseline_used_gib - 0.5,
    )
    if safe_command_max_gib <= 0:
        raise SystemExit("not enough RAM headroom for a guarded command")
    safe_command_high_gib = min(args.command_high_gib, safe_command_max_gib * 0.98)

    args.log.parent.mkdir(parents=True, exist_ok=True)
    summary_path = args.log.with_suffix(".summary.json")
    unit_base = f"sglang-ramguard-{os.getpid()}-{int(time.time())}"
    unit = f"{unit_base}.scope"
    scope_command = [
        "systemd-run",
        "--user",
        "--scope",
        f"--unit={unit_base}",
        "--property=MemoryAccounting=yes",
        f"--property=MemoryHigh={int(safe_command_high_gib * GIB)}",
        f"--property=MemoryMax={int(safe_command_max_gib * GIB)}",
        f"--property=MemorySwapMax={int(args.command_swap_max_gib * GIB)}",
        "--quiet",
        "--",
        *args.command,
    ]

    command_log_file: TextIO | None = None
    if args.command_log is not None:
        args.command_log.parent.mkdir(parents=True, exist_ok=True)
        command_log_file = args.command_log.open("w", encoding="utf-8")

    start = time.monotonic()
    launcher = subprocess.Popen(
        scope_command,
        start_new_session=True,
        stdout=command_log_file,
        stderr=subprocess.STDOUT if command_log_file is not None else None,
    )
    scope_path = find_scope_cgroup(unit)
    memory_current_path = scope_path / "memory.current" if scope_path else None
    memory_peak_path = scope_path / "memory.peak" if scope_path else None
    memory_events_path = scope_path / "memory.events" if scope_path else None

    triggered = False
    trigger_reason: str | None = None
    peak_system_used_gib = baseline_used_gib
    peak_scope_gib = 0.0

    def stop_from_signal(signum: int, _frame: object) -> None:
        nonlocal triggered, trigger_reason
        triggered = True
        trigger_reason = f"received signal {signum}"
        kill_scope(unit, launcher)

    old_sigint = signal.signal(signal.SIGINT, stop_from_signal)
    old_sigterm = signal.signal(signal.SIGTERM, stop_from_signal)

    try:
        with args.log.open("w", newline="", encoding="utf-8") as log_file:
            writer = csv.writer(log_file)
            writer.writerow(
                [
                    "timestamp",
                    "elapsed_s",
                    "system_effective_used_gib",
                    "mem_available_gib",
                    "swap_used_gib",
                    "scope_memory_gib",
                    "event",
                ]
            )
            while launcher.poll() is None:
                info = read_meminfo()
                scope_current = read_int(memory_current_path)
                system_used_gib, _ = write_sample(
                    writer, log_file, start, info, scope_current, ""
                )
                peak_system_used_gib = max(peak_system_used_gib, system_used_gib)
                if scope_current is not None:
                    peak_scope_gib = max(peak_scope_gib, scope_current / GIB)

                if system_used_gib >= args.trip_gib:
                    triggered = True
                    trigger_reason = (
                        f"system effective RAM reached {system_used_gib:.3f} GiB "
                        f"(trip={args.trip_gib:.3f} GiB, "
                        f"absolute limit={args.system_limit_gib:.3f} GiB)"
                    )
                    write_sample(
                        writer,
                        log_file,
                        start,
                        info,
                        scope_current,
                        f"TRIP: {trigger_reason}",
                    )
                    kill_scope(unit, launcher)
                    break
                time.sleep(args.interval)

            try:
                returncode = launcher.wait(timeout=10)
            except subprocess.TimeoutExpired:
                triggered = True
                trigger_reason = trigger_reason or "scope launcher did not exit"
                kill_scope(unit, launcher)
                returncode = launcher.wait(timeout=5)

            final_info = read_meminfo()
            final_scope = read_int(memory_current_path)
            write_sample(
                writer,
                log_file,
                start,
                final_info,
                final_scope,
                "EXIT",
            )
    finally:
        signal.signal(signal.SIGINT, old_sigint)
        signal.signal(signal.SIGTERM, old_sigterm)
        if command_log_file is not None:
            command_log_file.close()

    cgroup_peak = read_int(memory_peak_path)
    if cgroup_peak is not None:
        peak_scope_gib = max(peak_scope_gib, cgroup_peak / GIB)
    memory_events = None
    if memory_events_path is not None:
        try:
            memory_events = memory_events_path.read_text(encoding="utf-8").strip()
        except (FileNotFoundError, PermissionError):
            pass
    cgroup_oom_kills = 0
    if memory_events:
        for line in memory_events.splitlines():
            key, _, value = line.partition(" ")
            if key in {"oom_kill", "oom_group_kill"} and value.isdigit():
                cgroup_oom_kills += int(value)
    if cgroup_oom_kills and not triggered:
        triggered = True
        trigger_reason = (
            f"cgroup hard limit activated; kernel OOM-killed "
            f"{cgroup_oom_kills} command process(es)"
        )

    summary = {
        "command": args.command,
        "command_display": shlex.join(args.command),
        "command_log": str(args.command_log) if args.command_log else None,
        "returncode": returncode,
        "watchdog_triggered": triggered,
        "trigger_reason": trigger_reason,
        "cgroup_oom_kills": cgroup_oom_kills,
        "baseline_system_effective_used_gib": round(baseline_used_gib, 4),
        "peak_system_effective_used_gib": round(peak_system_used_gib, 4),
        "peak_scope_memory_gib": round(peak_scope_gib, 4),
        "system_absolute_limit_gib": args.system_limit_gib,
        "system_trip_gib": args.trip_gib,
        "scope_memory_high_gib": round(safe_command_high_gib, 4),
        "scope_memory_max_gib": round(safe_command_max_gib, 4),
        "scope_swap_max_gib": args.command_swap_max_gib,
        "scope_unit": unit,
        "scope_path": str(scope_path) if scope_path else None,
        "memory_events": memory_events,
        "elapsed_s": round(time.monotonic() - start, 3),
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False), file=sys.stderr)
    # Failed transient scopes otherwise remain listed until reset.
    systemctl("reset-failed", unit)
    return 137 if triggered else returncode


if __name__ == "__main__":
    raise SystemExit(main())

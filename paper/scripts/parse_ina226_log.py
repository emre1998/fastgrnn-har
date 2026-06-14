#!/usr/bin/env python3
"""Summarize INA226 CSV logs from arduino/ina226_meter.

The Arduino sketch prints a few human-readable banner lines followed by:

    t_ms,mA,mV,mW

This helper ignores non-CSV lines and reports a stable average over either
the last N seconds or the last N rows.
"""

from __future__ import annotations

import argparse
import math
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Sample:
    t_ms: int
    mA: float
    mV: float
    mW: float


def parse_samples(text: str) -> list[Sample]:
    samples: list[Sample] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.lower().startswith("t_ms"):
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 4:
            continue
        try:
            samples.append(
                Sample(
                    t_ms=int(float(parts[0])),
                    mA=float(parts[1]),
                    mV=float(parts[2]),
                    mW=float(parts[3]),
                )
            )
        except ValueError:
            continue
    return samples


def choose_window(
    samples: list[Sample],
    last_seconds: float | None,
    last_rows: int | None,
) -> list[Sample]:
    if last_rows is not None:
        if last_rows <= 0:
            raise ValueError("--last-rows must be positive")
        return samples[-last_rows:]

    if last_seconds is None:
        last_seconds = 30.0
    if last_seconds <= 0:
        raise ValueError("--last-seconds must be positive")

    cutoff = samples[-1].t_ms - int(last_seconds * 1000.0)
    return [sample for sample in samples if sample.t_ms >= cutoff]


def mean(values: list[float]) -> float:
    return statistics.fmean(values)


def stdev(values: list[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else 0.0


def summarize(samples: list[Sample]) -> dict[str, float]:
    current = [sample.mA for sample in samples]
    voltage = [sample.mV for sample in samples]
    power = [sample.mW for sample in samples]
    duration_s = (samples[-1].t_ms - samples[0].t_ms) / 1000.0
    return {
        "rows": float(len(samples)),
        "duration_s": duration_s,
        "mA_mean": mean(current),
        "mA_stdev": stdev(current),
        "mV_mean": mean(voltage),
        "mV_stdev": stdev(voltage),
        "mW_mean": mean(power),
        "mW_stdev": stdev(power),
        "mW_from_vi": mean(current) * mean(voltage) / 1000.0,
        "mA_min": min(current),
        "mA_max": max(current),
    }


def format_summary(summary: dict[str, float]) -> str:
    return "\n".join(
        [
            f"rows: {int(summary['rows'])}",
            f"duration_s: {summary['duration_s']:.1f}",
            f"current_mA_mean: {summary['mA_mean']:.3f}",
            f"current_mA_stdev: {summary['mA_stdev']:.3f}",
            f"current_mA_min_max: {summary['mA_min']:.3f}, {summary['mA_max']:.3f}",
            f"bus_mV_mean: {summary['mV_mean']:.1f}",
            f"bus_mV_stdev: {summary['mV_stdev']:.1f}",
            f"power_mW_mean: {summary['mW_mean']:.3f}",
            f"power_mW_stdev: {summary['mW_stdev']:.3f}",
            f"power_mW_from_mean_v_i: {summary['mW_from_vi']:.3f}",
        ]
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "logfile",
        nargs="?",
        type=Path,
        help="INA226 Serial Monitor log. Reads stdin when omitted.",
    )
    parser.add_argument(
        "--last-seconds",
        type=float,
        default=30.0,
        help="Average over the last N seconds of samples (default: 30).",
    )
    parser.add_argument(
        "--last-rows",
        type=int,
        default=None,
        help="Average over the last N CSV rows instead of --last-seconds.",
    )
    args = parser.parse_args()

    text = args.logfile.read_text(encoding="utf-8") if args.logfile else sys.stdin.read()
    samples = parse_samples(text)
    if not samples:
        print("No INA226 CSV samples found.", file=sys.stderr)
        return 2

    window = choose_window(samples, args.last_seconds, args.last_rows)
    if not window:
        print("No samples in the requested averaging window.", file=sys.stderr)
        return 2
    if len(window) == 1:
        print("Warning: only one sample in averaging window.", file=sys.stderr)

    summary = summarize(window)
    if not math.isclose(summary["mW_mean"], summary["mW_from_vi"], rel_tol=0.10):
        print(
            "Warning: INA226 power differs from mean(V)*mean(I) by more than 10%.",
            file=sys.stderr,
        )
    print(format_summary(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

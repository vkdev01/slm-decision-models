#!/usr/bin/env python3
"""Quality gate: fail the build when a decision model regresses.

Compares an ``eval.json`` produced by ``benchmarks/evaluate.py`` against a committed
baseline. Thresholds are deliberately conservative - a gate that fires on noise gets
disabled within a month.
"""
from __future__ import annotations

import argparse
import json

# (metric, rule, argument)
GATES = [
    ("accuracy", "min_delta", -0.01),          # no more than 1 point worse
    ("cost_per_case", "max_ratio", 1.05),      # no more than 5% more expensive
    ("abstain_rate", "band", (0.8, 1.2)),      # a jump usually means a threshold bug
    ("ece", "max_abs", 0.05),                  # calibration must stay trustworthy
    ("p95_ms", "max_ratio", 1.20),             # no more than 20% slower
]


def check(current: dict, baseline: dict, gates=GATES):
    failures = []
    for metric, rule, arg in gates:
        cur, ref = current.get(metric), baseline.get(metric)
        if cur is None or ref is None:
            continue
        if rule == "min_delta" and cur - ref < arg:
            failures.append("%s: %.4f vs %.4f (delta %+.4f, floor %+.2f)"
                            % (metric, cur, ref, cur - ref, arg))
        elif rule == "max_ratio" and cur > ref * arg:
            failures.append("%s: %.4f exceeds %.4f x %.2f" % (metric, cur, ref, arg))
        elif rule == "band" and not (ref * arg[0] <= cur <= ref * arg[1]):
            failures.append("%s: %.4f outside [%.2f, %.2f] x %.4f"
                            % (metric, cur, arg[0], arg[1], ref))
        elif rule == "max_abs" and cur > arg:
            failures.append("%s: %.4f above %.2f" % (metric, cur, arg))
    return failures


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("current")
    ap.add_argument("--baseline", default="benchmarks/baseline.json")
    args = ap.parse_args()

    current = json.load(open(args.current))
    baseline = json.load(open(args.baseline))

    failures = check(current, baseline)
    if failures:
        print("FAIL")
        for f in failures:
            print("  -", f)
        return 1
    print("PASS")
    for metric, _rule, _arg in GATES:
        if metric in current:
            print("  %-14s %.4f" % (metric, current[metric]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

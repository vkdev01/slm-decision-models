#!/usr/bin/env python3
"""Reproducible evaluation harness.

Runs the decision policies on a deterministic synthetic workload so the numbers are
comparable across runs and machines, and writes ``eval.json`` for the CI quality gate.

Replace ``build_workload`` with your real validation slice; everything downstream -
the policies, the metrics, the JSON shape - stays the same.
"""
from __future__ import annotations

import argparse
import json
import platform
import time
from datetime import UTC, datetime

import numpy as np
from slm_decision import (
    CostOptimalPolicy,
    DecisionSpec,
    RulePolicy,
    TemperatureCalibrator,
    evaluate_policy,
)

SPEC = DecisionSpec(
    name="loan_triage",
    actions=["approve", "decline"],
    costs={("approve", "bad"): 120.0, ("approve", "good"): 0.0,
           ("decline", "bad"): 0.0, ("decline", "good"): 24.0},
    max_abstain_rate=0.0,
)


def build_workload(n: int = 6000, seed: int = 0, sharpness: float = 3.0):
    """A calibrated latent signal, deliberately over-sharpened by ``sharpness``."""
    rng = np.random.default_rng(seed)
    u = rng.normal(0, 1.0, n)
    y = (rng.random(n) < 1 / (1 + np.exp(-u))).astype(int)
    p = 1 / (1 + np.exp(-sharpness * u))
    split = int(0.5 * n)
    return p[:split], y[:split], p[split:], y[split:]


def latency_p95(fn, signals, repeat: int = 3) -> float:
    best = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn(signals)
        best.append((time.perf_counter() - t0) * 1000.0)
    return float(np.median(best))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="eval.json")
    ap.add_argument("--n", type=int, default=6000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    p_val, y_val, p_test, y_test = build_workload(args.n, args.seed)

    cal = TemperatureCalibrator().fit(p_val, y_val)
    p_cal = cal.transform(p_test)

    results = {}
    for name, scores in (("rule@0.5 (raw)", p_test), ("rule@0.5 (calibrated)", p_cal)):
        rep = evaluate_policy(SPEC, RulePolicy(SPEC, threshold=0.5), y_test, probs=scores)
        results[name] = rep.to_dict()

    for name, scores in (("cost-optimal (raw)", p_test), ("cost-optimal (calibrated)", p_cal)):
        rep = evaluate_policy(SPEC, CostOptimalPolicy(SPEC), y_test, probs=scores)
        results[name] = rep.to_dict()

    chosen = results["cost-optimal (calibrated)"]
    payload = {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "n": args.n,
        "seed": args.seed,
        "temperature": round(cal.temperature, 4),
        # headline metrics consumed by gate.py
        "accuracy": round(chosen["accuracy"], 4),
        "macro_f1": round(chosen["macro_f1"], 4) if chosen["macro_f1"] is not None else None,
        "cost_per_case": round(chosen["cost_per_case"], 4),
        "abstain_rate": round(chosen["abstain_rate"], 4),
        "ece": round(chosen["ece"], 4),
        # NOTE: this is the policy layer (thresholding + abstain cap), not model
        # inference. Model latency belongs in benchmarks/bench_latency.py.
        "p95_ms": round(latency_p95(CostOptimalPolicy(SPEC).act_batch, p_cal), 4),
        "p95_ms_scope": "policy layer only",
        "policies": results,
    }
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2)

    print("%-28s %s" % ("policy", "acc / cost / abstain / ece"))
    for name, r in results.items():
        print("%-28s %.3f / %6.2f / %.3f / %.4f"
              % (name, r["accuracy"], r["cost_per_case"], r["abstain_rate"], r["ece"]))
    print("\nwrote %s" % args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# SLM Decision Models

### Comprehensive guide to converting small language models into decision models

[![Docs](https://img.shields.io/badge/docs-GitHub%20Pages-4f46e5)](https://vkdev01.github.io/slm-decision-models/)
[![License](https://img.shields.io/badge/license-MIT-blue)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![Tests](https://github.com/vkdev01/slm-decision-models/actions/workflows/ci.yml/badge.svg)](https://github.com/vkdev01/slm-decision-models/actions/workflows/ci.yml)

A language model becomes a **decision model** the moment its output picks an action. That single change raises the
bar: the model must return a defined action, a calibrated confidence, an explicit abstain path, a bounded latency and
a documented failure mode.

This repository is a working guide to doing that with **small language models (10M–4B parameters)** — the theory, the
implementations, the numbers, the tests, and the deployment.

**📖 Read the guide: https://vkdev01.github.io/slm-decision-models/**

---

## What is in here

| Path | Contents |
|---|---|
| `docs/` | The full guide, as static HTML (this is the GitHub Pages source) |
| `code/src/slm_decision/` | The Python library used by every example: specs, policies, calibration, metrics, evaluation |
| `notebooks/` | Four runnable notebooks (distillation, triage under costs, prompt experiments, evaluation) |
| `tests/` | 47 pytest tests covering the arithmetic, the policies, calibration and the report |
| `benchmarks/` | A reproducible evaluation harness plus the CI quality gate |
| `.github/workflows/ci.yml` | Lint, types, tests, quality gate, notebook execution |

## Quick start

```bash
git clone https://github.com/vkdev01/slm-decision-models.git
cd slm-decision-models

python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e code/            # installs slm_decision

pytest tests -q                 # 47 tests, ~2 s, no torch required
```

Optional extras, only needed for the model wrappers:

```bash
pip install -e "code/[ml]"      # torch, transformers, datasets, scikit-learn
```

## The sixty-second version

```python
from slm_decision import DecisionSpec, CostOptimalPolicy, evaluate_policy

spec = DecisionSpec(
    name="loan_triage",
    actions=["approve", "decline"],
    # (action, outcome) -> cost. Outcomes are sorted, so 0 = bad, 1 = good.
    costs={("approve", "bad"): 120_000, ("approve", "good"): 0,
           ("decline", "bad"): 0,       ("decline", "good"): 24_000},
)

# the threshold is derived from the costs: 120k / (120k + 24k) = 0.833
print(spec.thresholds())            # {'decline': 0.0, 'approve': 0.8333}

policy = CostOptimalPolicy(spec)
report = evaluate_policy(spec, policy, y_true=y_test, probs=p_test)
print(report.cost_per_case, report.abstain_rate, report.ece)
```

The three numbers that decide go/no-go are **cost per case**, **abstain rate** and **ECE** — not accuracy.
The guide explains why in [Decision theory](https://vkdev01.github.io/slm-decision-models/guides/02-decision-theory.html).

## The guide

**Fundamentals** — [Quick start](https://vkdev01.github.io/slm-decision-models/guides/00-quick-start.html) ·
[SLM fundamentals](https://vkdev01.github.io/slm-decision-models/guides/01-slm-fundamentals.html) ·
[Decision theory](https://vkdev01.github.io/slm-decision-models/guides/02-decision-theory.html) ·
[Why SLMs](https://vkdev01.github.io/slm-decision-models/guides/03-why-slms.html)

**Implementation** — [Distillation](https://vkdev01.github.io/slm-decision-models/guides/04-distillation.html) ·
[Fine-tuning](https://vkdev01.github.io/slm-decision-models/guides/05-finetuning.html) ·
[Prompt engineering](https://vkdev01.github.io/slm-decision-models/guides/06-prompt-engineering.html) ·
[Hybrid systems](https://vkdev01.github.io/slm-decision-models/guides/07-hybrid-systems.html)

**Use cases** — [Healthcare](https://vkdev01.github.io/slm-decision-models/use-cases/healthcare.html) ·
[Finance](https://vkdev01.github.io/slm-decision-models/use-cases/finance.html) ·
[Industrial](https://vkdev01.github.io/slm-decision-models/use-cases/industrial.html) ·
[Gaming](https://vkdev01.github.io/slm-decision-models/use-cases/gaming.html) ·
[Education](https://vkdev01.github.io/slm-decision-models/use-cases/education.html) ·
[Autonomous systems](https://vkdev01.github.io/slm-decision-models/use-cases/autonomous.html)

**Testing** — [Unit](https://vkdev01.github.io/slm-decision-models/testing/unit-testing.html) ·
[Integration](https://vkdev01.github.io/slm-decision-models/testing/integration-testing.html) ·
[Edge cases](https://vkdev01.github.io/slm-decision-models/testing/edge-cases.html) ·
[Benchmarking](https://vkdev01.github.io/slm-decision-models/testing/benchmarking.html)

**Deployment** — [Docker](https://vkdev01.github.io/slm-decision-models/deployment/docker.html) ·
[Cloud](https://vkdev01.github.io/slm-decision-models/deployment/cloud.html) ·
[Edge](https://vkdev01.github.io/slm-decision-models/deployment/edge.html) ·
[CI/CD](https://vkdev01.github.io/slm-decision-models/deployment/cicd.html)

**Benchmarks** — [Metrics](https://vkdev01.github.io/slm-decision-models/benchmarks/metrics.html) ·
[Comparison](https://vkdev01.github.io/slm-decision-models/benchmarks/comparison.html) ·
[Optimization](https://vkdev01.github.io/slm-decision-models/benchmarks/optimization.html)

## The library

```python
from slm_decision import (
    DecisionSpec,              # actions, cost matrix, abstain policy, acceptable actions
    RulePolicy,                # the baseline you must beat
    CostOptimalPolicy,         # argmin expected cost, with an abstain cap
    TemperatureCalibrator,     # scalar calibration, monotone
    IsotonicCalibrator,        # non-parametric
    ConformalCalibrator,       # coverage guarantee
    evaluate_policy,           # one report per slice
    ece, brier, macro_f1, expected_cost, risk_coverage,
)
```

The core is **numpy-only**. `torch` and `transformers` are imported lazily, so specs, policies, metrics and
calibration run on a laptop with no ML stack installed — which is also why CI finishes in seconds.

## Notebooks

```bash
jupyter lab notebooks/
```

| Notebook | What it shows |
|---|---|
| `01-distillation-basics.ipynb` | Soft vs hard targets: a 20-point gap at n=100, 1 point at n=8000 |
| `02-finetuning-medical.ipynb` | Three-action triage under an asymmetric cost matrix (synthetic — **not** a medical device) |
| `03-prompt-engineering.ipynb` | Four prompt variants, one harness, McNemar's test on paired results |
| `04-evaluation-metrics.ipynb` | Reliability, cost curve and risk-coverage: threshold 0.83 vs the naive 0.5 |

All four run in under 15 minutes on 2 vCPU with **no network access and no model downloads**.

## Benchmarks and the quality gate

```bash
python benchmarks/evaluate.py --out eval.json
python benchmarks/gate.py eval.json --baseline benchmarks/baseline.json
```

The gate fails a build on: accuracy −1 point, cost +5%, abstain rate outside ±20% of baseline, ECE above 0.05, or p95
latency +20%. Thresholds are conservative on purpose — a gate that fires on noise gets switched off within a month.

## Honest notes about the numbers

Every performance table in the guide is either attributed to a published paper (DistilBERT, LoRA, QLoRA, CoT,
self-consistency, conformal prediction) or explicitly marked **illustrative**. Latency and accuracy depend on your
hardware, tokenizer, label set and data. The scripts here are the ones to run on your own workload — replace the
numbers before quoting them.

The `p95_ms` figure in `eval.json` measures the **policy layer** (thresholding and the abstain cap), not model
inference.

## Contributing

Issues and pull requests are welcome. If you change a number in the guide, change the script that produced it in the
same commit — otherwise the docs drift and the next reader cannot tell which claim is still true.

## License

MIT — see [LICENSE](LICENSE).

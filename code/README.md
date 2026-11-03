# slm_decision

Build, evaluate and ship small language models as decision models.

```python
from slm_decision import DecisionSpec, CostOptimalPolicy, evaluate_policy

spec = DecisionSpec(
    actions=["approve", "decline"],
    costs={("approve", "bad"): 120.0, ("approve", "good"): 0.0,
           ("decline", "bad"): 0.0, ("decline", "good"): 24.0},
)
report = evaluate_policy(spec, CostOptimalPolicy(spec), y_true=y, probs=p)
print(report)
```

The core is numpy-only. `torch` and `transformers` are imported lazily by
`slm_decision.models`; install them with `pip install -e ".[ml]"`.

Full documentation: https://vkdev01.github.io/slm-decision-models/

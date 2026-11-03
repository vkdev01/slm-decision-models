"""Model wrappers: SLMs as decision models, with torch imported lazily.

Nothing in this module imports torch or transformers at module import time, so the
rest of the library (specs, policies, metrics, calibration) stays usable on a
machine with no ML stack installed.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from .spec import DecisionSpec


class MissingDependency(RuntimeError):
    """Raised when a model path is used without torch/transformers installed."""


@dataclass
class DecisionResult:
    """A single decision, with the evidence needed to audit it later."""

    action: str
    confidence: float
    probs: Dict[str, float] = field(default_factory=dict)
    raw: str = ""
    latency_ms: float = 0.0
    reason_code: str = ""
    model: str = ""

    # convenience aliases so that `result.decision` and `result["decision"]` both work
    @property
    def decision(self) -> str:
        return self.action

    def __getitem__(self, key: str):
        if key == "decision":
            return self.action
        return getattr(self, key)

    def to_dict(self) -> dict:
        return {
            "action": self.action,
            "confidence": round(float(self.confidence), 6),
            "reason_code": self.reason_code,
            "model": self.model,
            "latency_ms": round(float(self.latency_ms), 3),
        }


class SLMDecisionModel:
    """A small language model used as a decision model.

    The wrapper does three things the raw model does not: it constrains the output
    to the spec's action set, it scores each action so the cost matrix can be
    applied, and it fails closed to the abstain action on anything unexpected.
    """

    def __init__(self, model=None, tokenizer=None, actions: Optional[Sequence[str]] = None,
                 version: str = "unknown", abstain: Optional[str] = None) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.actions = list(actions or [])
        self.version = version
        self.abstain = abstain

    # ------------------------------------------------------------------ loading
    @classmethod
    def from_pretrained(cls, checkpoint: str, device: str = "cpu",
                        torch_dtype: Optional[str] = None, actions: Optional[Sequence[str]] = None,
                        **kwargs: Any) -> "SLMDecisionModel":
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - depends on the environment
            raise MissingDependency(
                "SLMDecisionModel needs torch and transformers; install the extra with "
                "`pip install -e 'code/[ml]'`"
            ) from exc

        dtype = None
        if torch_dtype in ("int8", "8bit"):
            dtype = torch.int8
        elif torch_dtype in ("float16", "fp16", "half"):
            dtype = torch.float16
        elif torch_dtype in ("bfloat16", "bf16"):
            dtype = torch.bfloat16

        tokenizer = AutoTokenizer.from_pretrained(checkpoint, **kwargs)
        load_kwargs = dict(kwargs)
        if dtype is not None:
            load_kwargs["torch_dtype"] = dtype
        model = AutoModelForCausalLM.from_pretrained(checkpoint, **load_kwargs)
        model.eval()
        model.to(device)
        return cls(model=model, tokenizer=tokenizer, actions=actions,
                   version=checkpoint.rsplit("/", 1)[-1], abstain=kwargs.get("abstain"))

    # ------------------------------------------------------------------ inference
    def _score_actions(self, prompt: str, max_len: int = 1024) -> Dict[str, float]:
        """Probability of each action's first token, renormalised over the action set."""
        import torch

        ids = self.tokenizer(prompt, return_tensors="pt", truncation=True,
                             max_length=max_len).input_ids
        with torch.no_grad():
            logits = self.model(ids).logits[0, -1]
        probs = torch.softmax(logits.float(), dim=-1)
        raw = {}
        for a in self.actions:
            toks = self.tokenizer.encode(" " + a, add_special_tokens=False)
            raw[a] = float(probs[toks[0]]) if toks else 0.0
        total = sum(raw.values()) or 1.0
        return {a: v / total for a, v in raw.items()}

    def decide(self, text: str, prompt_template: Optional[str] = None,
               max_len: int = 1024) -> DecisionResult:
        start = time.perf_counter()
        prompt = (prompt_template or "Choose one of {actions}.\n\nInput: {text}\nAnswer:").format(
            actions=", ".join(self.actions) or "the available actions", text=text)
        if not text or not text.strip():
            return DecisionResult(action=self.abstain or self.actions[0], confidence=0.0,
                                  reason_code="EMPTY", model=self.version)
        try:
            probs = self._score_actions(prompt, max_len=max_len)
        except MissingDependency:
            raise
        except Exception as exc:  # pragma: no cover - defensive
            return DecisionResult(action=self.abstain or self.actions[0], confidence=0.0,
                                  reason_code="INFER_ERROR:%s" % type(exc).__name__,
                                  model=self.version)
        action = max(probs, key=probs.get) if probs else (self.abstain or self.actions[0])
        return DecisionResult(
            action=action,
            confidence=probs.get(action, 0.0),
            probs=probs,
            raw=action,
            latency_ms=(time.perf_counter() - start) * 1000.0,
            model=self.version,
        )

    def decide_with_context(self, query: str, context: Dict[str, Any],
                            max_len: int = 1024) -> DecisionResult:
        """Decide with options and constraints supplied as structured context."""
        options = context.get("options") or self.actions
        constraints = context.get("constraints") or []
        template = ("Options: {options}\nConstraints: {constraints}\n\nInput: {text}\n"
                    "Best option:")
        return self.decide(
            query,
            prompt_template=template.format(options=", ".join(options),
                                            constraints="; ".join(constraints) or "none",
                                            text="{text}"),
            max_len=max_len,
        )

    def as_policy(self, spec: DecisionSpec, prompt: Optional[str] = None):
        """Adapt to the Policy interface so it can be evaluated like any other policy."""
        from .policies import Policy

        outer = self
        self.actions = list(spec.actions)
        self.abstain = spec.abstain

        class _SLMPolicy(Policy):
            def act(self, signal) -> str:
                return outer.decide(str(signal), prompt_template=prompt).action

        return _SLMPolicy(spec)

    def warmup(self, n: int = 5) -> None:
        """Pay allocator and cache costs before real traffic arrives."""
        for _ in range(n):
            try:
                self.decide("warmup")
            except Exception:  # pragma: no cover
                break


class DistilledDecisionModel(SLMDecisionModel):
    """A student model trained on a teacher's output distribution."""

    def __init__(self, teacher: Optional[str] = None, temperature: float = 4.0, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.teacher = teacher
        self.temperature = temperature


class FineTunedDecisionModel(SLMDecisionModel):
    """A model tuned on gold labels, optionally with a LoRA adapter."""

    def __init__(self, adapter_path: Optional[str] = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.adapter_path = adapter_path


class Retriever:
    """Minimal lexical retriever (Jaccard over tokens).

    Good enough to demonstrate the hybrid pattern and to run offline. Swap in an
    embedding index for production; the interface is what matters.
    """

    def __init__(self, documents: Sequence[str], top_k: int = 3, index: str = "default",
                 embed_model: Optional[str] = None) -> None:
        self.documents = list(documents)
        self.top_k = int(top_k)
        self.index = index
        self.embed_model = embed_model
        self._sets = [set(d.lower().split()) for d in self.documents]

    def search(self, query: str) -> List[str]:
        q = set(query.lower().split())
        scored = []
        for doc, s in zip(self.documents, self._sets):
            inter = len(q & s)
            union = len(q | s) or 1
            scored.append((inter / union, doc))
        scored.sort(key=lambda t: -t[0])
        return [d for score, d in scored[: self.top_k] if score > 0]


class HybridPolicy:
    """Retrieval-augmented policy: fetch context, then let the SLM decide."""

    def __init__(self, spec: DecisionSpec, slm: SLMDecisionModel, retriever: Retriever,
                 template: str = "Context:\n{context}\n\nCase:\n{query}\n\nAction?",
                 max_context_tokens: int = 700) -> None:
        self.spec = spec
        self.slm = slm
        self.retriever = retriever
        self.template = template
        self.max_context_tokens = max_context_tokens

    def decide(self, query: str) -> DecisionResult:
        docs = self.retriever.search(query)
        context = "\n".join(docs)[: self.max_context_tokens * 4]  # ~4 chars per token
        prompt = self.template.format(context=context or "(no context found)", query=query)
        return self.slm.decide(query, prompt_template=prompt)

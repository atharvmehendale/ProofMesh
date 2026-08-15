"""
schemas.py

Single source of truth for the data shapes that flow between pipeline
stages. app.py, core/sympy_verifier.py, models/featherless_client.py, and
models/prompts.py should all import from here rather than re-defining
these shapes locally - that's the whole point of pulling this into its
own file instead of letting each module assume its own dict layout.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Literal


# ---------------------------------------------------------------------------
# What the extraction model must emit, per derivation step.
# prompts.py's extraction prompt should be written to produce exactly this
# shape (as JSON) - see EXTRACTION_OUTPUT_EXAMPLE below.
# ---------------------------------------------------------------------------

@dataclass
class ExtractedStep:
    index: int
    latex: str                 # display-only, never parsed by sympy_verifier
    expr: Optional[str]        # constrained plain-text expr, e.g. "x**2 + 2*x + 1"
    raw_source_text: str = ""  # original snippet from the PDF/input, for traceability

    def to_dict(self) -> dict:
        return {"index": self.index, "latex": self.latex, "expr": self.expr}


# One canonical example, importable so prompts.py can show it to the model
# and tests can validate against it - keeps the "contract" in one place.
EXTRACTION_OUTPUT_EXAMPLE = [
    {"index": 0, "latex": "(x+1)^2", "expr": "(x+1)**2"},
    {"index": 1, "latex": "x^2 + 2x + 1", "expr": "x**2 + 2*x + 1"},
]


# ---------------------------------------------------------------------------
# What sympy_verifier.py produces per step-to-step transition.
# Mirrors core/sympy_verifier.py's StepResult - kept here too so app.py and
# featherless_client.py don't need to import sympy_verifier just to know the
# shape of a result.
# ---------------------------------------------------------------------------

VerifyStatus = Literal["valid", "discrepancy", "unverifiable"]


@dataclass
class VerifiedStep:
    index: int
    status: VerifyStatus
    detail: str = ""
    parsed_via: str = ""
    math_model_opinion: Optional["MathModelOpinion"] = None  # filled in only if escalated

    def needs_math_model_check(self) -> bool:
        # Escalation rule referenced in models/featherless_client.py:
        # only steps SymPy itself couldn't cleanly resolve get a second
        # opinion from a specialized math model - not every step.
        return self.status == "unverifiable"


# ---------------------------------------------------------------------------
# What the specialized math model (Qwen2.5-Math) returns when a step is
# escalated to it. Kept separate from VerifiedStep so "SymPy said" and
# "the model guessed" are never conflated in the data itself.
# ---------------------------------------------------------------------------

@dataclass
class MathModelOpinion:
    step_index: int
    model_id: str               # e.g. "Qwen/Qwen2.5-Math-72B-Instruct"
    verdict: Literal["valid", "invalid", "cannot_determine"]
    explanation: str = ""


# ---------------------------------------------------------------------------
# What the Judge model receives and returns. It only ever sees the *flagged*
# steps (see core/sympy_verifier.discrepancies_for_judge) - never the full
# derivation - to keep its prompt small and its job narrow: explain, don't
# re-verify.
# ---------------------------------------------------------------------------

@dataclass
class JudgeInput:
    flagged_steps: list[dict]         # from discrepancies_for_judge()
    math_model_opinions: list[dict]   # from any escalated steps, or []


@dataclass
class JudgeVerdict:
    step_index: int
    is_error: bool
    plain_language_explanation: str
    confidence_note: str = ""  # e.g. "SymPy-confirmed" vs "flagged by math model, SymPy inconclusive"


# ---------------------------------------------------------------------------
# Top-level result returned by app.py to the UI - the thing that actually
# gets rendered.
# ---------------------------------------------------------------------------

@dataclass
class ProofAuditResult:
    source_filename: str
    total_steps: int
    verified_steps: list[VerifiedStep] = field(default_factory=list)
    judge_verdicts: list[JudgeVerdict] = field(default_factory=list)

    @property
    def has_errors(self) -> bool:
        return any(v.is_error for v in self.judge_verdicts)

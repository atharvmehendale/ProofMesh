"""
core/sympy_verifier.py

Neuro-symbolic verification layer for ProofMesh.

DESIGN CHOICE (read this before wiring it up):
We do NOT parse arbitrary freeform LaTeX straight into SymPy. Raw LaTeX from
real derivations (\\sum, \\int, aligned/multi-line environments, custom
macros) breaks naive LaTeX->SymPy parsers constantly - that's the failure
mode most likely to sink this feature during the demo.

Instead, the extraction LLM (see models/prompts.py) is prompted to emit each
derivation step as structured JSON with TWO fields:
  - "latex": the original/display form - shown in the UI, never parsed
  - "expr" : a constrained, SymPy-parseable plain-text expression
             (standard operators, ** for powers, no LaTeX commands)

This file only ever parses "expr" on the primary path. latex2sympy2 is kept
as a single best-effort fallback for steps where "expr" is missing or fails
to parse - never as the primary path. A step that can't be parsed either way
is marked "unverifiable", not silently dropped and not treated as an error -
that distinction matters for the Judge LLM and for the UI.

Example extraction JSON this expects per step:
    {"latex": "x^2 + 2x + 1", "expr": "x**2 + 2*x + 1"}
    {"latex": "(x+1)^2 = 0",  "expr": "Eq((x+1)**2, 0)"}
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import sympy
from sympy import Eq, simplify
from sympy.core.sympify import SympifyError
from sympy.parsing.sympy_parser import (
    parse_expr,
    standard_transformations,
    implicit_multiplication_application,
    convert_xor,
)

from schemas import VerifiedStep

try:
    from latex2sympy2 import latex2sympy
except ImportError:
    latex2sympy = None


_TRANSFORMS = standard_transformations + (
    implicit_multiplication_application,
    convert_xor,
)


@dataclass
class _ParseOutcome:
    """Internal-only: diagnostics for a single step's parse attempt (not a
    comparison result, so it's kept separate from schemas.VerifiedStep,
    which represents a step-to-step comparison, not a parse). Used inside
    this module by parse_step()/verify_derivation() only - nothing else
    should need to import this."""
    index: int
    latex: str
    expr_text: Optional[str]
    parsed_ok: bool
    detail: str = ""
    parsed_via: str = ""  # "expr" | "latex_fallback" | ""


def _safe_parse(expr_text: str):
    """Parse a constrained plain-text expression into a SymPy object.
    Supports bare expressions ('x**2 + 1') and equations via Eq(...)."""
    local_dict = {"Eq": Eq}
    return parse_expr(expr_text, local_dict=local_dict, transformations=_TRANSFORMS, evaluate=True)


def _fallback_latex_parse(latex_str: str):
    """Best-effort parse of raw LaTeX. Only used when 'expr' is unusable.

    IMPORTANT: latex2sympy2 does not raise an error on LaTeX commands it
    doesn't recognize (\\nabla, \\vec{F}, custom macros, etc.) - it silently
    creates a Symbol literally named "\\nabla" or "\\vec{F}" and proceeds as
    if that were an ordinary variable. That means a true identity involving
    such notation can come back as a false 'discrepancy', and an actually-
    wrong step can come back as a false 'valid' - both worse than an honest
    'unverifiable', since they're confidently wrong rather than admitting
    uncertainty. This was caught by testing, not assumed: \\nabla \\times
    (\\nabla \\times \\vec{F}) parsed as ordinary symbols and a genuine
    vector calculus identity came back flagged as an error.

    The guard below rejects any fallback parse whose free symbols contain
    a backslash or brace - a real, correctly-parsed symbol never looks
    like that, so its presence means latex2sympy2 gave up silently rather
    than actually understanding the input. This does NOT fully solve
    notation the fallback mishandles without leaving a leftover LaTEX
    command (e.g. it also silently dropped a transpose superscript in
    testing without leaving any trace to detect) - matrix/vector/tensor
    notation should be treated as genuinely out of scope, not just
    guarded against, until this fallback is hardened further."""
    if latex2sympy is None or not latex_str:
        return None
    try:
        result = latex2sympy(latex_str)
        bad_symbols = [s for s in result.free_symbols if "\\" in str(s) or "{" in str(s)]
        if bad_symbols:
            return None  # garbage parse - treat exactly like a failed parse
        return result
    except Exception:
        return None


def parse_step(index: int, latex: str, expr_text: Optional[str]):
    """Parse one derivation step. Tries the constrained 'expr' first, then
    falls back to raw LaTeX parsing if that's all we have. Returns
    (parsed_sympy_object_or_None, _ParseOutcome)."""
    detail = ""
    if expr_text:
        try:
            parsed = _safe_parse(expr_text)
            return parsed, _ParseOutcome(index, latex, expr_text, True, parsed_via="expr")
        except (SympifyError, SyntaxError, TypeError, ValueError) as e:
            detail = f"expr parse failed: {e}"
    else:
        detail = "no expr field provided by extraction step"

    fallback = _fallback_latex_parse(latex)
    if fallback is not None:
        return fallback, _ParseOutcome(index, latex, expr_text, True, detail, "latex_fallback")

    return None, _ParseOutcome(index, latex, expr_text, False, detail)


def verify_step_pair(prev_index: int, prev_expr, curr_index: int, curr_expr) -> VerifiedStep:
    """Checks whether curr_expr is algebraically consistent with prev_expr,
    i.e. whether the difference simplifies to zero. Handles both bare
    expressions and full equations (moves lhs-rhs to one side first).

    Known limitation: sympy.simplify() won't catch every true equivalence
    (e.g. some trig/log identities need targeted rewrites). For a hackathon
    demo, flag anything simplify() can't reduce to zero as 'discrepancy' and
    let the Judge LLM add the caveat that this is a symbolic-simplification
    check, not a full theorem prover - don't oversell it as infallible.
    """
    try:
        if isinstance(prev_expr, Eq) and isinstance(curr_expr, Eq):
            diff = simplify((prev_expr.lhs - prev_expr.rhs) - (curr_expr.lhs - curr_expr.rhs))
        else:
            diff = simplify(prev_expr - curr_expr)

        is_zero = (diff == 0)
        status = "valid" if is_zero else "discrepancy"
        detail = "" if is_zero else (
            f"step {prev_index}->{curr_index}: difference simplifies to {diff}, expected 0"
        )
        return VerifiedStep(index=curr_index, status=status, detail=detail)
    except Exception as e:
        return VerifiedStep(index=curr_index, status="unverifiable", detail=f"comparison failed: {e}")


def verify_derivation(steps: list[dict]) -> list[VerifiedStep]:
    """
    steps: list of {"latex": str, "expr": str | None}, in derivation order,
    as emitted by the extraction LLM (schemas.ExtractedStep.to_dict()).

    Returns one schemas.VerifiedStep per step-to-step transition
    (len(steps) - 1 results), each flagging whether that step is
    algebraically consistent with the one before it. Steps that fail to
    parse are 'unverifiable', not silently skipped - only a SymPy-confirmed
    non-zero difference is a 'discrepancy'. Keeping these separate matters:
    a bad extraction and a real math error should never look the same to
    the user. Any step marked 'unverifiable' here is exactly the set that
    VerifiedStep.needs_math_model_check() will flag for escalation to the
    specialized math model in models/featherless_client.py.
    """
    parsed: list = []
    parse_outcomes: list[_ParseOutcome] = []
    for i, step in enumerate(steps):
        expr, outcome = parse_step(i, step.get("latex", ""), step.get("expr"))
        parsed.append(expr)
        parse_outcomes.append(outcome)

    comparisons: list[VerifiedStep] = []
    for i in range(1, len(steps)):
        prev_expr, curr_expr = parsed[i - 1], parsed[i]
        if prev_expr is None or curr_expr is None:
            bad = parse_outcomes[i] if parsed[i] is None else parse_outcomes[i - 1]
            comparisons.append(VerifiedStep(
                index=i, status="unverifiable",
                detail=f"step {bad.index} failed to parse: {bad.detail}",
                parsed_via=bad.parsed_via,
            ))
            continue
        comparisons.append(verify_step_pair(i - 1, prev_expr, i, curr_expr))

    return comparisons


def discrepancies_for_judge(comparisons: list[VerifiedStep], extracted_steps: list[dict]) -> list[dict]:
    """Filter down to just what the Judge LLM needs to explain in plain
    language, INCLUDING the original LaTeX for the flagged step and the
    one before it - not just SymPy's technical detail string. Without the
    LaTeX, the judge model has no way to notice that a flagged discrepancy
    might stem from ambiguous notation (e.g. a missing parenthesis) rather
    than a genuine derivation error - it can't reason about something it
    never sees. extracted_steps is the original list from run_extraction,
    used here purely to look up latex by index."""
    result = []
    for r in comparisons:
        if r.status not in ("discrepancy", "unverifiable"):
            continue
        prev_latex = extracted_steps[r.index - 1].get("latex", "") if r.index > 0 else ""
        curr_latex = extracted_steps[r.index].get("latex", "") if r.index < len(extracted_steps) else ""
        result.append({
            "step_index": r.index,
            "status": r.status,
            "detail": r.detail,
            "previous_step_latex": prev_latex,
            "current_step_latex": curr_latex,
        })
    return result


if __name__ == "__main__":
    # quick smoke test - a correct step and a broken one
    demo_steps = [
        {"latex": "(x+1)^2", "expr": "(x+1)**2"},
        {"latex": "x^2 + 2x + 1", "expr": "x**2 + 2*x + 1"},   # valid expansion
        {"latex": "x^2 + 2x + 2", "expr": "x**2 + 2*x + 2"},   # inserted error
    ]
    results = verify_derivation(demo_steps)
    for r in results:
        print(r.index, r.status, r.detail, "| needs escalation:", r.needs_math_model_check())
    print("For judge:", discrepancies_for_judge(results))

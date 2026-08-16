"""
app.py

ProofMesh entrypoint - this is the file Streamlit Cloud points at
(MAIN_FILE = "app.py"). Thin by design: it wires the pipeline stages
together and renders results. All the actual logic lives in
core/sympy_verifier.py, core/pdf_parser.py, models/featherless_client.py,
models/prompts.py, and schemas.py - this file should stay
import-and-orchestrate, not grow business logic of its own.

Two input paths, both feeding the same run_pipeline(): paste derivation
text directly, or upload a PDF (core/pdf_parser.get_derivation_text
handles text-layer extraction with vision-OCR fallback for scanned pages).
"""

import os

import streamlit as st

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from schemas import ProofAuditResult
from sympy_verifier import verify_derivation, discrepancies_for_judge
from featherless_client import run_extraction, run_math_check, run_judge, ModelOutputError
from pdf_parser import get_derivation_text

_SECRET_KEYS = ["PROVIDER", "GROQ_API_KEY", "FEATHERLESS_API_KEY"]


def get_secrets() -> dict:
    """Merges st.secrets (used once deployed on Streamlit Cloud) with
    plain environment variables / a local .env file (used for local
    testing without any secrets.toml at all). st.secrets wins on conflict,
    so a deployed app's dashboard settings always take priority - this is
    purely about not requiring a specific mechanism locally."""
    merged = {k: os.getenv(k) for k in _SECRET_KEYS if os.getenv(k)}
    try:
        merged.update({k: v for k, v in dict(st.secrets).items()})
    except Exception:
        pass
    return merged


def run_pipeline(extracted: list[dict], secrets: dict, source_filename: str = "pasted_text") -> ProofAuditResult:
    """The verify -> escalate -> judge loop, given steps that have ALREADY
    been extracted. Deliberately does not call run_extraction itself - the
    caller extracts once and passes the result in, so this function can't
    accidentally double a Featherless/Groq call. Kept separate from
    Streamlit widget calls so it's testable without a running app: pass in
    a hardcoded extracted list and no network call happens until the
    escalation/judge stages, which you can also mock in tests.
    """
    verified = verify_derivation(extracted)

    # Escalate ONLY steps SymPy couldn't resolve - see
    # schemas.VerifiedStep.needs_math_model_check() and the design note in
    # core/sympy_verifier.py. This is the one place the pipeline calls out
    # to a second model, and only when it's actually needed.
    math_opinions_for_judge = []
    for v in verified:
        if v.needs_math_model_check() and v.index > 0:
            prev_step = extracted[v.index - 1]
            curr_step = extracted[v.index]
            opinion = run_math_check(
                secrets, v.index,
                prev_step.get("latex", ""), curr_step.get("latex", ""),
            )
            v.math_model_opinion = opinion
            math_opinions_for_judge.append({
                "step_index": opinion.step_index,
                "model_id": opinion.model_id,
                "verdict": opinion.verdict,
                "explanation": opinion.explanation,
            })

    flagged = discrepancies_for_judge(verified, extracted)
    judge_verdicts = run_judge(secrets, flagged, math_opinions_for_judge) if flagged else []

    return ProofAuditResult(
        source_filename=source_filename,
        total_steps=len(extracted),
        verified_steps=verified,
        judge_verdicts=judge_verdicts,
    )


def render_result(result: ProofAuditResult, extracted_steps: list[dict]) -> None:
    """All display logic lives here, not scattered through main().

    Each step's equation and its verdict render as ONE colored block, not
    plain LaTeX followed by a separate message underneath it. That's
    deliberate, not cosmetic: a flagged equation that renders identically
    to a correct one - with the only signal being text below it - relies
    entirely on someone reading past the equation to find out something's
    wrong. Someone skimming, or who stops reading once they recognize
    their own equation, gets no warning at all. The equation itself has to
    carry the signal, which means it lives inside the colored container,
    not next to it."""
    verdict_by_index = {v.step_index: v for v in result.judge_verdicts}

    if not result.has_errors and not any(
        v.status == "unverifiable" for v in result.verified_steps
    ):
        st.success(f"All {result.total_steps} steps check out. No discrepancies found.")

    unresolved_since = None  # earliest step index still carrying a flagged, uncorrected issue

    for step in extracted_steps:
        idx = step["index"]
        latex_block = f"$${step['latex']}$$"

        matching = next((v for v in result.verified_steps if v.index == idx), None)
        if matching is None:
            st.markdown(latex_block)
            st.caption(f"Step {idx}: starting point - not checked against anything before it.")
            continue

        jv = verdict_by_index.get(idx)

        if matching.status == "valid":
            if unresolved_since is not None:
                # Locally consistent with the step right before it - but
                # that step traces back to a flagged, never-corrected
                # issue. Rendering this as plain green would read as "the
                # problem's gone," which isn't true - it's just that this
                # specific step didn't introduce a NEW error on top of the
                # old one.
                st.info(
                    f"{latex_block}\n\nStep {idx}: follows correctly from step {idx - 1}, "
                    f"but still carries the unresolved issue flagged at step {unresolved_since}."
                )
            else:
                st.success(f"{latex_block}\n\nStep {idx}: consistent with the previous step.")
        elif matching.status == "discrepancy":
            explanation = jv.plain_language_explanation if jv else matching.detail
            st.error(f"{latex_block}\n\nStep {idx}: {explanation}")
            if unresolved_since is None:
                unresolved_since = idx
        else:  # unverifiable
            explanation = jv.plain_language_explanation if jv else matching.detail
            st.warning(f"{latex_block}\n\nStep {idx}: could not be fully verified - {explanation}")
            if unresolved_since is None:
                unresolved_since = idx


def main() -> None:
    st.set_page_config(page_title="ProofMesh", page_icon="\u2211")
    st.title("ProofMesh")
    st.caption("Paste a multi-step derivation. Each step is checked against the one before it.")

    secrets = get_secrets()
    active_provider = secrets.get('PROVIDER', 'groq')
    st.caption(f"Provider: {active_provider} | Key found: {bool(secrets.get(f'{active_provider.upper()}_API_KEY'))}")
    if not secrets.get(f"{active_provider.upper()}_API_KEY"):
        st.error(
            "No API key configured. Set PROVIDER and the matching "
            "*_API_KEY in Streamlit secrets before running."
        )
        st.stop()

    input_mode = st.radio("Input", ["Paste text", "Upload PDF"], horizontal=True, label_visibility="collapsed")

    source_text = ""
    source_filename = "pasted_text"

    if input_mode == "Paste text":
        source_text = st.text_area(
            "Derivation", height=200,
            placeholder="e.g. (x+1)^2 = x^2 + 2x + 1 = x^2 + 2x + 2   <- has a seeded error",
        )
        run_clicked = st.button("Run audit", type="primary") and bool(source_text.strip())

    else:
        uploaded = st.file_uploader("PDF", type="pdf", label_visibility="collapsed")
        run_clicked = False
        if uploaded is not None and st.button("Run audit", type="primary"):
            source_filename = uploaded.name
            with st.spinner("Reading PDF..."):
                try:
                    source_text, info = get_derivation_text(secrets, uploaded)
                except RuntimeError as e:
                    st.error(str(e))
                    st.stop()
            if info["method"] == "ocr":
                st.info(f"No text layer found - ran OCR on {info['pages_read']} of {info['total_pages']} pages.")
            run_clicked = bool(source_text.strip())

    if run_clicked:
        with st.spinner("Extracting steps..."):
            try:
                extracted = run_extraction(secrets, source_text)
            except ModelOutputError as e:
                st.error("The extraction model's response couldn't be parsed. Raw output below:")
                st.code(e.raw_text)
                st.stop()

        if len(extracted) < 2:
            st.warning("Need at least two steps to check consistency between them.")
            st.stop()

        with st.spinner("Verifying..."):
            result = run_pipeline(extracted, secrets, source_filename)

        render_result(result, extracted)


if __name__ == "__main__":
    main()

"""
app.py

ProofMesh entrypoint - this is the file Streamlit Cloud points at
(MAIN_FILE = "app.py"). Thin by design: it wires the pipeline stages
together and renders results. All the actual logic lives in
core/sympy_verifier.py, core/pdf_parser.py, models/featherless_client.py,
models/prompts.py, and schemas.py - this file should stay
import-and-orchestrate, not grow business logic of its own.
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
    """Merges st.secrets with environment variables / .env."""
    merged = {k: os.getenv(k) for k in _SECRET_KEYS if os.getenv(k)}
    try:
        merged.update({k: v for k, v in dict(st.secrets).items()})
    except Exception:
        pass
    return merged


def run_pipeline(extracted: list[dict], secrets: dict, source_filename: str = "pasted_text") -> ProofAuditResult:
    """The verify -> escalate -> judge loop."""
    verified = verify_derivation(extracted)

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


def inject_css() -> None:
    """Light, restrained styling."""
    st.markdown("""
    <style>
    /* Overall spacing and typography */
    .block-container {
        padding-top: 1.8rem;
        padding-bottom: 3rem;
        max-width: 820px;
    }
    h1 {
        font-weight: 600;
        letter-spacing: -0.02em;
        margin-bottom: 0.15rem !important;
    }
    .stCaption {
        margin-bottom: 1.4rem;
    }

    /* Status boxes – slightly softer, more card-like */
    div[data-testid="stAlert"] {
        border-radius: 8px;
        padding: 0.9rem 1.1rem;
        border-left-width: 5px;
    }
    div[data-testid="stAlert"] p {
        margin-bottom: 0.35rem;
    }

    /* Summary chips */
    .summary-row {
        display: flex;
        flex-wrap: wrap;
        gap: 0.55rem;
        margin: 0.8rem 0 1.4rem 0;
    }
    .summary-chip {
        background: #f4f6f8;
        border: 1px solid #e2e6ea;
        border-radius: 999px;
        padding: 0.28rem 0.85rem;
        font-size: 0.88rem;
        color: #334155;
    }
    .summary-chip.error {
        background: #fef2f2;
        border-color: #fecaca;
        color: #991b1b;
    }
    .summary-chip.warn {
        background: #fffbeb;
        border-color: #fde68a;
        color: #92400e;
    }
    .summary-chip.ok {
        background: #f0fdf4;
        border-color: #bbf7d0;
        color: #166534;
    }

    /* Subtle section labels */
    .section-label {
        font-size: 0.78rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        color: #64748b;
        margin: 1.6rem 0 0.5rem 0;
    }

    /* Darker helper text inside sample expander */
    .expected-result {
        color: #334155;
        font-size: 0.92rem;
        margin-top: 0.15rem;
        margin-bottom: 0.6rem;
    }
    </style>
    """, unsafe_allow_html=True)


def render_summary(result: ProofAuditResult) -> None:
    """Compact status strip above the step list."""
    n_disc = sum(1 for v in result.verified_steps if v.status == "discrepancy")
    n_unv = sum(1 for v in result.verified_steps if v.status == "unverifiable")
    n_errors = sum(1 for v in result.judge_verdicts if v.is_error)

    chips = [f'<span class="summary-chip">{result.total_steps} steps</span>']

    if n_errors > 0:
        chips.append(f'<span class="summary-chip error">{n_errors} error{"s" if n_errors != 1 else ""}</span>')
    if n_disc > 0 and n_errors == 0:
        chips.append(f'<span class="summary-chip error">{n_disc} discrepancy{"ies" if n_disc != 1 else ""}</span>')
    if n_unv > 0:
        chips.append(f'<span class="summary-chip warn">{n_unv} unverifiable</span>')
    if n_errors == 0 and n_disc == 0 and n_unv == 0:
        chips.append('<span class="summary-chip ok">All consistent</span>')

    st.markdown(f'<div class="summary-row">{"".join(chips)}</div>', unsafe_allow_html=True)


def render_result(result: ProofAuditResult, extracted_steps: list[dict]) -> None:
    """All display logic lives here.

    Each step's equation and its verdict render as ONE colored block.
    The equation itself carries the signal.
    """
    verdict_by_index = {v.step_index: v for v in result.judge_verdicts}

    render_summary(result)

    if not result.has_errors and not any(v.status == "unverifiable" for v in result.verified_steps):
        st.success(f"All {result.total_steps} steps check out. No discrepancies found.")

    unresolved_since = None

    for step in extracted_steps:
        idx = step["index"]
        latex_block = f"$${step['latex']}$$"

        matching = next((v for v in result.verified_steps if v.index == idx), None)
        if matching is None:
            st.markdown(latex_block)
            st.caption(f"Step {idx} · starting point — not checked against anything before it.")
            continue

        jv = verdict_by_index.get(idx)

        if matching.status == "valid":
            if unresolved_since is not None:
                st.info(
                    f"{latex_block}\n\n"
                    f"**Step {idx}** · follows correctly from the previous step, "
                    f"but still carries the unresolved issue from step {unresolved_since}."
                )
            else:
                st.success(
                    f"{latex_block}\n\n"
                    f"**Step {idx}** · consistent with the previous step."
                )
        elif matching.status == "discrepancy":
            explanation = jv.plain_language_explanation if jv else matching.detail
            st.error(
                f"{latex_block}\n\n"
                f"**Step {idx}** · {explanation}"
            )
            if unresolved_since is None:
                unresolved_since = idx
        else:  # unverifiable
            explanation = jv.plain_language_explanation if jv else matching.detail
            st.warning(
                f"{latex_block}\n\n"
                f"**Step {idx}** · could not be fully verified — {explanation}"
            )
            if unresolved_since is None:
                unresolved_since = idx


def main() -> None:
    st.set_page_config(
        page_title="ProofMesh",
        page_icon="∑",
        layout="centered",
        initial_sidebar_state="collapsed",
    )
    inject_css()

    # ── Header ──────────────────────────────────────────────
    st.title("ProofMesh")
    st.caption("Check multi-step derivations. Each step is verified against the one before it.")

    with st.expander("How it works", expanded=False):
        st.markdown(
            """
            1. **Extract** — an LLM turns the text (or PDF) into ordered steps with both display LaTeX and a SymPy-friendly expression.  
            2. **Verify** — SymPy checks algebraic consistency between consecutive steps.  
            3. **Escalate** — only steps SymPy cannot resolve are sent to a specialized math model.  
            4. **Explain** — a judge model turns technical flags into plain-language notes.

            The equation itself is colored so you can see the status without reading past it.
            """
        )

    secrets = get_secrets()
    if not secrets.get(f"{secrets.get('PROVIDER', 'groq').upper()}_API_KEY"):
        st.error(
            "No API key configured. Set `PROVIDER` and the matching "
            "`*_API_KEY` in Streamlit secrets before running."
        )
        st.stop()

    # ── Input ───────────────────────────────────────────────
    st.markdown('<div class="section-label">Input</div>', unsafe_allow_html=True)

    input_mode = st.radio(
        "Input method",
        ["Paste text", "Upload PDF"],
        horizontal=True,
        label_visibility="collapsed",
    )

    source_text = ""
    source_filename = "pasted_text"
    run_clicked = False

    if input_mode == "Paste text":
        # Primary action stays high on the page
        source_text = st.text_area(
            "Derivation",
            height=180,
            placeholder="Paste a multi-step derivation here…\n\ne.g. (x+1)^2 = x^2 + 2x + 1 = x^2 + 2x + 2",
            key="derivation_text",
            label_visibility="collapsed",
        )
        run_clicked = st.button("Run audit", type="primary") and bool(source_text.strip())

        # Samples are optional help, placed below so they never push the Run button away
        with st.expander("Try a sample derivation", expanded=False):
            st.markdown(
                "These short examples show what ProofMesh does. "
                "Click **Load this example**, then press **Run audit** above."
            )

            st.markdown("**1. Correct expansion**")
            st.code("(x+1)^2 = x^2 + 2x + 1", language=None)
            st.markdown(
                '<p class="expected-result">Expected result: every step is marked consistent (green).</p>',
                unsafe_allow_html=True,
            )
            if st.button("Load this example", key="ex_correct", use_container_width=True):
                st.session_state["derivation_text"] = "(x+1)^2 = x^2 + 2x + 1"
                st.rerun()

            st.markdown("---")
            st.markdown("**2. Seeded error**")
            st.code("(x+1)^2 = x^2 + 2x + 1 = x^2 + 2x + 2", language=None)
            st.markdown(
                '<p class="expected-result">Expected result: the final step is flagged as a discrepancy (red).</p>',
                unsafe_allow_html=True,
            )
            if st.button("Load this example", key="ex_error", use_container_width=True):
                st.session_state["derivation_text"] = "(x+1)^2 = x^2 + 2x + 1 = x^2 + 2x + 2"
                st.rerun()

            st.markdown("---")
            st.markdown("**3. Chained steps**")
            st.code("x^2 - 4 = (x-2)(x+2) = x(x+2) - 2(x+2)", language=None)
            st.markdown(
                '<p class="expected-result">Expected result: all steps consistent.</p>',
                unsafe_allow_html=True,
            )
            if st.button("Load this example", key="ex_chain", use_container_width=True):
                st.session_state["derivation_text"] = "x^2 - 4 = (x-2)(x+2) = x(x+2) - 2(x+2)"
                st.rerun()

    else:
        uploaded = st.file_uploader(
            "PDF",
            type="pdf",
            label_visibility="collapsed",
            help="Text-layer PDFs are preferred. Scanned pages fall back to vision OCR (slower, uses credits).",
        )
        if uploaded is not None and st.button("Run audit", type="primary"):
            source_filename = uploaded.name
            with st.spinner("Reading PDF…"):
                try:
                    source_text, info = get_derivation_text(secrets, uploaded)
                except RuntimeError as e:
                    st.error(str(e))
                    st.stop()
            if info["method"] == "ocr":
                st.info(
                    f"No text layer found — ran OCR on {info['pages_read']} of "
                    f"{info['total_pages']} pages."
                )
            run_clicked = bool(source_text.strip())

    # ── Pipeline ────────────────────────────────────────────
    if run_clicked:
        with st.spinner("Extracting steps…"):
            try:
                extracted = run_extraction(secrets, source_text)
            except ModelOutputError as e:
                st.error("The extraction model’s response could not be parsed. Raw output below:")
                st.code(e.raw_text)
                st.stop()

        if len(extracted) < 2:
            st.warning("Need at least two steps to check consistency between them.")
            st.stop()

        with st.spinner("Verifying…"):
            result = run_pipeline(extracted, secrets, source_filename)

        st.markdown('<div class="section-label">Results</div>', unsafe_allow_html=True)
        render_result(result, extracted)


if __name__ == "__main__":
    main()

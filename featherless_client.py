"""
models/featherless_client.py

Featherless-only client for all three LLM stages (extraction, math-check
escalation, judge) plus OCR.

HISTORY: this was originally provider-agnostic (Groq + Featherless), used
during development to test the pipeline shape before Featherless credits
were available. Groq has been fully removed now that Featherless is
confirmed working - keeping dead provider-switching logic around after a
submission is more confusing than useful, not safer.
"""

from __future__ import annotations

import ast
import json
import re
from typing import Optional

from openai import OpenAI, APIStatusError, APIError, RateLimitError

from schemas import MathModelOpinion, JudgeVerdict
from prompts import (
    EXTRACTION_SYSTEM_PROMPT,
    MATH_CHECK_SYSTEM_PROMPT,
    JUDGE_SYSTEM_PROMPT,
)


FEATHERLESS_BASE_URL = "https://api.featherless.ai/v1"

# Each stage maps to a LIST of candidate models, tried in order. If the
# first model fails (including exhausting its own retries on transient
# errors like capacity_exhausted), the next one is tried automatically -
# real redundancy, not just hoping one model choice stays available.
MODEL_MAP = {
    "extraction": ["Qwen/Qwen2.5-Math-72B-Instruct", "deepseek-ai/DeepSeek-R1-0528"],
    "math_check": ["Qwen/Qwen2.5-Math-72B-Instruct", "deepseek-ai/DeepSeek-R1-0528"],
    "judge": ["deepseek-ai/DeepSeek-R1-0528", "Qwen/Qwen2.5-Math-72B-Instruct"],
    "ocr": ["Qwen/Qwen2.5-VL-72B-Instruct"],  # needs vision support - no confirmed fallback vision model yet
}


def _get_client(secrets: dict) -> OpenAI:
    """secrets: whatever dict-like object you're pulling config from -
    st.secrets in app.py, or a plain dict for local testing without
    Streamlit at all."""
    api_key = secrets.get("FEATHERLESS_API_KEY")

    if not api_key:
        raise RuntimeError(
            "No API key found. Expected st.secrets['FEATHERLESS_API_KEY']."
        )

    # Stripping whitespace is critical - a trailing space or newline from
    # a copy-paste is exactly what caused a real AuthenticationError
    # during testing, and it doesn't show visually in a secrets editor.
    api_key = str(api_key).strip()

    return OpenAI(base_url=FEATHERLESS_BASE_URL, api_key=api_key, timeout=30.0)


def _models_for(stage: str) -> list[str]:
    return MODEL_MAP[stage]


class ModelOutputError(Exception):
    """Raised when a model's response can't be turned into usable data,
    after every reasonable extraction attempt. Carries the raw text so the
    caller can show it - a blind crash with no visibility into what the
    model actually said is much harder to debug than seeing the real
    output, especially on Streamlit Cloud where tracebacks are redacted."""
    def __init__(self, raw_text: str, context: str = ""):
        self.raw_text = raw_text
        msg = f"{context}\n\nRaw API output:\n{raw_text[:3000]}" if context else raw_text[:3000]
        super().__init__(msg)


def _extract_json(raw_text: str):
    """Turns a model's raw response into parsed data, trying progressively
    looser strategies - models reliably break a 'JSON only' instruction in
    a handful of predictable ways."""
    candidates = [raw_text.strip()]

    fence_stripped = re.sub(
        r"^```(?:json)?\s*|\s*```$", "", raw_text.strip(),
        flags=re.MULTILINE | re.IGNORECASE,
    )
    candidates.append(fence_stripped)

    bracket_match = re.search(r"(\[.*\]|\{.*\})", fence_stripped, flags=re.DOTALL)
    if bracket_match:
        candidates.append(bracket_match.group(1))

    for candidate in candidates:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue

    for candidate in candidates:
        try:
            return ast.literal_eval(candidate)
        except (ValueError, SyntaxError):
            continue

    raise ModelOutputError(raw_text, context="Could not parse model output as JSON or as a Python literal, after trying fence-stripping and bracket extraction.")


def _chat(client: OpenAI, model: str, system_prompt: str, user_content: str,
          temperature: float = 0.0, max_tokens: int = 2048, retries: int = 3) -> str:
    """Retries on transient failures - rate limits, and capacity errors
    like 'temporarily at capacity' (a real error hit during testing, not
    hypothetical) - since those are explicitly temporary per Featherless's
    own error message, and a silent auto-retry beats a failed take during
    recording. Does NOT retry on things retrying can't fix (bad model
    name, auth failure, malformed request) - those fail immediately."""
    import time

    last_err = None
    for attempt in range(retries):
        raw_text_body = ""
        try:
            raw_resp = client.with_raw_response.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ],
                temperature=temperature,
                max_completion_tokens=max_tokens,  # not max_tokens - some models reject the older parameter name with a 400
            )

            raw_text_body = raw_resp.text

            if raw_resp.status_code == 200:
                response = raw_resp.parse()
                return _extract_content(response, raw_text_body)

            is_retryable = raw_resp.status_code == 429 or "capacity" in raw_text_body.lower()
            if is_retryable and attempt < retries - 1:
                last_err = RuntimeError(f"HTTP {raw_resp.status_code}: {raw_text_body}")
                time.sleep(2 * (attempt + 1))
                continue

            raise RuntimeError(
                f"Featherless API returned HTTP status {raw_resp.status_code}:\n{raw_text_body}\n\n"
                f"Troubleshooting Tips:\n"
                f"• If status is 403: The model '{model}' is gated. Go to your Featherless dashboard and click 'Unlock Model' to accept its license.\n"
                f"• If status is 400: The model is cold/not ready. Wait a moment or try again."
            )

        except (RateLimitError, APIStatusError) as e:
            status = getattr(e, "status_code", None)
            body = e.response.text if hasattr(e, "response") else str(e)
            is_retryable = status == 429 or "capacity" in body.lower()
            last_err = RuntimeError(f"Featherless API Error (Status {status}): {body}")
            if is_retryable and attempt < retries - 1:
                time.sleep(2 * (attempt + 1))
                continue
            raise last_err from e
        except APIError as e:
            raise RuntimeError(f"Featherless API Error: {e}") from e
        except Exception as e:
            if isinstance(e, RuntimeError):
                raise
            raise RuntimeError(f"Provider API call failed: {e}") from e

    raise last_err or RuntimeError(f"_chat failed after {retries} attempts with no captured error")


def _extract_content(response, raw_text_body: str) -> str:
    """Pulls the message text out of a parsed response, handling both the
    typed object and the raw-dict shapes the SDK can return."""
    try:
        if hasattr(response, "model_dump"):
            data = response.model_dump()
            choices = data.get("choices")
            if not choices:
                raise ValueError(f"API response contained no choices. Raw body: {raw_text_body}")
            content = choices[0]["message"]["content"]
        elif hasattr(response, "choices") and response.choices:
            content = response.choices[0].message.content
        else:
            raise ValueError(f"Unrecognized response structure. Raw body: {raw_text_body}")

        if content is None:
            raise ValueError(f"Message content was explicitly null. Raw body: {raw_text_body}")

        return str(content)

    except (IndexError, KeyError, TypeError, AttributeError, ValueError) as e:
        raise ModelOutputError(
            raw_text=raw_text_body or str(response),
            context=f"Failed to extract text from API response ({type(e).__name__}: {e})."
        )


def _chat_with_fallback(client: OpenAI, models: list[str], system_prompt: str,
                         user_content: str, **kwargs) -> tuple[str, str]:
    """Tries each model in order. A model only gets skipped after its own
    retries (inside _chat) are exhausted - this isn't 'try everything
    once', it's 'exhaust the primary model's retries, THEN fail over to a
    genuinely different model' - real redundancy against one specific
    model being down, not just a second roll of the same dice.

    Returns (content, model_that_actually_answered) - callers that record
    which model produced a result (like run_math_check's model_id field)
    need the real answer, not an assumption that the first model in the
    list always won."""
    last_err = None
    for model in models:
        try:
            return _chat(client, model, system_prompt, user_content, **kwargs), model
        except Exception as e:
            last_err = e
            continue
    raise last_err or RuntimeError("No models available in fallback chain")


# ---------------------------------------------------------------------------
# Stage 1: Extraction
# ---------------------------------------------------------------------------

def run_extraction(secrets: dict, source_text: str) -> list[dict]:
    client = _get_client(secrets)
    raw, _ = _chat_with_fallback(client, _models_for("extraction"), EXTRACTION_SYSTEM_PROMPT, source_text)
    parsed = _extract_json(raw)

    if isinstance(parsed, list):
        return parsed

    if isinstance(parsed, dict):
        for key in ("steps", "extracted_steps", "derivation"):
            if isinstance(parsed.get(key), list):
                return parsed[key]
        raise ModelOutputError(raw, context="Extraction model returned a JSON object, but none of the expected keys (steps/extracted_steps/derivation) held a list.")

    raise ModelOutputError(raw, context=f"Extraction model returned valid JSON, but it was a {type(parsed).__name__}, not a list.")


# ---------------------------------------------------------------------------
# Stage 2: Math-check escalation
# ---------------------------------------------------------------------------

def run_math_check(secrets: dict, step_index: int, prev_latex: str, curr_latex: str) -> MathModelOpinion:
    client = _get_client(secrets)
    user_content = json.dumps({
        "step_index": step_index,
        "previous_step_latex": prev_latex,
        "current_step_latex": curr_latex,
    })
    raw, model_used = _chat_with_fallback(client, _models_for("math_check"), MATH_CHECK_SYSTEM_PROMPT, user_content)
    parsed = _extract_json(raw)
    return MathModelOpinion(
        step_index=parsed["step_index"],
        model_id=model_used,
        verdict=parsed["verdict"],
        explanation=parsed.get("explanation", ""),
    )


# ---------------------------------------------------------------------------
# Stage 3: Judge
# ---------------------------------------------------------------------------

def run_judge(secrets: dict, flagged_steps: list[dict],
              math_model_opinions: Optional[list[dict]] = None) -> list[JudgeVerdict]:
    client = _get_client(secrets)
    user_content = json.dumps({
        "flagged_steps": flagged_steps,
        "math_model_opinions": math_model_opinions or [],
    })
    raw, _ = _chat_with_fallback(client, _models_for("judge"), JUDGE_SYSTEM_PROMPT, user_content)
    parsed = _extract_json(raw)
    return [
        JudgeVerdict(
            step_index=v["step_index"],
            is_error=v["is_error"],
            plain_language_explanation=v["plain_language_explanation"],
            confidence_note=v.get("confidence_note", ""),
        )
        for v in parsed
    ]


# ---------------------------------------------------------------------------
# Stage 0: OCR (only called when core/pdf_parser.py finds no text layer)
# ---------------------------------------------------------------------------

_OCR_INSTRUCTION = (
    "Transcribe all visible text in this page image exactly as written. "
    "Output only the transcribed text, nothing else - no commentary, no "
    "markdown fences."
)


def run_ocr(secrets: dict, page_image_b64: str, retries: int = 3) -> str:
    import time

    client = _get_client(secrets)
    model = _models_for("ocr")[0]  # single vision-capable model for now, no fallback yet

    last_err = None
    for attempt in range(retries):
        try:
            raw_resp = client.with_raw_response.chat.completions.create(
                model=model,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": _OCR_INSTRUCTION},
                        {"type": "image_url", "image_url": {
                            "url": f"data:image/png;base64,{page_image_b64}"
                        }},
                    ],
                }],
            )
            
            if raw_resp.status_code != 200:
                raise RuntimeError(f"OCR HTTP {raw_resp.status_code}: {raw_resp.text}")

            response = raw_resp.parse()
            if hasattr(response, "model_dump"):
                choices = response.model_dump().get("choices")
                if not choices:
                    raise ValueError(f"OCR response contained no choices: {raw_resp.text}")
                content = choices[0]["message"]["content"]
            else:
                content = response.choices[0].message.content
                
            return str(content).strip()
            
        except Exception as e:
            last_err = e
            if attempt < retries - 1:
                is_rate_limit = "429" in str(e) or "rate" in str(e).lower()
                time.sleep(2 * (attempt + 1) if is_rate_limit else 1)
    raise RuntimeError(f"OCR failed after {retries} attempts: {last_err}")


if __name__ == "__main__":
    fake_secrets = {}
    try:
        run_extraction(fake_secrets, "x + 1 = 2")
    except RuntimeError as e:
        print("Expected failure with no key configured:", e)

"""
models/featherless_client.py

Provider-agnostic client for all three LLM stages. Groq and Featherless
both expose OpenAI-compatible /chat/completions endpoints, so this is one
implementation with base_url/api_key/model swapped by config - not two
separate clients.

SWAP MECHANISM: set st.secrets["PROVIDER"] to "groq" or "featherless".
Nothing else in this file, or in app.py, needs to change. That's the whole
point of MODEL_MAP below - the provider swap is a one-line config edit,
not a code edit, which is what makes "swap Friday" actually true rather
than aspirational.

MODEL_MAP CAVEAT: Groq's catalog is small and curated - it will not have
Qwen2.5-Math or most Featherless-exclusive models. The "groq" row below is
a best-effort substitute for dev/testing the *pipeline shape*, not a
like-for-like stand-in. Don't treat a clean Groq test run as proof the
Featherless run will behave the same - the math-check stage in particular
is testing a different, weaker model until you flip PROVIDER to
"featherless" on real Featherless credits.
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


BASE_URLS = {
    "groq": "https://api.groq.com/openai/v1",
    "featherless": "https://api.featherless.ai/v1",
}

# One row per pipeline stage. "groq" values are dev substitutes, not the
# real submission models - see caveat above.
MODEL_MAP = {
    "extraction": {
        "groq": "openai/gpt-oss-120b",
        "featherless": "Qwen/Qwen2.5-Math-72B-Instruct",
    },
    "math_check": {
        "groq": "openai/gpt-oss-120b",
        "featherless": "Qwen/Qwen2.5-Math-72B-Instruct",
    },
    "judge": {
        "groq": "openai/gpt-oss-120b",
        "featherless": "deepseek-ai/DeepSeek-R1-0528",
    },
    "ocr": {
        "groq": "qwen/qwen3.6-27b",  # needs vision support, only used for scanned PDFs
        "featherless": "Qwen/Qwen2.5-VL-72B-Instruct",
    },
}


def _get_client(secrets: dict) -> OpenAI:
    """secrets: whatever dict-like object you're pulling config from -
    st.secrets in app.py, or a plain dict for local testing without
    Streamlit at all."""
    
    # 1. Safely extract and clean the provider string.
    raw_provider = secrets.get("PROVIDER", "featherless")
    provider = str(raw_provider).strip().lower()

    if provider not in BASE_URLS:
        raise RuntimeError(
            f"Invalid PROVIDER: '{provider}'. "
            f"Check your secrets.toml (must be 'groq' or 'featherless')."
        )

    # 2. Extract and clean the API key. Stripping whitespace is critical
    #    because a trailing space will cause a 401 AuthenticationError.
    api_key_name = f"{provider.upper()}_API_KEY"
    api_key = secrets.get(api_key_name)

    if not api_key:
        raise RuntimeError(
            f"No API key found for provider '{provider}'. "
            f"Expected st.secrets['{api_key_name}']."
        )
    
    api_key = str(api_key).strip()

    # 3. Safeguard against key misrouting: sending a Featherless key to Groq returns 401.
    if provider == "groq" and not api_key.startswith("gsk_"):
        raise RuntimeError(
            f"Authentication safeguard: PROVIDER is set to '{provider}' but the key in '{api_key_name}' "
            f"does not start with 'gsk_'. You are likely sending a Featherless key to Groq's endpoint. "
            f"Update your secrets.toml to set PROVIDER = \"featherless\"."
        )

    return OpenAI(base_url=BASE_URLS[provider], api_key=api_key), provider


def _model_for(stage: str, provider: str) -> str:
    return MODEL_MAP[stage][provider]


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
          temperature: float = 0.0, max_tokens: int = 2048) -> str:
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
    except APIStatusError as e:
        # Handles 400 (Cold model), 401, 403, 503 (Insufficient capacity), etc.
        err_body = e.response.text if hasattr(e, "response") else str(e)
        raise RuntimeError(
            f"Featherless API Error (Status {e.status_code}): {err_body}\n\n"
            f"Note: If this is a 400 error stating the model is 'cold', check your Featherless dashboard or try again once warmed up."
        ) from e
    except RateLimitError as e:
        raise RuntimeError(f"Featherless Rate Limit Exceeded (429): {e}") from e
    except APIError as e:
        raise RuntimeError(f"Featherless API Error: {e}") from e
    except Exception as e:
        raise RuntimeError(f"Provider API call failed: {e}") from e

    # Safely extract choices/content
    try:
        if hasattr(response, "model_dump"):
            data = response.model_dump()
            choices = data.get("choices")
            if not choices:
                raise ValueError(f"API response contained no choices. Full response: {data}")
            content = choices[0]["message"]["content"]
        elif hasattr(response, "choices") and response.choices:
            content = response.choices[0].message.content
        else:
            raise ValueError(f"Unrecognized or empty response structure: {response}")

        if content is None:
            raise ValueError("Message content was explicitly null.")
            
        return str(content)
        
    except (IndexError, KeyError, TypeError, AttributeError, ValueError) as e:
        raise ModelOutputError(
            raw_text=str(response), 
            context=f"Failed to extract text from API response ({type(e).__name__}: {e})."
        )


# ---------------------------------------------------------------------------
# Stage 1: Extraction
# ---------------------------------------------------------------------------

def run_extraction(secrets: dict, source_text: str) -> list[dict]:
    client, provider = _get_client(secrets)
    raw = _chat(client, _model_for("extraction", provider), EXTRACTION_SYSTEM_PROMPT, source_text)
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
    client, provider = _get_client(secrets)
    model = _model_for("math_check", provider)
    user_content = json.dumps({
        "step_index": step_index,
        "previous_step_latex": prev_latex,
        "current_step_latex": curr_latex,
    })
    raw = _chat(client, model, MATH_CHECK_SYSTEM_PROMPT, user_content)
    parsed = _extract_json(raw)
    return MathModelOpinion(
        step_index=parsed["step_index"],
        model_id=model,
        verdict=parsed["verdict"],
        explanation=parsed.get("explanation", ""),
    )


# ---------------------------------------------------------------------------
# Stage 3: Judge
# ---------------------------------------------------------------------------

def run_judge(secrets: dict, flagged_steps: list[dict],
              math_model_opinions: Optional[list[dict]] = None) -> list[JudgeVerdict]:
    client, provider = _get_client(secrets)
    model = _model_for("judge", provider)
    user_content = json.dumps({
        "flagged_steps": flagged_steps,
        "math_model_opinions": math_model_opinions or [],
    })
    raw = _chat(client, model, JUDGE_SYSTEM_PROMPT, user_content)
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

    client, provider = _get_client(secrets)
    model = _model_for("ocr", provider)

    last_err = None
    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
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
            
            if hasattr(response, "model_dump"):
                data = response.model_dump()
                choices = data.get("choices")
                if not choices:
                    raise ValueError(f"OCR response contained no choices: {data}")
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

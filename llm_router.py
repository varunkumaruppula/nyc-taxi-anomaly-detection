"""
LLM router for the taxi data explorer (optional enhancement layer).

Lets a user type a question in plain English; a small local model
(qwen2.5-coder:1.5b via Ollama) maps it to one of the canonical intents,
whose tested SQL then runs. The model NEVER writes SQL -- it only picks
from the fixed intent menu, so it cannot produce a wrong or unsafe query.

Designed for a memory-constrained machine (8GB, often busy):
  - the LLM is OPTIONAL. If Ollama isn't running or is too slow, every
    caller falls back to the dropdown, which always works.
  - a strict timeout means a bogged-down machine never hangs the UI.
  - all failures are caught and reported as "use the dropdown", never as
    a crash.
"""

import requests

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "qwen2.5-coder:1.5b"
TIMEOUT_SECONDS = 12  # strict: a busy machine fails fast to the dropdown


def ollama_available():
    """Quick health check -- is Ollama running and reachable at all?"""
    try:
        r = requests.get("http://localhost:11434/api/tags", timeout=2)
        return r.status_code == 200
    except Exception:
        return False


def build_router_prompt(question, intents):
    """The prompt: the question plus the menu of intents to choose from."""
    lines = [
        "You are a router for a taxi-data question system. Given the user's "
        "question, return EXACTLY ONE intent name from the list below that "
        "best matches it. If nothing fits, return the single word NONE. "
        "Return only the intent name, with no explanation.",
        "",
        "Available intents:",
    ]
    for name, spec in intents.items():
        lines.append(f"- {name}: {spec['description']}")
    lines += ["", f"User question: {question}", "Intent name:"]
    return "\n".join(lines)


def parse_router_response(raw, intent_names):
    """
    Extract a valid intent name from the model's (possibly messy) output.
    Prefers the LONGEST matching name so that, if one intent name is a
    substring of another, the more specific one wins.
    """
    cleaned = raw.strip().lower()
    matches = [n for n in intent_names if n.lower() in cleaned]
    if not matches:
        return None
    return max(matches, key=len)


def route_question(question, intents):
    """
    Map a plain-English question to an intent name using the local LLM.
    Returns the intent name, or None if the LLM is unavailable, times out,
    errors, or finds no match -- in every one of those cases the caller
    should fall back to the dropdown.
    """
    if not question or not question.strip():
        return None
    if not ollama_available():
        return None

    prompt = build_router_prompt(question, intents)
    try:
        resp = requests.post(
            OLLAMA_URL,
            json={"model": MODEL, "prompt": prompt, "stream": False,
                  "options": {"temperature": 0}},  # deterministic routing
            timeout=TIMEOUT_SECONDS,
        )
        if resp.status_code != 200:
            return None
        raw = resp.json().get("response", "")
        return parse_router_response(raw, list(intents.keys()))
    except requests.exceptions.Timeout:
        return None
    except Exception:
        return None


def explain_result(question, intent_description, result_preview):
    """
    Optional: turn a query result into a one-sentence plain-English summary.
    Same defensive design -- returns None on any failure, and the caller
    just shows the table/chart without a narrated summary.
    """
    if not ollama_available():
        return None
    prompt = (
        f"In ONE plain-English sentence for a non-technical reader, summarize "
        f"the answer to the question '{question}'. The data shows: "
        f"{result_preview}. Be concise and specific. One sentence only."
    )
    try:
        resp = requests.post(
            OLLAMA_URL,
            json={"model": MODEL, "prompt": prompt, "stream": False,
                  "options": {"temperature": 0.2}},
            timeout=TIMEOUT_SECONDS,
        )
        if resp.status_code != 200:
            return None
        return resp.json().get("response", "").strip() or None
    except Exception:
        return None
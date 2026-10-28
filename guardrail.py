"""
guardrails.py (NEW FILE) - safety checks around the chatbot.

    allow_request(user_id)     rate limit: stops one user from flooding your local model
    check_question(question)   before the agent: blocks prompt-injection, hides private numbers
    check_answer(result)       after the agent: no leaked rules, warns when an answer is not cited/verified

Everything is plain Python, free, and adds no model calls. Limits can be changed in .env.
"""
import os
import re
import threading
import time
from collections import defaultdict, deque

# ---------- 1. rate limit ----------
RATE_LIMIT = int(os.getenv("RATE_LIMIT_QUESTIONS", "20"))      # questions ...
RATE_WINDOW = int(os.getenv("RATE_LIMIT_WINDOW_SECONDS", "60"))  # ... per this many seconds, per user
_hits: dict[int, deque] = defaultdict(deque)
_lock = threading.Lock()


def allow_request(user_id: int) -> bool:
    """True if this user may ask now. Sliding window, kept in memory."""
    now = time.time()
    with _lock:
        q = _hits[user_id]
        while q and now - q[0] > RATE_WINDOW:
            q.popleft()
        if len(q) >= RATE_LIMIT:
            return False
        q.append(now)
        return True


# ---------- 2. question checks ----------
REFUSAL = "I can only answer questions about your uploaded documents."

INJECTION = re.compile(
    r"(ignore|disregard|forget|override)\s+(all\s+|any\s+|the\s+|your\s+|previous\s+|above\s+|prior\s+)*"
    r"(instructions?|rules?|prompts?)"
    r"|(reveal|show|print|repeat|tell me)\s+(me\s+)?(your|the)\s+(system\s+)?(prompt|instructions?|rules)"
    r"|you are now\b|act as (an? )?(unrestricted|jailbroken|dan)\b|developer mode|jailbreak",
    re.I,
)

# private numbers that should not be sent to the model or written to logs
PII = [   # card numbers first: a 16-digit card contains a 12-digit run
    (re.compile(r"\b(?:\d[ -]?){13,16}\b"), "[CARD NUMBER REMOVED]"),            # card numbers
    (re.compile(r"\b\d{4}[ -]?\d{4}[ -]?\d{4}\b"), "[AADHAAR REMOVED]"),          # 12-digit Aadhaar
    (re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b"), "[PAN REMOVED]"),                    # Indian PAN
]


def check_question(question: str) -> tuple[str, str | None]:
    """Returns (clean_question, blocked_message). If blocked_message is not None, do not run the agent."""
    if INJECTION.search(question):
        return question, REFUSAL
    clean = question
    for pattern, label in PII:
        clean = pattern.sub(label, clean)
    return clean, None


# ---------- 3. answer checks ----------
MAX_ANSWER_CHARS = int(os.getenv("MAX_ANSWER_CHARS", "6000"))
LEAK = re.compile(r"=== DOCUMENT:|\[Excerpt \d+ \||You are a careful assistant|CALCULATION RESULTS \(computed", re.I)
CITATION = re.compile(r"\bpage\s*\d+|\.pdf\b", re.I)

NOTE_NO_CITATION = "\n\n> ⚠ This answer has no page citation. Please check it against your documents."
NOTE_UNVERIFIED = "\n\n> ⚠ The agent corrected this answer after its self-check and could not check it again. Please verify."


def check_answer(result: dict) -> dict:
    """Last line of defence on the finished result (agent or simple mode). Returns the (possibly edited) result."""
    answer = str(result.get("answer") or "")

    if LEAK.search(answer):                      # the model repeated its own prompt or raw context labels
        result["answer"] = "I couldn't produce a safe answer. Please rephrase your question."
        result["retrieval_ok"] = False
        return result

    answer = answer[:MAX_ANSWER_CHARS]
    if result.get("retrieval_ok") and not CITATION.search(answer):
        answer += NOTE_NO_CITATION
    if result.get("revised") and not result.get("verified"):
        answer += NOTE_UNVERIFIED
    result["answer"] = answer
    return result
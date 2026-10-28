import json
import re
import threading
from collections import defaultdict, deque

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

import config
from config import AGENT_JSON_TOKENS, LLM_MODEL, NUM_CTX, OLLAMA_BASE_URL, TOP_K
import database
from ingestion import embedding_model

RELEVANCE_THRESHOLD = 0.25

SYSTEM_PROMPT = """You are a knowledgeable assistant that explains the user's PDF in depth.

Rules:
- Base every fact on the provided CONTEXT. Never invent facts that are not in it.
- Give a thorough, well-structured answer, not a one-line reply.
- Start with a short direct answer (1-2 sentences), then explain the details.
- Cover every relevant point found across ALL the context chunks, and combine
  them into one clear explanation.
- Explain terms in simple language and add a short example when it helps.
- Use bullet points or short headings when there are several points or steps.
- When comparing items or presenting structured data, use a markdown table with a header row.
- Mention the page numbers you used, e.g. (page 2).
- If the context only covers part of the question, answer that part fully, then
  say clearly which part is not covered in the document.
- If the context has nothing relevant, say you don't know."""

# Short chat memory: last 3 question/answer pairs per thread_id, so follow-up
# questions ("what about that?") keep their context without leaking between users.
_HISTORY_MESSAGES = 6
_history = defaultdict(lambda: deque(maxlen=_HISTORY_MESSAGES))
_history_version = defaultdict(int)
_history_lock = threading.Lock()


def _history_snapshot(thread_id: str) -> list:
    with _history_lock:
        return list(_history[thread_id])


def thread_signature(thread_id: str) -> int:
    """Increments after every remembered turn; keeps the answer cache fresh."""
    with _history_lock:
        return _history_version[thread_id]


def remember_turn(thread_id: str, question: str, answer: str) -> None:
    """Save the plain question/answer (not the big context block) for this thread."""
    if not thread_id:
        return
    with _history_lock:
        _history[thread_id].append(HumanMessage(question))
        _history[thread_id].append(AIMessage(answer))
        _history_version[thread_id] += 1


def _build_llms():
    """Create the answer model and the JSON helper model for the configured provider.

    Groq and Ollama take different constructor arguments, so passing Ollama options
    (num_ctx / num_predict / format) to Groq used to make every LLM call fail.
    """
    provider = config.LLM_PROVIDER

    if provider == "groq":
        try:
            from langchain_groq import ChatGroq
        except ImportError as e:
            raise RuntimeError(
                "LLM_PROVIDER=groq but langchain-groq is not installed. "
                "Run: pip install langchain-groq"
            ) from e
        if not config.GROQ_API_KEY:
            raise RuntimeError(
                "LLM_PROVIDER=groq but GROQ_API_KEY is missing. "
                "Add GROQ_API_KEY=<your key> to the .env file."
            )

        settings = {
            "model": LLM_MODEL,
            "api_key": config.GROQ_API_KEY,
            "max_retries": 2,
            "timeout": 120,
        }
        # gpt-oss models reason by default; keep the small helper calls cheap and fast.
        json_extra = {"reasoning_effort": "low"} if "gpt-oss" in LLM_MODEL.lower() else {}
        answer = ChatGroq(**settings, temperature=0.3, max_tokens=config.LLM_MAX_TOKENS)
        helpers = ChatGroq(
            **settings,
            temperature=0,
            max_tokens=AGENT_JSON_TOKENS,
            **json_extra,
        ).bind(response_format={"type": "json_object"})
        return answer, helpers

    if provider != "ollama":
        raise RuntimeError(
            f"Unknown LLM_PROVIDER={provider!r}. Use 'groq' or 'ollama' in .env."
        )

    from langchain_ollama import ChatOllama

    answer = ChatOllama(
        model=LLM_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=0.3,              # slightly higher = more natural, fuller wording
        num_predict=config.LLM_MAX_TOKENS,
        num_ctx=NUM_CTX,              # lower it if Ollama runs out of memory
    )
    # format="json" makes Ollama return valid JSON; temperature=0 keeps helpers steady.
    helpers = ChatOllama(
        model=LLM_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=0,
        num_predict=AGENT_JSON_TOKENS,
        num_ctx=NUM_CTX,
        format="json",
    )
    return answer, helpers


# Built once at import time, not per request
_llm, _llm_json = _build_llms()


def embed_query(question: str) -> list[float]:
    vector = embedding_model.encode(
        [question],
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return vector[0].tolist()


def build_context_block(chunks: list[dict]) -> str:
    parts = []
    for i, c in enumerate(chunks, start=1):
        parts.append(
            f"[Chunk {i} | Page {c['page_number']} | Similarity {c['similarity_score']:.3f}]\n"
            f"{c['chunk_text']}"
        )
    return "\n\n---\n\n".join(parts)


def generate_answer(question: str, context_chunks: list[dict],
                    thread_id: str = "default") -> str:
    context_block = build_context_block(context_chunks)

    messages = [SystemMessage(SYSTEM_PROMPT)]
    messages += _history_snapshot(thread_id)
    messages.append(HumanMessage(f"CONTEXT:\n{context_block}\n\nQUESTION:\n{question}"))

    return _llm.invoke(messages).content


# --- AGENTIC CHANGE (3/6): new building blocks used by agent.py -------------------------
def parse_json(text: str):
    """Pull a JSON object out of a model reply, even with code fences or chatter. None if hopeless."""
    if not text:
        return None
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    try:
        return json.loads(cleaned)
    except ValueError:
        pass
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)  # first "{" to last "}"
    if match:
        try:
            return json.loads(match.group(0))
        except ValueError:
            return None
    return None


def ask_json(system: str, prompt: str):
    """A small helper call that must answer with JSON. Returns the parsed value, or None.

    Invalid or truncated JSON is retried once with a short repair instruction.
    """
    messages = [SystemMessage(system), HumanMessage(prompt)]
    for attempt in range(2):
        reply = _llm_json.invoke(messages).content
        data = parse_json(reply)
        if data is not None:
            return data
        if attempt == 0:
            messages += [
                AIMessage(reply),
                HumanMessage("That reply was not valid JSON. Reply again with ONLY one valid JSON object, no markdown."),
            ]
    return None


def search_chunks(query: str, top_k: int, document_id: int | None, user_id: int | None) -> list[dict]:
    """One vector search (same call answer_question makes). Returns raw chunks with similarity_score."""
    return database.similarity_search(
        embed_query(query), top_k=top_k, document_id=document_id, user_id=user_id
    )


def revise_answer(question: str, context_chunks: list[dict], draft: str, problems: list[str]) -> str:
    """Rewrite a draft after the self-check found claims the context does not support."""
    prompt = (
        f"CONTEXT:\n{build_context_block(context_chunks)}\n\nQUESTION:\n{question}\n\n"
        f"DRAFT ANSWER:\n{draft}\n\n"
        "The checker found these claims that the CONTEXT does NOT support:\n- " + "\n- ".join(problems) +
        "\n\nWrite a corrected answer. Remove or fix those claims. Keep the page numbers."
    )
    return _llm.invoke([SystemMessage(SYSTEM_PROMPT), HumanMessage(prompt)]).content


def answer_question(question: str, document_id: int | None = None,
                    top_k: int = TOP_K, user_id: int | None = None,
                    thread_id: str = "default") -> dict:
    """Retrieve relevant chunks, then answer using only those chunks.

    Args:
        question: the user's question.
        document_id: optionally restrict the search to one PDF.
        top_k: how many chunks to retrieve.
        user_id: restrict retrieval to this user's documents.
        thread_id: conversation id used for short chat memory.

    Returns:
        dict with "answer", "sources" and "retrieval_ok".
    """
    query_embedding = embed_query(question)
    retrieved = database.similarity_search(
        query_embedding, top_k=top_k, document_id=document_id, user_id=user_id
    )
    relevant_chunks = [c for c in retrieved if c["similarity_score"] >= RELEVANCE_THRESHOLD]

    if not relevant_chunks:
        return {
            "answer": ("I couldn't find anything in the uploaded PDF that's relevant "
                       "to this question. Try rephrasing, or confirm this topic is "
                       "actually covered in the document."),
            "sources": retrieved,
            "retrieval_ok": False,
        }

    answer = generate_answer(question, relevant_chunks, thread_id=thread_id)
    remember_turn(thread_id, question, answer)
    return {"answer": answer, "sources": relevant_chunks, "retrieval_ok": True}

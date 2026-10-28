import os
from dotenv import load_dotenv

load_dotenv()

# --- App API key (clients send it as the X-API-Key header) ---
APP_API_KEY = os.getenv("APP_API_KEY", "").strip()

# --- LLM provider ---
# "groq" = hosted Groq API (needs GROQ_API_KEY), "ollama" = local Ollama server.
# If LLM_PROVIDER is not set, Groq is used when a key is present, otherwise Ollama.
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip().strip('"').strip("'")
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq" if GROQ_API_KEY else "ollama").strip().lower()

# --- Model ---
# Default model depends on the provider, so Groq never gets an Ollama-only name.
_DEFAULT_LLM_MODEL = "openai/gpt-oss-120b" if LLM_PROVIDER == "groq" else "llama3.1:8b"
LLM_MODEL = os.getenv("LLM_MODEL", "").strip() or _DEFAULT_LLM_MODEL

# --- Ollama (local LLM) ---
# No API key needed: Ollama runs on your machine.
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
# Max tokens generated for a full answer (Groq: max_tokens, Ollama: num_predict).
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "2048"))
# Context window in tokens. Memory use grows with it (8192 needs ~1 GB extra for llama3.1:8b).
# If Ollama says "out of memory ... kv cache", lower this (2048 or 3072) or use a smaller model.
NUM_CTX = int(os.getenv("NUM_CTX", "4096"))

# --- Embeddings (local sentence-transformers model) ---
# EMBEDDING_DIM must match EMBEDDING_MODEL:
#   all-MiniLM-L6-v2 -> 384
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "384"))

# --- PostgreSQL ---
POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
POSTGRES_PORT = os.getenv("POSTGRES_PORT", "5432")
POSTGRES_DB = os.getenv("POSTGRES_DB", "rag_assistant")
POSTGRES_USER = os.getenv("POSTGRES_USER", "postgres")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "root%40123")

DATABASE_URL = (
    f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}"
    f"@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"
)

# --- Chunking ---
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))       # characters per chunk
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))  # overlap between chunks

# --- Retrieval ---
TOP_K = int(os.getenv("TOP_K", "3"))  # how many chunks to retrieve per question

# --- Uploads ---
UPLOAD_DIR = os.getenv("UPLOAD_DIR", "uploads")

# --- AGENTIC CHANGE (1/6): agent budgets. Every value can be overridden in .env ---
def _bool(name: str, default: bool) -> bool:
    return os.getenv(name, str(default)).strip().lower() in ("1", "true", "yes", "on")


AGENT_MAX_SUBQUESTIONS = int(os.getenv("AGENT_MAX_SUBQUESTIONS", "3"))  # planner splits a question into at most this many
AGENT_MAX_SEARCH_TRIES = int(os.getenv("AGENT_MAX_SEARCH_TRIES", "3"))  # searches per sub-question (1st try + rewrites)
AGENT_MAX_CHUNKS = int(os.getenv("AGENT_MAX_CHUNKS", "8"))              # chunks handed to the answer step
AGENT_STRONG_SCORE = float(os.getenv("AGENT_STRONG_SCORE", "0.55"))     # >= this (with 3+ chunks) = no need to ask the judge
AGENT_ALLOW_WIDEN = _bool("AGENT_ALLOW_WIDEN", True)                    # if one PDF is selected and has no answer, also search the others
AGENT_VERIFY = _bool("AGENT_VERIFY", True)                              # self-check the draft answer
AGENT_JSON_TOKENS = int(os.getenv("AGENT_JSON_TOKENS", "1024"))         # max tokens for the small JSON helper calls

"""
FastAPI application entry point (with login sessions + answer cache).

Endpoints:
    GET  /signup, /login      -> auth pages
    POST /signup, /login      -> create account / log in
    POST /logout              -> log out
    GET  /api/me              -> current user
    POST /api/upload          -> upload + ingest a PDF   (login required)
    POST /api/ask             -> ask a question (RAG)    (login required)
                                 body.mode = "agent" (default) or "simple"
    GET  /api/documents       -> list MY documents       (login required)
    GET  /                    -> frontend                (login required)

Setup:
    pip install fastapi uvicorn python-multipart bcrypt cachetools itsdangerous
    set SESSION_SECRET in .env (a long random string)

Run with:
    uvicorn main:app --reload --port 8000
"""
import hashlib
import os
import secrets
import shutil
import threading
import traceback
import uuid
from typing import Literal
from cachetools import TTLCache
from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, Request
from fastapi.responses import FileResponse, RedirectResponse, JSONResponse
from pydantic import BaseModel, Field
from starlette.middleware.sessions import SessionMiddleware

from config import APP_API_KEY, UPLOAD_DIR
import Auth
import guardrail  # GUARDRAILS: new safety checks
import agent  # AGENTIC CHANGE (4/6): the new agent module
import database
import ingestion
import rag

# 1. Initialize FastAPI Application
app = FastAPI(title="AI PDF Knowledge Assistant (RAG)")

# 2. Sessions (signed cookie). The frontend is served from this same app,
#    so CORS is not needed. Do NOT combine allow_origins=["*"] with cookies.
SESSION_SECRET = os.getenv("SESSION_SECRET")
if not SESSION_SECRET:
    raise RuntimeError(
        "SESSION_SECRET is not set. Add SESSION_SECRET=<long-random-string> to your .env file."
    )

app.include_router(Auth.router)


@app.middleware("http")
async def require_api_key(request: Request, call_next):
    """Require X-API-Key on /api/* when APP_API_KEY is set in .env.

    Logged-in browser sessions are exempt, so the bundled UI keeps working.
    """
    if APP_API_KEY and request.url.path.startswith("/api/"):
        if request.session.get("user_id"):
            return await call_next(request)
        supplied = request.headers.get("X-API-Key", "")
        if not secrets.compare_digest(supplied, APP_API_KEY):
            return JSONResponse(status_code=401, content={"detail": "Missing or invalid X-API-Key."})
    return await call_next(request)


# Added AFTER the @app.middleware above so SessionMiddleware is the outer layer
# and `request.session` is already available when require_api_key runs.
app.add_middleware(
    SessionMiddleware,
    secret_key=SESSION_SECRET,
    max_age=60 * 60 * 24 * 7,                       # stay logged in for 7 days
    same_site="lax",
    https_only=os.getenv("HTTPS_ONLY", "0") == "1",  # set HTTPS_ONLY=1 in production
)

# 3. Ensure Upload Directory Exists
os.makedirs(UPLOAD_DIR, exist_ok=True)


@app.on_event("startup")
def on_startup():
    database.init_db()
    Auth.init_users()


# ---------- Answer cache ----------
# Key = user + their "library version" + document + top_k + normalized question.
# Uploading a new PDF bumps the user's version, which invalidates their old answers.
answer_cache = TTLCache(maxsize=2000, ttl=3600)  # 1 hour
_cache_lock = threading.Lock()
_library_version: dict[int, int] = {}


def _cache_key(user_id: int, document_id, top_k: int, question: str,
               mode: str = "simple", thread_sig: int = 0) -> str:
    q = " ".join(question.lower().split())
    ver = _library_version.get(user_id, 0)
    # mode + thread_sig keep cached answers valid as the conversation moves on.
    raw = f"{user_id}|{ver}|{document_id}|{top_k}|{mode}|{thread_sig}|{q}"
    return hashlib.sha256(raw.encode()).hexdigest()


def _bump_library(user_id: int):
    with _cache_lock:
        _library_version[user_id] = _library_version.get(user_id, 0) + 1


# ---------- Request / Response Models ----------

class AskRequest(BaseModel):
    question: str
    document_id: int | None = None  # optional: restrict search to one PDF
    top_k: int | None = Field(default=5, gt=0)
    # AGENTIC CHANGE (4/6): "agent" = plan / retry / widen / self-check, "simple" = the old one search + one answer
    mode: Literal["simple", "agent"] = "agent"


# ---------- Endpoints ----------
# These are plain `def` (not `async def`) because ingestion / RAG are blocking
# calls; FastAPI runs them in a thread pool so they don't freeze the server.

MAX_FILES = 10
MAX_FILE_MB = 25

@app.post("/api/upload")
def upload_pdfs(
    files: list[UploadFile] = File(...),
    user_id: int = Depends(Auth.require_user),
):
    if not files:
        raise HTTPException(status_code=400, detail="No files were uploaded.")
    if len(files) > MAX_FILES:
        raise HTTPException(status_code=400, detail=f"Upload at most {MAX_FILES} files at a time.")

    results = []
    for file in files:
        name = file.filename or "unnamed.pdf"

        if not name.lower().endswith(".pdf"):
            results.append({"filename": name, "success": False, "error": "Only PDF files are supported."})
            continue

        save_path = os.path.join(UPLOAD_DIR, f"{uuid.uuid4().hex}.pdf")
        try:
            # copy to disk, enforcing a size limit
            size = 0
            with open(save_path, "wb") as f:
                while chunk := file.file.read(1024 * 1024):
                    size += len(chunk)
                    if size > MAX_FILE_MB * 1024 * 1024:
                        raise ValueError(f"File is larger than {MAX_FILE_MB} MB.")
                    f.write(chunk)

            # check the real file signature, not just the extension
            with open(save_path, "rb") as f:
                if f.read(5) != b"%PDF-":
                    raise ValueError("This is not a valid PDF.")

            summary = ingestion.ingest_pdf(save_path, filename=name, user_id=user_id)
            results.append(summary)
        except Exception as e:
            print("INGESTION ERROR TRACEBACK:\n", traceback.format_exc())
            results.append({"filename": name, "success": False, "error": str(e)})
        finally:
            if os.path.exists(save_path):
                os.remove(save_path)

    if any(r.get("success") for r in results):
        _bump_library(user_id)  # once, after the batch

    return {
        "uploaded": sum(1 for r in results if r.get("success")),
        "failed": sum(1 for r in results if not r.get("success")),
        "results": results,
    }


@app.post("/api/ask")
def ask_question(payload: AskRequest, user_id: int = Depends(Auth.require_user)):
    question = payload.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    if len(question) > 2000:
        raise HTTPException(status_code=400, detail="Question is too long.")

    # GUARDRAILS (1/3): rate limit, then block prompt-injection and hide private numbers
    if not guardrail.allow_request(user_id):
        raise HTTPException(status_code=429, detail="Too many questions. Please wait a minute.")
    question, blocked = guardrail.check_question(question)
    if blocked:
        return {"answer": blocked, "sources": [], "retrieval_ok": False, "blocked": True}

    top_k = payload.top_k if (payload.top_k and payload.top_k > 0) else 5
    top_k = min(top_k, 20)

    # AGENTIC CHANGE (4/6): choose the path. thread_id is now per user, so one user's
    # short chat memory is never shown to another user (it used to be the shared "default").
    thread_id = f"user-{user_id}"

    key = _cache_key(user_id, payload.document_id, top_k, question,
                     payload.mode, rag.thread_signature(thread_id))
    with _cache_lock:
        hit = answer_cache.get(key)
    if hit is not None:
        return {**hit, "cached": True}

    try:
        if payload.mode == "agent":
            result = agent.run(
                question=question,
                document_id=payload.document_id,
                top_k=top_k,
                user_id=user_id,
                thread_id=thread_id,
            )
        else:
            result = rag.answer_question(
                question=question,
                document_id=payload.document_id,
                top_k=top_k,
                user_id=user_id,
                thread_id=thread_id,
            )
    except Exception as e:
        print("RAG ERROR TRACEBACK:\n", traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"RAG pipeline failed: {str(e)}")

    result = guardrail.check_answer(result)  # GUARDRAILS (2/3): check the finished answer before caching/returning
    with _cache_lock:
        answer_cache[key] = result
    return {**result, "cached": False}


@app.get("/api/documents")
def get_documents(user_id: int = Depends(Auth.require_user)):
    try:
        return database.list_documents(user_id=user_id)
    except Exception as e:
        print("LIST DOCUMENTS ERROR:\n", traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


# ---------- Serve Frontend ----------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INDEX_PATH = os.path.join(BASE_DIR, "index.html")


@app.get("/")
def serve_index(request: Request):
    if not Auth.is_logged_in(request):
        return RedirectResponse("/login", status_code=303)
    if not os.path.isfile(INDEX_PATH):
        raise HTTPException(status_code=404, detail="Frontend files are not available.")
    return FileResponse(INDEX_PATH)
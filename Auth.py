"""
auth.py - signup / login / logout using server-side signed-cookie sessions.

Users live in their own SQLite file (users.db) so your existing database.py
is not touched. Sessions are handled by Starlette's SessionMiddleware, which
is added in main.py.
"""
import os
import re
import sqlite3
import time

import bcrypt
from fastapi import APIRouter, Request, Form, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse

router = APIRouter()

USERS_DB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "users.db")

# Hash used to keep login timing the same when the email doesn't exist
_DUMMY_HASH = bcrypt.hashpw(b"dummy-password", bcrypt.gensalt())

# Simple in-memory brute-force guard: 5 failures -> 5 minute lock per (ip, email)
_failures: dict[str, list[float]] = {}
MAX_FAILS, LOCK_SECONDS = 5, 300


def _db():
    con = sqlite3.connect(USERS_DB)
    con.row_factory = sqlite3.Row
    return con


def init_users():
    with _db() as con:
        con.execute(
            """CREATE TABLE IF NOT EXISTS users(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email TEXT UNIQUE NOT NULL,
                password_hash BLOB NOT NULL,
                created_at REAL NOT NULL DEFAULT (strftime('%s','now')))"""
        )


# ---------- dependency used by protected routes ----------

def require_user(request: Request) -> int:
    uid = request.session.get("user_id")
    if not uid:
        raise HTTPException(status_code=401, detail="Not logged in")
    return uid


def is_logged_in(request: Request) -> bool:
    return bool(request.session.get("user_id"))


# ---------- HTML pages ----------

PAGE = """<!doctype html>
<html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title}</title>
<style>
 body{{font-family:system-ui,sans-serif;background:#f4f5f7;display:flex;justify-content:center;padding-top:80px}}
 .card{{background:#fff;padding:28px;border-radius:12px;width:340px;box-shadow:0 2px 12px #0001}}
 input{{width:100%;box-sizing:border-box;padding:10px;margin:6px 0;border:1px solid #ccc;border-radius:8px}}
 button{{width:100%;padding:10px;margin-top:8px;border:0;border-radius:8px;background:#2563eb;color:#fff;font-size:15px;cursor:pointer}}
 .err{{color:#c00;min-height:1.2em}} a{{color:#2563eb}}
</style></head><body><div class="card">
<h2>{title}</h2><p class="err">{error}</p>
<form method="post" action="{action}">
 <input name="email" type="email" placeholder="Email" required>
 <input name="password" type="password" placeholder="Password (min 8 chars)" minlength="8" required>
 <button>{title}</button>
</form>
<p><a href="{alt_href}">{alt_text}</a></p>
</div></body></html>"""


def _page(kind: str, error: str = "", status: int = 200) -> HTMLResponse:
    if kind == "signup":
        html = PAGE.format(title="Sign up", action="/signup", error=error,
                           alt_href="/login", alt_text="Already have an account? Log in")
    else:
        html = PAGE.format(title="Log in", action="/login", error=error,
                           alt_href="/signup", alt_text="New here? Sign up")
    return HTMLResponse(html, status_code=status)


# ---------- routes ----------

@router.get("/signup")
def signup_page(request: Request):
    if is_logged_in(request):
        return RedirectResponse("/", status_code=303)
    return _page("signup")


@router.post("/signup")
def signup(request: Request, email: str = Form(...), password: str = Form(...)):
    email = email.strip().lower()
    if not re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", email):
        return _page("signup", "Please enter a valid email.", 400)
    if len(password) < 8 or len(password) > 128:
        return _page("signup", "Password must be 8-128 characters.", 400)

    hashed = bcrypt.hashpw(password.encode(), bcrypt.gensalt())
    try:
        with _db() as con:
            cur = con.execute(
                "INSERT INTO users(email, password_hash) VALUES(?, ?)", (email, hashed)
            )
            uid = cur.lastrowid
    except sqlite3.IntegrityError:
        return _page("signup", "That email is already registered.", 409)

    request.session.clear()
    request.session["user_id"] = uid
    return RedirectResponse("/", status_code=303)


@router.get("/login")
def login_page(request: Request):
    if is_logged_in(request):
        return RedirectResponse("/", status_code=303)
    return _page("login")


@router.post("/login")
def login(request: Request, email: str = Form(...), password: str = Form(...)):
    email = email.strip().lower()
    key = f"{request.client.host if request.client else '?'}|{email}"

    now = time.time()
    recent = [t for t in _failures.get(key, []) if now - t < LOCK_SECONDS]
    _failures[key] = recent
    if len(recent) >= MAX_FAILS:
        return _page("login", "Too many attempts. Try again in a few minutes.", 429)

    with _db() as con:
        row = con.execute("SELECT * FROM users WHERE email = ?", (email,)).fetchone()

    stored = row["password_hash"] if row else _DUMMY_HASH
    ok = bcrypt.checkpw(password.encode(), stored) and row is not None
    if not ok:
        _failures[key].append(now)
        return _page("login", "Wrong email or password.", 401)

    _failures.pop(key, None)
    request.session.clear()  # new session on login
    request.session["user_id"] = row["id"]
    return RedirectResponse("/", status_code=303)


@router.post("/logout")
def logout(request: Request):
    request.session.clear()
    return RedirectResponse("/login", status_code=303)


@router.get("/api/me")
def me(request: Request):
    uid = require_user(request)
    with _db() as con:
        row = con.execute("SELECT email FROM users WHERE id = ?", (uid,)).fetchone()
    if not row:
        request.session.clear()
        raise HTTPException(status_code=401, detail="Not logged in")
    return {"id": uid, "email": row["email"]}
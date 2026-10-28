from fastapi.testclient import TestClient

import Auth
import agent
import main
import rag

main.app.dependency_overrides[Auth.require_user] = lambda: 1
client = TestClient(main.app, headers={"X-API-Key": "test-api-key"})  # no `with`: skips startup (no database needed)
FAKE = {"answer": "a", "sources": [], "retrieval_ok": True}


def test_api_key_required():
    bare = TestClient(main.app)
    assert bare.post("/api/ask", json={"question": "hello there"}).status_code == 401


def test_default_mode_is_agent(monkeypatch):
    monkeypatch.setattr(agent, "run", lambda **k: {**FAKE, "mode_used": "agent"})
    monkeypatch.setattr(rag, "answer_question", lambda **k: (_ for _ in ()).throw(AssertionError("simple")))
    r = client.post("/api/ask", json={"question": "hello there"}).json()
    assert r["mode_used"] == "agent"


def test_simple_mode_uses_old_path(monkeypatch):
    monkeypatch.setattr(agent, "run", lambda **k: (_ for _ in ()).throw(AssertionError("agent")))
    monkeypatch.setattr(rag, "answer_question", lambda **k: {**FAKE, "mode_used": "simple"})
    r = client.post("/api/ask", json={"question": "hello there", "mode": "simple"}).json()
    assert r["mode_used"] == "simple"


def test_unknown_mode_rejected():
    assert client.post("/api/ask", json={"question": "hello there", "mode": "magic"}).status_code == 422


def test_cache_is_separate_per_mode(monkeypatch):
    main.answer_cache.clear()
    monkeypatch.setattr(agent, "run", lambda **k: {**FAKE, "which": "agent"})
    monkeypatch.setattr(rag, "answer_question", lambda **k: {**FAKE, "which": "simple"})
    a = client.post("/api/ask", json={"question": "cache me please", "mode": "agent"}).json()
    s = client.post("/api/ask", json={"question": "cache me please", "mode": "simple"}).json()
    assert a["which"] == "agent" and s["which"] == "simple"

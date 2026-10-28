import agent
import config
import rag


def chunk(i, score, doc=1):
    return {"document_id": doc, "filename": f"d{doc}.pdf", "page_number": i, "chunk_text": f"text {i}",
            "similarity_score": score}


STRONG = [chunk(1, 0.9), chunk(2, 0.8), chunk(3, 0.7)]


def fake_json(replies):
    calls = []

    def fake(system, prompt):
        role = system.split()[3]  # "You are the PLANNER ..." -> PLANNER
        calls.append(role)
        v = replies[role]
        return v.pop(0) if isinstance(v, list) else v

    fake.calls = calls
    return fake


def run(**kw):
    return agent.run("q?", kw.get("document_id"), 5, 1, "t")


def test_merge_dedupes_and_orders():
    m = agent.merge_chunks([chunk(1, 0.4), chunk(2, 0.9)], [chunk(1, 0.7)], limit=5)
    assert [(c["page_number"], c["similarity_score"]) for c in m] == [(2, 0.9), (1, 0.7)]
    assert len(agent.merge_chunks([chunk(i, i / 10) for i in range(1, 9)], limit=3)) == 3


def test_plan_fallback_and_cap(monkeypatch):
    monkeypatch.setattr(rag, "ask_json", lambda *a: None)
    assert agent.plan("Fees?", agent.Trace()) == ["Fees?"]
    monkeypatch.setattr(rag, "ask_json", lambda *a: {"questions": list("abcde")})
    assert len(agent.plan("x", agent.Trace())) == config.AGENT_MAX_SUBQUESTIONS


def test_grade_skips_model_when_obvious(monkeypatch):
    monkeypatch.setattr(rag, "ask_json", lambda *a: (_ for _ in ()).throw(AssertionError("no model call")))
    assert agent.grade("q", STRONG, agent.Trace()) is True
    assert agent.grade("q", [], agent.Trace()) is False


def test_no_evidence_means_no_answer_call(monkeypatch):
    monkeypatch.setattr(rag, "ask_json", fake_json({"PLANNER": {"questions": ["q"]}, "SEARCH": {"query": "zz"}}))
    monkeypatch.setattr(rag, "search_chunks", lambda *a: [])
    monkeypatch.setattr(rag, "generate_answer", lambda *a, **k: (_ for _ in ()).throw(AssertionError("no")))
    r = run()
    assert r["retrieval_ok"] is False and r["sources"] == [] and r["answer"] == agent.NOT_FOUND


def test_happy_path(monkeypatch):
    f = fake_json({"PLANNER": {"questions": ["q"]}, "CHECKER": {"unsupported": []}})
    monkeypatch.setattr(rag, "ask_json", f)
    monkeypatch.setattr(rag, "search_chunks", lambda *a: STRONG)
    monkeypatch.setattr(rag, "generate_answer", lambda q, c, **k: "Answer (page 1).")
    r = run()
    assert f.calls == ["PLANNER", "CHECKER"]
    assert r["verified"] is True and r["revised"] is False and r["retrieval_ok"] is True


def test_rewrite_retry_succeeds(monkeypatch):
    seen = []

    def search(query, k, doc, uid):
        seen.append(query)
        return [] if len(seen) == 1 else STRONG

    monkeypatch.setattr(rag, "ask_json", fake_json({"PLANNER": {"questions": ["fees"]},
                                                    "SEARCH": {"query": "tuition cost"}, "CHECKER": {"unsupported": []}}))
    monkeypatch.setattr(rag, "search_chunks", search)
    monkeypatch.setattr(rag, "generate_answer", lambda q, c, **k: "ok")
    r = run()
    assert seen == ["fees", "tuition cost"] and "rewrite" in [s["step"] for s in r["trace"]]


def test_widens_to_other_pdfs_once(monkeypatch):
    scopes = []

    def search(query, k, doc, uid):
        scopes.append(doc)
        return STRONG if doc is None else []

    monkeypatch.setattr(rag, "ask_json", fake_json({"PLANNER": {"questions": ["a", "b"]},
                                                    "SEARCH": {"query": "zzz"}, "CHECKER": {"unsupported": []}}))
    monkeypatch.setattr(rag, "search_chunks", search)
    monkeypatch.setattr(rag, "generate_answer", lambda q, c, **k: "ok")
    r = run(document_id=7)
    assert r["widened"] is True
    assert [s["step"] for s in r["trace"]].count("widen") == 1  # once per question, not per sub-question


def test_no_widen_when_all_pdfs_already_searched(monkeypatch):
    monkeypatch.setattr(rag, "ask_json", fake_json({"PLANNER": {"questions": ["q"]}, "SEARCH": {"query": "zz"}}))
    monkeypatch.setattr(rag, "search_chunks", lambda *a: [])
    assert run(document_id=None)["widened"] is False


def test_widen_can_be_switched_off(monkeypatch):
    monkeypatch.setattr(config, "AGENT_ALLOW_WIDEN", False)
    monkeypatch.setattr(rag, "ask_json", fake_json({"PLANNER": {"questions": ["q"]}, "SEARCH": {"query": "zz"}}))
    monkeypatch.setattr(rag, "search_chunks", lambda *a: [])
    assert run(document_id=7)["widened"] is False


def test_unsupported_claims_trigger_one_revision(monkeypatch):
    monkeypatch.setattr(rag, "ask_json", fake_json({"PLANNER": {"questions": ["q"]},
                                                    "CHECKER": {"unsupported": ["room rent is free"]}}))
    monkeypatch.setattr(rag, "search_chunks", lambda *a: STRONG)
    monkeypatch.setattr(rag, "generate_answer", lambda q, c, **k: "draft")
    got = {}
    monkeypatch.setattr(rag, "revise_answer", lambda q, c, d, p: got.update(p=p) or "fixed")
    r = run()
    assert got["p"] == ["room rent is free"]
    assert r["answer"] == "fixed" and r["revised"] is True and r["verified"] is False


def test_verify_can_be_switched_off(monkeypatch):
    f = fake_json({"PLANNER": {"questions": ["q"]}})
    monkeypatch.setattr(config, "AGENT_VERIFY", False)
    monkeypatch.setattr(rag, "ask_json", f)
    monkeypatch.setattr(rag, "search_chunks", lambda *a: STRONG)
    monkeypatch.setattr(rag, "generate_answer", lambda q, c, **k: "ok")
    assert run()["verified"] is False and f.calls == ["PLANNER"]


def test_parse_json_handles_fences_and_noise():
    assert rag.parse_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert rag.parse_json('Sure! {"a": 2} done') == {"a": 2}
    assert rag.parse_json("nonsense") is None


class _Reply:
    def __init__(self, content): self.content = content


def test_rag_helpers_talk_to_the_model(monkeypatch):
    class FakeJson:
        def invoke(self, msgs): return _Reply('```json\n{"enough": true}\n```')
    class FakeLlm:
        def invoke(self, msgs): return _Reply("fixed answer")
    monkeypatch.setattr(rag, "_llm_json", FakeJson())
    monkeypatch.setattr(rag, "_llm", FakeLlm())
    assert rag.ask_json("sys", "prompt") == {"enough": True}
    assert rag.revise_answer("q", STRONG, "draft", ["bad claim"]) == "fixed answer"

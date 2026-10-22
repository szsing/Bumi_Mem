"""Long-term vs run scope on the REAL pinned Mem0 add() pipeline.

Only the embedder (deterministic hashing) and the extraction LLM (scripted, no
LLM-side dedup) are faked; filters, Qdrant, hash dedup, last-k messages and entity
linking are Mem0 2.0.20's own code. Each case prints its diagnostics (pytest -s).
"""
import json
import os

os.environ.setdefault("MEM0_TELEMETRY", "False")

import hashlib
import math
import threading
from types import SimpleNamespace
from uuid import uuid4

import pytest

pytest.importorskip("qdrant_client")
mem0_main = pytest.importorskip("mem0.memory.main")

from mem0.memory.utils import parse_messages  # noqa: E402

from bumi_mem.long_term import Mem0AdapterConfig, Mem0LongTermMemory  # noqa: E402
from bumi_mem.providers import ProviderConfig  # noqa: E402
from bumi_mem.schemas import MemoryScopeError, MemoryWriteUncertainError, RetryableMemoryError  # noqa: E402
from bumi_mem.session_store import SQLiteSessionStore  # noqa: E402

AGENT = "bumi_robot_001"
DIMS = 64
FACTS = {
    "我一直都不吃香菜": "用户一直不吃香菜",
    "我喜欢蓝色": "用户喜欢蓝色",
    "我现在更喜欢绿色": "用户现在更喜欢绿色",
    "我特别喜欢香菜": "用户特别喜欢香菜",
    "我不吃香菜": "用户不吃香菜",
    "我周末常去公园": "用户周末常去公园",
    "今天聊点别的": None,
}


def _vector(text):
    v = [0.0] * DIMS
    v[0] = 1.0  # never a zero vector
    for i in range(len(text) - 1):
        v[1 + int(hashlib.md5(text[i:i + 2].encode()).hexdigest(), 16) % (DIMS - 1)] += 1.0
    norm = math.sqrt(sum(x * x for x in v))
    return [x / norm for x in v]


class FakeEmbedder:
    def embed(self, text, memory_action=None):
        return _vector(text)

    def embed_batch(self, texts, memory_action="add"):
        return [_vector(t) for t in texts]


def _section(prompt, name):
    head = f"## {name}\n"
    if head not in prompt:
        return ""
    return prompt.split(head, 1)[1].split("\n\n## ", 1)[0].strip()


class ScriptedExtractor:
    """Deterministic extractor that NEVER dedups by itself: any duplicate prevention
    observed in these tests comes from Mem0's own existing-memory scope + hash dedup."""

    def __init__(self):
        self.calls = []

    def __call__(self, messages, response_format=None, **kwargs):
        prompt = messages[1]["content"]
        existing = [m["text"] for m in json.loads(_section(prompt, "Existing Memories") or "[]")]
        last_k = [line for line in _section(prompt, "Last k Messages").splitlines() if line.strip()]
        new = _section(prompt, "New Messages")
        self.calls.append({"existing_memories": existing, "last_messages": last_k})
        texts = [fact for said, fact in FACTS.items() if fact and said in new]
        return json.dumps({"memory": [{"id": str(i), "text": t, "attributed_to": "user"}
                                      for i, t in enumerate(texts)]}, ensure_ascii=False)


@pytest.fixture
def real(tmp_path, monkeypatch):
    monkeypatch.delenv("BUMI_QDRANT_URL", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    # The fake endpoint never performs HTTP. Do not let developer/CI proxy
    # configuration make OpenAI client construction require optional socksio.
    for name in ("ALL_PROXY", "HTTPS_PROXY", "HTTP_PROXY", "all_proxy", "https_proxy", "http_proxy"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr("mem0.memory.telemetry.MEM0_TELEMETRY", False)
    monkeypatch.setattr(mem0_main, "EmbedderFactory", SimpleNamespace(create=lambda *a, **k: FakeEmbedder()))
    adapter = Mem0LongTermMemory(
        ProviderConfig("fake", "fake-model", "http://127.0.0.1:9", "fake-key", temperature=0.0),
        Mem0AdapterConfig(data_dir=tmp_path / "l3", embed_model=tmp_path / "model",
                          embed_device="cpu", embedding_dims=DIMS))
    extractor = ScriptedExtractor()
    adapter.memory.llm.generate_response = extractor
    yield adapter, extractor
    adapter.close()


def seed_memory(adapter, *, user_id, text, query_text, generation=None,
                source_message_ids=None, vector=None, run_id="seed-run"):
    """Insert a controlled raw candidate without involving extraction."""
    memory_id = str(uuid4())
    payload = {"data": text, "hash": hashlib.md5(text.encode()).hexdigest(),
               "user_id": user_id, "agent_id": AGENT, "run_id": run_id}
    if generation is not None:
        payload["memory_generation"] = generation
    if source_message_ids is not None:
        payload["source_message_ids"] = list(source_message_ids)
    query_vector = vector or _vector(parse_messages([{"role": "user", "content": query_text}]))
    adapter.memory.vector_store.insert(vectors=[query_vector], ids=[memory_id], payloads=[payload])
    return memory_id


def consolidate(adapter, extractor, user_id, chunk_id, text, *, generation=1,
                source_message_ids=None, **kwargs):
    """One production-shaped add_session(); returns that call's diagnostics."""
    metadata = {"source": "l1_session", "consolidation_id": chunk_id,
                "memory_generation": generation}
    if source_message_ids is not None:
        metadata["source_message_ids"] = list(source_message_ids)
    events = adapter.add_session(
        user_id=user_id, agent_id=AGENT, messages=[{"role": "user", "content": text}],
        metadata=metadata, prompt="", **kwargs)
    seen = extractor.calls[-1]
    scope = adapter.last_scope_diagnostics or {}
    return {"user_id": user_id, "agent_id": AGENT,
            "run_id": f"bumi-generation-{generation}-chunk-{chunk_id}",
            "existing_memories": seen["existing_memories"], "last_messages": seen["last_messages"],
            "existing_scope": scope.get("existing_scope"),
            "existing_records": [{k: r[k] for k in ("user_id", "run_id", "memory")}
                                 for r in scope.get("existing_memories", [])],
            "hook_hits": scope.get("hook_hits", {}),
            "requested_top_k": scope.get("requested_top_k"),
            "fetched_top_k": scope.get("fetched_top_k"),
            "filtered_count": scope.get("filtered_count"),
            "returned_count": scope.get("returned_count"),
            "written": [e["memory"] for e in events]}


def stored(adapter, user_id):
    return sorted(r["memory"] for r in adapter.get_all(user_id=user_id, agent_id=AGENT))


@pytest.mark.parametrize('structured', [False, True])
def test_fact_sources_are_separate_before_embedding_hash_and_persistence(real, structured):
    adapter, _ = real
    text = 'Jon plans a fair in May 2023'
    calls = 0

    def extraction(**kwargs):
        nonlocal calls
        calls += 1
        source = f'source-{calls}'
        item = {'id':'0','text':text, 'attributed_to':'user'}
        if structured:
            item['source_message_ids'] = [source]
        else:
            item['text'] += f' (source_message_id: {source})'
        return json.dumps({'memory':[item]})

    adapter.memory.llm.generate_response = extraction
    original_insert = adapter.memory.vector_store.insert
    for index in (1,2):
        source = f'source-{index}'
        events = adapter.add_session(user_id='Jon',agent_id=AGENT,
            messages=[{'role':'user','content':'I plan a fair next month.'}],
            metadata={'source':'l1_session','consolidation_id':f'chunk-{index}',
                'source_message_ids':[source,'other-source'],
                'source_messages':[{'source_message_id':source,'observed_at':'2023-04-03T12:00:00+00:00'},
                    {'source_message_id':'other-source','observed_at':'2023-04-04T12:00:00+00:00'}]},prompt='')
        assert [e['memory'] for e in events] == ([text] if index==1 else [])
        assert adapter.memory.llm.generate_response is extraction
        assert adapter.memory.vector_store.insert == original_insert
    rows = adapter.get_all(user_id='Jon',agent_id=AGENT)
    assert len(rows)==1
    row=rows[0]
    assert row['memory']==text
    assert row['metadata']['source_message_ids']==['source-1','other-source']
    assert row['metadata']['fact_source_message_ids']==['source-1']
    assert row['metadata']['fact_source_messages']==[
        {'source_message_id':'source-1','observed_at':'2023-04-03T12:00:00+00:00'}]
    point=adapter.memory.vector_store.client.retrieve(collection_name='bumi_mem_l3',
        ids=[row['id']],with_vectors=True)[0]
    assert point.payload['hash']==hashlib.md5(text.encode()).hexdigest()
    dense = point.vector.get('') if isinstance(point.vector, dict) else point.vector
    assert dense==pytest.approx(_vector(text))


@pytest.mark.parametrize('invalid', ['foreign_citation','inline_anchor','unknown_inline'])
def test_invalid_provenance_is_rejected_before_write_and_hooks_are_restored(real, invalid):
    adapter, _ = real
    def extraction(**kwargs):
        if invalid == 'foreign_citation':
            item = {'id':'0','text':'a fact','source_message_ids':['another-users-source']}
        elif invalid == 'unknown_inline':
            item = {'id':'0','text':'a fact (source_message_id: hallucinated-source)'}
        else:
            item = {'id':'0','text':'a fact from own-source'}
        return json.dumps({'memory':[item]})
    adapter.memory.llm.generate_response=extraction
    original_insert=adapter.memory.vector_store.insert
    with pytest.raises(RetryableMemoryError,match='PROVENANCE_'):
        adapter.add_session(user_id='Jon',agent_id=AGENT,messages=[{'role':'user','content':'a fact'}],
            metadata={'consolidation_id':'bad-citation','source_message_ids':['own-source']},prompt='')
    assert adapter.get_all(user_id='Jon',agent_id=AGENT)==[]
    assert adapter.memory.llm.generate_response is extraction
    assert adapter.memory.vector_store.insert==original_insert
    assert adapter._scope.active is None


def show(case, *diagnostics):
    print(f"\n[{case}] " + json.dumps(list(diagnostics), ensure_ascii=False, indent=1))


def test_case1_same_speaker_cross_session_duplicate(real):
    adapter, extractor = real
    a = consolidate(adapter, extractor, "dad", "job-A:chunk:0", "我一直都不吃香菜")
    b = consolidate(adapter, extractor, "dad", "job-B:chunk:0", "我一直都不吃香菜")
    show("case1", a, b)
    assert a["run_id"] != b["run_id"]
    assert "用户一直不吃香菜" in b["existing_memories"]
    assert b["existing_records"] == [{"user_id": "dad", "run_id": a["run_id"], "memory": "用户一直不吃香菜"}]
    assert a["hook_hits"]["vector_search"] == b["hook_hits"]["vector_search"] == 1
    assert set(b["hook_hits"]) == {"vector_search", "entity_list", "entity_search_batch"}
    assert b["written"] == []
    assert stored(adapter, "dad") == ["用户一直不吃香菜"]


def test_case2_same_speaker_cross_session_state_change(real):
    adapter, extractor = real
    a = consolidate(adapter, extractor, "kid", "job-A:chunk:0", "我喜欢蓝色")
    b = consolidate(adapter, extractor, "kid", "job-B:chunk:0", "我现在更喜欢绿色")
    show("case2", a, b)
    assert "用户喜欢蓝色" in b["existing_memories"]
    assert b["written"] == ["用户现在更喜欢绿色"]  # ADD-only lifecycle is out of scope


def test_case3_no_cross_speaker_contamination(real):
    adapter, extractor = real
    rows = [consolidate(adapter, extractor, "dad", "job-D1:chunk:0", "我不吃香菜"),
            consolidate(adapter, extractor, "mom", "job-M1:chunk:0", "我特别喜欢香菜"),
            consolidate(adapter, extractor, "dad", "job-D2:chunk:0", "我不吃香菜"),
            consolidate(adapter, extractor, "mom", "job-M2:chunk:0", "我特别喜欢香菜"),
            consolidate(adapter, extractor, "kid", "job-K1:chunk:0", "我不吃香菜")]
    show("case3", *rows)
    contamination = sum(rec["user_id"] != row["user_id"] for row in rows for rec in row["existing_records"])
    print(f"[case3] cross-speaker contamination = {contamination}")
    assert contamination == 0
    assert all(row["existing_scope"] == {"user_id": row["user_id"], "agent_id": AGENT} for row in rows)
    assert [len(r["existing_records"]) for r in rows] == [0, 0, 1, 1, 0]
    assert "用户特别喜欢香菜" not in rows[2]["existing_memories"]   # dad never sees mom
    assert "用户不吃香菜" not in rows[3]["existing_memories"]       # mom never sees dad
    assert rows[4]["existing_memories"] == []                         # kid's first write sees nobody
    assert stored(adapter, "dad") == ["用户不吃香菜"]
    assert stored(adapter, "mom") == ["用户特别喜欢香菜"]
    assert stored(adapter, "kid") == ["用户不吃香菜"]


def test_case4_last_messages_stay_in_current_run(real):
    adapter, extractor = real
    a1 = consolidate(adapter, extractor, "mom", "job-A:chunk:0", "我周末常去公园")
    a2 = consolidate(adapter, extractor, "mom", "job-A:chunk:0", "今天聊点别的")   # same run
    b = consolidate(adapter, extractor, "mom", "job-B:chunk:0", "今天聊点别的")    # new session
    show("case4", a1, a2, b)
    assert "user: 我周末常去公园" in a2["last_messages"]          # same run keeps its raw context
    assert "用户周末常去公园" in b["existing_memories"]            # long-term memory crosses runs
    assert b["existing_records"][0]["run_id"] == a1["run_id"]
    assert b["last_messages"] == []                                 # no raw chat from session A


def test_case5_same_run_retry_is_idempotent(real):
    adapter, extractor = real
    first = consolidate(adapter, extractor, "dad", "job-R:chunk:0", "我一直都不吃香菜")
    retry = consolidate(adapter, extractor, "dad", "job-R:chunk:0", "我一直都不吃香菜")
    show("case5", first, retry)
    assert first["run_id"] == retry["run_id"]
    assert "用户一直不吃香菜" in retry["existing_memories"]
    assert retry["written"] == []
    assert stored(adapter, "dad") == ["用户一直不吃香菜"]


def test_withdrawn_memory_is_not_extraction_context_and_does_not_block_restatement(real):
    adapter, extractor = real
    consolidate(adapter, extractor, "dad", "job-A:chunk:0", "我一直都不吃香菜")
    withdrawn = {r["id"] for r in adapter.get_all(user_id="dad", agent_id=AGENT)}
    screen = lambda rows: [r for r in rows if r["id"] not in withdrawn]  # store.filter_memory_rows stand-in
    b = consolidate(adapter, extractor, "dad", "job-B:chunk:0", "我一直都不吃香菜", existing_memory_filter=screen)
    show("withdrawn", b)
    assert b["existing_memories"] == [] and adapter.last_scope_diagnostics["screened_out"] == 1
    assert b["written"] == ["用户一直不吃香菜"]   # hash dedup no longer sees the withdrawn copy


def test_other_generation_memory_is_not_extraction_context(real):
    adapter, extractor = real
    adapter.add_session(user_id="dad", agent_id=AGENT, messages=[{"role": "user", "content": "我一直都不吃香菜"}],
                        metadata={"consolidation_id": "old:chunk:0", "memory_generation": 0}, prompt="")
    b = consolidate(adapter, extractor, "dad", "job-B:chunk:0", "我一直都不吃香菜")
    assert b["existing_memories"] == [] and b["written"] == ["用户一直不吃香菜"]


def test_filtered_top_ten_refills_to_find_legal_eleventh_memory(real):
    adapter, extractor = real
    probe = "今天聊点别的"
    query_vector = _vector(parse_messages([{"role": "user", "content": probe}]))
    for index in range(10):
        seed_memory(adapter, user_id="dad", text=f"旧 epoch 记录 {index}", query_text=probe,
                    generation=0, vector=query_vector, run_id=f"old-{index}")
    legal_vector = list(query_vector)
    zero_index = next(i for i, value in enumerate(legal_vector) if value == 0.0)
    legal_vector[zero_index] = 0.2
    norm = math.sqrt(sum(value * value for value in legal_vector))
    legal_vector = [value / norm for value in legal_vector]
    seed_memory(adapter, user_id="dad", text="合法历史偏好", query_text=probe,
                generation=1, vector=legal_vector, run_id="legal-run")

    result = consolidate(adapter, extractor, "dad", "job-refill:chunk:0", probe, generation=1)

    assert result["existing_memories"] == ["合法历史偏好"]
    assert result["requested_top_k"] == 10
    assert 10 < result["fetched_top_k"] <= 100
    assert result["filtered_count"] == 10
    assert result["returned_count"] == 1


@pytest.mark.parametrize(
    ("active_generation", "row_generation", "visible"),
    [(0, None, True), (1, None, False), (1, 0, False), (1, 1, True)],
)
def test_generation_epoch_legacy_visibility(real, active_generation, row_generation, visible):
    adapter, extractor = real
    probe = "今天聊点别的"
    seed_memory(adapter, user_id="dad", text="受 generation 控制的历史", query_text=probe,
                generation=row_generation)

    result = consolidate(adapter, extractor, "dad", "job-generation:chunk:0", probe,
                         generation=active_generation)

    assert ("受 generation 控制的历史" in result["existing_memories"]) is visible
    assert result["returned_count"] == int(visible)


@pytest.mark.parametrize("row_generation", [1.5, "1", True])
def test_malformed_row_generation_is_not_visible(real, row_generation):
    adapter, extractor = real
    probe = "今天聊点别的"
    seed_memory(adapter, user_id="dad", text="畸形 generation 历史", query_text=probe,
                generation=row_generation)

    result = consolidate(adapter, extractor, "dad", "job-malformed-row:chunk:0", probe, generation=1)

    assert "畸形 generation 历史" not in result["existing_memories"]


def test_active_generation_requires_non_negative_integer(real):
    adapter, _ = real
    for generation in (1.5, "1", True, -1):
        with pytest.raises(ValueError, match="non-negative integer"):
            adapter.add_session(
                user_id="dad", agent_id=AGENT, messages=[{"role": "user", "content": "测试"}],
                metadata={"consolidation_id": "job-invalid-generation:chunk:0",
                          "memory_generation": generation}, prompt="",
            )
        assert adapter.last_scope_diagnostics is None


def test_source_blocked_candidate_is_filtered_by_real_store_before_extraction(real, tmp_path):
    adapter, extractor = real
    store = SQLiteSessionStore(tmp_path / "scope.sqlite3")
    session = store.create_session("dad", AGENT)
    source = store.append_message(session.session_id, role="user", content="我一直都不吃香菜", token_count=8)
    consolidate(adapter, extractor, "dad", "job-A:chunk:0", "我一直都不吃香菜",
                source_message_ids=[source.message_id])
    original = adapter.get_all(user_id="dad", agent_id=AGENT)[0]
    store.request_forget_memory(user_id="dad", agent_id=AGENT, memory=original)

    # Simulate a late/different backend record from the withdrawn source. Its ID
    # is not tombstoned, so only blocked_memory_sources can reject it.
    late_id = seed_memory(
        adapter, user_id="dad", text="同一撤回来源的迟到记录", query_text="我一直都不吃香菜",
        generation=1, source_message_ids=[source.message_id], run_id="late-run",
    )
    screened_ids = []

    def screen(rows):
        screened_ids.extend(row["id"] for row in rows)
        return store.filter_memory_rows(rows, user_id="dad", agent_id=AGENT)

    result = consolidate(
        adapter, extractor, "dad", "job-B:chunk:0", "我一直都不吃香菜",
        source_message_ids=["new-source"], existing_memory_filter=screen,
    )

    assert late_id in screened_ids
    assert "同一撤回来源的迟到记录" not in result["existing_memories"]
    assert result["existing_memories"] == []
    assert result["filtered_count"] >= 1


def test_missing_vector_hook_fails_fast_and_clears_previous_diagnostics(real, monkeypatch):
    adapter, extractor = real
    consolidate(adapter, extractor, "dad", "job-success:chunk:0", "今天聊点别的")
    previous = adapter.last_scope_diagnostics
    assert previous is not None and previous["hook_hits"]["vector_search"] == 1

    monkeypatch.setattr(adapter.memory, "add", lambda *args, **kwargs: {"results": []})
    with pytest.raises(MemoryWriteUncertainError, match="vector search hook"):
        adapter.add_session(
            user_id="dad", agent_id=AGENT, messages=[{"role": "user", "content": "第二次"}],
            metadata={"consolidation_id": "job-failure:chunk:0", "memory_generation": 1}, prompt="",
        )

    assert adapter.last_scope_diagnostics is None


def test_direct_adapter_concurrent_scope_activation_fails_fast(real):
    adapter, _ = real
    entered = threading.Event()
    release = threading.Event()
    extractor_calls = []
    first_errors = []

    def blocking_extractor(messages, response_format=None, **kwargs):
        extractor_calls.append(messages)
        entered.set()
        if not release.wait(5):
            raise RuntimeError("test did not release the first extraction")
        return json.dumps({"memory": []})

    adapter.memory.llm.generate_response = blocking_extractor

    def first_call():
        try:
            adapter.add_session(
                user_id="dad", agent_id=AGENT, messages=[{"role": "user", "content": "第一条"}],
                metadata={"consolidation_id": "job-A:chunk:0", "memory_generation": 1}, prompt="",
            )
        except Exception as exc:  # pragma: no cover - asserted below
            first_errors.append(exc)

    thread = threading.Thread(target=first_call)
    thread.start()
    assert entered.wait(5)
    try:
        with pytest.raises(MemoryWriteUncertainError, match="requires serialized access"):
            adapter.add_session(
                user_id="mom", agent_id=AGENT, messages=[{"role": "user", "content": "第二条"}],
                metadata={"consolidation_id": "job-B:chunk:0", "memory_generation": 1}, prompt="",
            )
        with pytest.raises(MemoryScopeError, match="requires serialized access"):
            adapter.delete_all(user_id="mom", agent_id=AGENT)
        assert adapter._scope.active.user_id == "dad"
        assert len(extractor_calls) == 1
    finally:
        release.set()
        thread.join(timeout=5)

    assert not thread.is_alive()
    assert first_errors == []
    assert adapter._scope.active is None
    assert adapter.last_scope_diagnostics["user_id"] == "dad"


def test_entities_link_across_runs_and_forget_all_leaves_no_entity_residue(real):
    adapter, extractor = real
    FACTS["I moved to Paris"] = "User moved to Paris"
    FACTS["I still live in Paris"] = "User still lives in Paris"
    try:
        consolidate(adapter, extractor, "dad", "job-A:chunk:0", "I moved to Paris")
        consolidate(adapter, extractor, "dad", "job-B:chunk:0", "I still live in Paris")
        consolidate(adapter, extractor, "mom", "job-M:chunk:0", "I moved to Paris")
    finally:
        del FACTS["I moved to Paris"], FACTS["I still live in Paris"]
    entities = adapter.memory.entity_store

    def rows(user):
        listed = entities.list(filters={"user_id": user, "agent_id": AGENT}, top_k=100)
        return listed[0] if listed and isinstance(listed[0], list) else listed

    paris = [r for r in rows("dad") if r.payload.get("data", "").lower() == "paris"]
    if not paris:
        pytest.skip("spaCy NER model unavailable: no entities extracted")
    assert len(paris) == 1 and len(paris[0].payload["linked_memory_ids"]) == 2  # one entity, both runs
    adapter.delete_all(user_id="dad", agent_id=AGENT)
    assert rows("dad") == []
    assert [r.payload["data"].lower() for r in rows("mom")] == ["paris"]      # other speaker untouched


@pytest.mark.parametrize("foreign_owner", [("mom", AGENT), ("dad", "other-agent")])
def test_entity_cleanup_rejects_foreign_provider_row_before_delete(real, monkeypatch, foreign_owner):
    adapter, _ = real
    entities = adapter.memory.entity_store
    memory_id = seed_memory(
        adapter, user_id="dad", text="待删除记忆", query_text="待删除记忆", generation=1,
    )
    owned = SimpleNamespace(
        id=str(uuid4()),
        payload={"data": "owned", "user_id": "dad", "agent_id": AGENT,
                 "linked_memory_ids": [memory_id]},
    )
    foreign = SimpleNamespace(
        id=str(uuid4()),
        payload={"data": "foreign", "user_id": foreign_owner[0], "agent_id": foreign_owner[1],
                 "linked_memory_ids": [memory_id]},
    )
    deleted = []
    updated = []
    list_calls = []

    def bad_list(**kwargs):
        list_calls.append(kwargs)
        if "run_id" in kwargs["filters"]:
            return ([owned, foreign], None)
        return ([], None)

    monkeypatch.setattr(entities, "list", bad_list)
    monkeypatch.setattr(entities, "delete", lambda *, vector_id: deleted.append(vector_id))
    monkeypatch.setattr(entities, "update", lambda **kwargs: updated.append(kwargs))

    with pytest.raises(MemoryScopeError, match="entity cleanup crossed"):
        adapter.delete_all(user_id="dad", agent_id=AGENT)

    assert any("run_id" in call["filters"] for call in list_calls)  # Mem0's internal cleanup was guarded.
    assert deleted == []
    assert updated == []


def test_consolidator_passes_store_withdrawal_filter(tmp_path):
    from test_bumi_memory import chat_turn, make_engine, submit
    engine, store, fake, _, _ = make_engine(tmp_path)
    chat_turn(engine, "我一直都不吃香菜")
    receipt = submit(engine)
    assert engine.worker.run_once() and store.get_job(receipt.job_id).status == "succeeded"
    screen = fake.add_calls[-1]["existing_memory_filter"]
    with store._connect(write=True) as conn:
        conn.execute("INSERT INTO memory_tombstones VALUES(?,?,?,?)", ("u1", "a1", "gone", "now"))
    rows = [{"id": "gone", "memory": "x", "metadata": {}}, {"id": "kept", "memory": "y", "metadata": {}}]
    assert [r["id"] for r in screen(rows)] == ["kept"]

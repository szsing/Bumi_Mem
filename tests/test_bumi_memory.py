from __future__ import annotations

from pathlib import Path

import pytest

from bumi_mem.consolidator import ConsolidatorConfig, SessionConsolidator
from bumi_mem.engine import BumiMemory, BumiMemoryConfig
from bumi_mem.long_term import FakeLongTermMemory
from bumi_mem.providers import FakeTextProvider
from bumi_mem.schemas import SessionBlockedError
from bumi_mem.session_store import SQLiteSessionStore
from bumi_mem.token_counter import FixedTokenCounter


def make_engine(
    tmp_path: Path,
    *,
    max_messages: int = 20,
    max_tokens: int = 8000,
    memory_provider: str = "deepseek",
    local_chunk_tokens: int = 3500,
):
    store = SQLiteSessionStore(tmp_path / "bumi.sqlite3")
    long_term = FakeLongTermMemory()
    chat = FakeTextProvider("chat")
    summary = FakeTextProvider("memory", ["测试会话摘要"] * 50)
    counter = FixedTokenCounter(1)
    consolidator = SessionConsolidator(
        store=store,
        long_term=long_term,
        summary_provider=summary,
        memory_provider_name=memory_provider,
        memory_token_counter=counter,
        config=ConsolidatorConfig(
            carryover_messages=4,
            local_chunk_tokens=local_chunk_tokens,
        ),
    )
    engine = BumiMemory(
        store=store,
        long_term=long_term,
        chat_provider=chat,
        consolidator=consolidator,
        chat_token_counter=counter,
        config=BumiMemoryConfig(
            max_messages=max_messages,
            max_l1_tokens=max_tokens,
            top_k=5,
            search_threshold=0.1,
            max_answer_tokens=128,
        ),
    )
    return engine, store, long_term, chat, summary


def chat_turn(engine: BumiMemory, text: str = "你好", user: str = "u1", agent: str = "a1"):
    return engine.chat(user_id=user, agent_id=agent, text=text)


def test_first_nineteen_messages_do_not_write_l3(tmp_path: Path) -> None:
    engine, store, long_term, _, _ = make_engine(tmp_path)
    for index in range(9):
        result = chat_turn(engine, f"第{index}轮")
        assert result.auto_consolidation is None
    session = store.get_open_session("u1", "a1")
    assert session is not None
    assert store.stats(session.session_id).effective_messages == 18
    assert long_term.add_calls == []


def test_twentieth_message_answers_then_auto_consolidates(tmp_path: Path) -> None:
    engine, store, long_term, _, _ = make_engine(tmp_path)
    observed: list[str] = []
    result = None
    for index in range(10):
        result = engine.chat(
            user_id="u1",
            agent_id="a1",
            text=f"第{index}轮",
            on_answer=lambda answer: observed.append(answer),
        )
    assert result is not None and result.auto_consolidation is not None
    assert result.auto_consolidation.success
    assert observed and len(observed) == 10
    assert len(long_term.add_calls) == 1
    new_stats = store.stats(result.active_session_id)
    assert new_stats.effective_messages == 0
    assert new_stats.context_only_messages == 4
    assert store.get_session(result.session_id).status == "consolidated"  # type: ignore[union-attr]


def test_token_limit_consolidates_after_answer(tmp_path: Path) -> None:
    engine, _, long_term, _, _ = make_engine(tmp_path, max_messages=20, max_tokens=10)
    shown: list[str] = []
    result = engine.chat(
        user_id="u1", agent_id="a1", text="达到 token 限制", on_answer=shown.append
    )
    assert shown == [result.answer]
    assert result.auto_consolidation is not None
    assert result.auto_consolidation.reason == "token_limit"
    assert len(long_term.add_calls) == 1


def test_manual_new_consolidates_early(tmp_path: Path) -> None:
    engine, store, long_term, _, _ = make_engine(tmp_path)
    chat_turn(engine, "我喜欢红茶")
    old = store.get_open_session("u1", "a1")
    result = engine.end_session(user_id="u1", agent_id="a1")
    assert result.success and result.new_session_id
    assert result.session_id == old.session_id  # type: ignore[union-attr]
    assert len(long_term.add_calls) == 1


def test_current_query_only_is_used_for_l3_search(tmp_path: Path) -> None:
    engine, _, long_term, _, _ = make_engine(tmp_path)
    chat_turn(engine, "第一件事")
    chat_turn(engine, "那后来呢")
    assert [call["query"] for call in long_term.search_calls] == ["第一件事", "那后来呢"]
    assert long_term.search_calls[-1]["top_k"] == 5
    assert long_term.search_calls[-1]["threshold"] == 0.1


def test_current_user_message_appears_once_in_prompt(tmp_path: Path) -> None:
    engine, _, _, _, _ = make_engine(tmp_path)
    result = chat_turn(engine, "唯一当前消息")
    contents = [m["content"] for m in result.prompt_messages if m["role"] == "user"]
    assert contents.count("唯一当前消息") == 1
    assert result.prompt_messages[0]["role"] == "system"


def test_l3_failure_keeps_l1_and_clears_successful_l2(tmp_path: Path) -> None:
    engine, store, long_term, _, _ = make_engine(tmp_path)
    chat_turn(engine, "需要保存的事实")
    session = store.get_open_session("u1", "a1")
    long_term.fail_add_count = 1
    result = engine.end_session(user_id="u1", agent_id="a1")
    assert not result.success
    failed = store.get_session(session.session_id)  # type: ignore[union-attr]
    assert failed is not None and failed.status == "failed"
    assert failed.summary is None
    assert failed.summary_status == "pending"
    assert store.stats(failed.session_id).effective_messages == 2


def test_retry_reruns_l2_and_l3_from_scratch(tmp_path: Path) -> None:
    engine, store, long_term, _, summary = make_engine(tmp_path)
    chat_turn(engine, "需要重试")
    long_term.fail_add_count = 1
    first = engine.end_session(user_id="u1", agent_id="a1")
    assert not first.success
    second = engine.end_session(user_id="u1", agent_id="a1")
    assert second.success
    assert len(summary.calls) == 2
    assert len(long_term.add_calls) == 2
    runs = store.list_consolidation_runs(first.session_id)
    assert {row["attempt"] for row in runs} == {1, 2}


def test_l2_failure_l3_success_uses_fallback(tmp_path: Path) -> None:
    engine, store, _, _, summary = make_engine(tmp_path)
    chat_turn(engine, "用户信息")
    summary.error = RuntimeError("summary unavailable")
    result = engine.end_session(user_id="u1", agent_id="a1")
    assert result.success
    assert result.summary_status == "fallback"
    old = store.get_session(result.session_id)
    assert old is not None
    assert old.summary_status == "fallback"
    assert "用户消息" in (old.summary or "")


def test_failed_session_blocks_new_chat_until_resolved(tmp_path: Path) -> None:
    engine, _, long_term, _, _ = make_engine(tmp_path)
    chat_turn(engine, "事实")
    long_term.fail_add_count = 1
    assert not engine.end_session(user_id="u1", agent_id="a1").success
    with pytest.raises(SessionBlockedError):
        chat_turn(engine, "不应继续")


def test_restart_restores_active_session(tmp_path: Path) -> None:
    engine, store, _, _, _ = make_engine(tmp_path)
    chat_turn(engine, "重启前")
    original = store.get_open_session("u1", "a1")
    reopened = SQLiteSessionStore(tmp_path / "bumi.sqlite3")
    restored = reopened.get_or_create_session("u1", "a1")
    assert original is not None
    assert restored.session_id == original.session_id
    assert reopened.stats(restored.session_id).effective_messages == 2


def test_carryover_is_context_only_and_not_reconsolidated(tmp_path: Path) -> None:
    engine, store, long_term, _, _ = make_engine(tmp_path, max_messages=2)
    first = chat_turn(engine, "形成一个完整轮")
    assert first.auto_consolidation and first.auto_consolidation.success
    stats = store.stats(first.active_session_id)
    assert stats.effective_messages == 0
    assert stats.context_only_messages == 2
    second = engine.end_session(user_id="u1", agent_id="a1")
    assert second.success
    assert len(long_term.add_calls) == 1


def test_discard_never_writes_l3_and_has_no_carryover(tmp_path: Path) -> None:
    engine, store, long_term, _, _ = make_engine(tmp_path)
    chat_turn(engine, "不要保存")
    new_session = engine.discard_session(user_id="u1", agent_id="a1")
    assert long_term.add_calls == []
    stats = store.stats(new_session.session_id)
    assert stats.effective_messages == 0
    assert stats.context_only_messages == 0


def test_user_agent_pairs_are_isolated(tmp_path: Path) -> None:
    engine, store, _, _, _ = make_engine(tmp_path)
    chat_turn(engine, "用户A", user="A", agent="robot")
    chat_turn(engine, "用户B", user="B", agent="robot")
    a = store.get_open_session("A", "robot")
    b = store.get_open_session("B", "robot")
    assert a is not None and b is not None and a.session_id != b.session_id


def test_forget_all_only_targets_current_pair(tmp_path: Path) -> None:
    engine, _, long_term, _, _ = make_engine(tmp_path)
    long_term.memories[("A", "robot")] = [{"memory": "A"}]
    long_term.memories[("B", "robot")] = [{"memory": "B"}]
    engine.forget_all(user_id="A", agent_id="robot")
    assert engine.get_memories(user_id="A", agent_id="robot") == []
    assert engine.get_memories(user_id="B", agent_id="robot")


def test_answer_callback_runs_before_automatic_l3_write(tmp_path: Path) -> None:
    engine, _, long_term, _, _ = make_engine(tmp_path, max_messages=2)
    order: list[str] = []
    original_add = long_term.add_session

    def observed_add(**kwargs):
        order.append("add")
        return original_add(**kwargs)

    long_term.add_session = observed_add  # type: ignore[method-assign]
    engine.chat(
        user_id="u1", agent_id="a1", text="先回答", on_answer=lambda _: order.append("answer")
    )
    assert order == ["answer", "add"]


def test_l1_current_statement_is_later_than_conflicting_l3(tmp_path: Path) -> None:
    engine, _, long_term, _, _ = make_engine(tmp_path)
    long_term.search_results = [{"memory": "[用户偏好] 用户喜欢长回答"}]
    result = chat_turn(engine, "我现在更喜欢简短回答")
    assert "用户喜欢长回答" in result.prompt_messages[0]["content"]
    assert result.prompt_messages[-1] == {"role": "user", "content": "我现在更喜欢简短回答"}


def test_new_on_empty_session_rotates_without_l3_call(tmp_path: Path) -> None:
    engine, store, long_term, _, _ = make_engine(tmp_path)
    original = engine.get_or_create_session(user_id="u1", agent_id="a1")
    result = engine.end_session(user_id="u1", agent_id="a1")
    assert result.success
    assert result.session_id == original.session_id
    assert result.new_session_id != original.session_id
    assert long_term.add_calls == []
    assert store.get_session(original.session_id).status == "consolidated"  # type: ignore[union-attr]


def test_summary_receives_complete_raw_session(tmp_path: Path) -> None:
    engine, _, _, _, summary = make_engine(tmp_path)
    chat_turn(engine, "第一条用户事实")
    chat_turn(engine, "第二条用户事实")
    result = engine.end_session(user_id="u1", agent_id="a1")
    assert result.success
    prompt = summary.calls[0][-1]["content"]
    assert "第一条用户事实" in prompt
    assert "第二条用户事实" in prompt
    assert "测试回答" in prompt


def test_l3_metadata_has_session_attempt_and_chunk_identity(tmp_path: Path) -> None:
    engine, _, long_term, _, _ = make_engine(tmp_path)
    chat_turn(engine, "带来源的事实")
    result = engine.end_session(user_id="u1", agent_id="a1")
    metadata = long_term.add_calls[0]["metadata"]
    assert metadata["source"] == "l1_session"
    assert metadata["session_id"] == result.session_id
    assert metadata["consolidation_attempt"] == 1
    assert metadata["chunk_index"] == 0
    assert metadata["chunk_count"] == 1
    assert result.session_id in metadata["consolidation_id"]


def test_failed_l3_run_is_recorded_for_audit(tmp_path: Path) -> None:
    engine, store, long_term, _, _ = make_engine(tmp_path)
    chat_turn(engine, "失败审计")
    long_term.fail_add_count = 1
    result = engine.end_session(user_id="u1", agent_id="a1")
    runs = store.list_consolidation_runs(result.session_id)
    assert len(runs) == 1
    assert runs[0]["status"] == "failed"
    assert "fake Mem0 failure" in str(runs[0]["error"])


def test_search_scope_contains_both_user_and_agent(tmp_path: Path) -> None:
    engine, _, long_term, _, _ = make_engine(tmp_path)
    engine.chat(user_id="specific-user", agent_id="specific-agent", text="查询")
    assert long_term.search_calls[-1]["user_id"] == "specific-user"
    assert long_term.search_calls[-1]["agent_id"] == "specific-agent"

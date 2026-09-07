from __future__ import annotations

from pathlib import Path

from bumi_mem.consolidator import ConsolidatorConfig, SessionConsolidator
from bumi_mem.context import ContextBuilder
from bumi_mem.long_term import FakeLongTermMemory
from bumi_mem.prompts import MEMORY_EXTRACTION_INSTRUCTIONS
from bumi_mem.providers import FakeTextProvider
from bumi_mem.schemas import MessageRecord
from bumi_mem.session_store import SQLiteSessionStore
from bumi_mem.token_counter import FixedTokenCounter


def msg(seq: int, role: str, tokens: int, *, context_only: bool = False) -> MessageRecord:
    return MessageRecord(
        message_id=str(seq),
        session_id="s",
        sequence_no=seq,
        role=role,
        content=f"m{seq}",
        token_count=tokens,
        context_only=context_only,
    )


def test_context_is_contiguous_newest_tail() -> None:
    builder = ContextBuilder(max_messages=20, max_l1_tokens=6)
    selected = builder.select_l1([msg(1, "user", 3), msg(2, "assistant", 3), msg(3, "user", 3)])
    assert [item.sequence_no for item in selected] == [2, 3]


def test_context_can_include_four_carryover_without_counting_them_as_real() -> None:
    builder = ContextBuilder(max_messages=2, max_l1_tokens=100)
    messages = [
        msg(1, "user", 1, context_only=True),
        msg(2, "assistant", 1, context_only=True),
        msg(3, "user", 1),
        msg(4, "assistant", 1),
    ]
    assert [item.sequence_no for item in builder.select_l1(messages)] == [1, 2, 3, 4]


def test_local_chunking_never_splits_a_dialogue_round(tmp_path: Path) -> None:
    store = SQLiteSessionStore(tmp_path / "db.sqlite3")
    session = store.create_session("u", "a")
    messages = []
    for index, role in enumerate(["user", "assistant", "user", "assistant", "user", "assistant"]):
        messages.append(
            store.append_message(session.session_id, role=role, content=f"m{index}", token_count=1000)
        )
    long_term = FakeLongTermMemory()
    consolidator = SessionConsolidator(
        store=store,
        long_term=long_term,
        summary_provider=FakeTextProvider("local", ["摘要"]),
        memory_provider_name="local",
        memory_token_counter=FixedTokenCounter(),
        config=ConsolidatorConfig(local_chunk_tokens=15),
    )
    from bumi_mem.worker import MemoryWorker
    receipt = consolidator.submit(session.session_id, reason="manual_new")
    worker = MemoryWorker(consolidator)
    worker.run_once()
    worker.stop()
    assert store.get_job(receipt.job_id).status == "succeeded"
    assert len(long_term.add_calls) == 3
    for call in long_term.add_calls:
        roles = [item["role"] for item in call["messages"]]
        assert roles == ["user", "assistant"]


def test_memory_prompt_has_required_typed_prefixes() -> None:
    for prefix in [
        "[用户事实]",
        "[用户偏好]",
        "[用户边界]",
        "[用户计划]",
        "[重要事件]",
        "[机器人承诺]",
        "[共同事件]",
    ]:
        assert prefix in MEMORY_EXTRACTION_INSTRUCTIONS
    assert "Do not store questions" in MEMORY_EXTRACTION_INSTRUCTIONS
    assert "Do not store ordinary assistant replies" in MEMORY_EXTRACTION_INSTRUCTIONS


def test_one_open_session_per_user_agent_pair(tmp_path: Path) -> None:
    store = SQLiteSessionStore(tmp_path / "db.sqlite3")
    first = store.get_or_create_session("u", "a")
    second = store.get_or_create_session("u", "a")
    assert first.session_id == second.session_id


def test_opening_store_does_not_relabel_running_job(tmp_path: Path) -> None:
    from test_bumi_memory import make_engine, chat_turn, submit
    engine, store, _, _, _ = make_engine(tmp_path)
    chat_turn(engine)
    receipt = submit(engine)
    job = store.claim_job("live-worker", lease_seconds=30)
    restarted = SQLiteSessionStore(store.path)
    assert restarted.get_job(job.job_id).status == "running"
    assert restarted.get_session(receipt.new_session_id).status == "active"

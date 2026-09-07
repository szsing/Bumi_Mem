"""Interactive CLI for the Bumi_Mem MVP."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any
from urllib.request import urlopen

from bumi_mem.consolidator import ConsolidatorConfig, SessionConsolidator
from bumi_mem.engine import BumiMemory, BumiMemoryConfig
from bumi_mem.long_term import Mem0AdapterConfig, Mem0LongTermMemory
from bumi_mem.memory_access import SerialMemoryAccess
from bumi_mem.paths import DATA_ROOT, MODEL_ROOT, PROJECT_ROOT, configure_runtime
from bumi_mem.providers import OpenAICompatibleProvider, ProviderConfig
from bumi_mem.schemas import ConsolidationSubmission, ProviderConfigurationError
from bumi_mem.session_store import SQLiteSessionStore
from bumi_mem.token_counter import ApproxTokenCounter, HuggingFaceTokenCounter, TokenCounter


def _normalize_base_url(value: str) -> str:
    value = value.rstrip("/")
    return value if value.endswith("/v1") else value + "/v1"


def _provider_config(
    provider: str,
    *,
    purpose: str,
    model_override: str | None,
    base_url_override: str | None,
    max_tokens: int,
) -> ProviderConfig:
    prefix = f"BUMI_{purpose.upper()}_"
    if provider == "local":
        model = model_override or os.getenv(prefix + "MODEL") or "qwen2.5-7b-instruct"
        base_url = base_url_override or os.getenv(prefix + "BASE_URL") or os.getenv(
            "LOCAL_LLM_BASE_URL", "http://127.0.0.1:8000/v1"
        )
        return ProviderConfig(
            provider="local",
            model=model,
            base_url=_normalize_base_url(base_url),
            api_key="local",
            max_tokens=max_tokens,
            temperature=0.2 if purpose == "chat" else 0.0,
        )

    if provider != "deepseek":
        raise ProviderConfigurationError(f"unsupported provider: {provider}")
    model = (
        model_override
        or os.getenv(prefix + "MODEL")
        or os.getenv("DEEPSEEK_MODEL")
        or "deepseek-chat"
    )
    base_url = (
        base_url_override
        or os.getenv(prefix + "BASE_URL")
        or os.getenv("OPENAI_BASE_URL")
        or os.getenv("MEMORY_API_BASE_URL")
        or os.getenv("DEEPSEEK_BASE_URL")
        or "https://api.deepseek.com/v1"
    )
    api_key = (
        os.getenv(prefix + "API_KEY")
        or os.getenv("OPENAI_API_KEY")
        or os.getenv("DEEPSEEK_API_KEY")
        or ""
    )
    return ProviderConfig(
        provider="deepseek",
        model=model,
        base_url=_normalize_base_url(base_url),
        api_key=api_key,
        max_tokens=max_tokens,
        temperature=0.2 if purpose == "chat" else 0.0,
    )


def _check_local_provider(config: ProviderConfig) -> None:
    health = config.base_url.removesuffix("/v1") + "/health"
    try:
        with urlopen(health, timeout=3) as response:  # noqa: S310 - local URL only
            if response.status >= 400:
                raise RuntimeError(f"HTTP {response.status}")
    except Exception as exc:
        raise ProviderConfigurationError(
            f"本地模型未就绪：{health}。请先运行 scripts/start_vllm_qwen.sh"
        ) from exc


def _token_counter(provider: str) -> TokenCounter:
    if provider == "local":
        return HuggingFaceTokenCounter(MODEL_ROOT / "Qwen2.5-7B-Instruct")
    return ApproxTokenCounter()


def build_engine(args: argparse.Namespace) -> BumiMemory:
    configure_runtime()
    chat_config = _provider_config(
        args.chat_provider,
        purpose="chat",
        model_override=args.chat_model,
        base_url_override=args.chat_base_url,
        max_tokens=args.max_answer_tokens,
    )
    memory_config = _provider_config(
        args.memory_provider,
        purpose="memory",
        model_override=args.memory_model,
        base_url_override=args.memory_base_url,
        max_tokens=args.summary_max_tokens,
    )
    if args.chat_provider == "local":
        _check_local_provider(chat_config)
    if args.memory_provider == "local":
        _check_local_provider(memory_config)

    chat_provider = OpenAICompatibleProvider(chat_config)
    summary_provider = OpenAICompatibleProvider(memory_config)
    chat_counter = _token_counter(args.chat_provider)
    memory_counter = _token_counter(args.memory_provider)

    db_path = Path(args.db_path).expanduser().resolve() if args.db_path else DATA_ROOT / "bumi_memory.sqlite3"
    store = SQLiteSessionStore(db_path)
    adapter_config = Mem0AdapterConfig(
        data_dir=DATA_ROOT / "bumi_mem_l3",
        embed_model=MODEL_ROOT / "Qwen3-Embedding-0.6B",
        embed_device=args.embed_device,
        collection_name=args.collection_name,
        memory_max_tokens=(args.memory_max_tokens if args.memory_max_tokens is not None else
                           (512 if args.memory_provider == "local" else int(os.getenv("MEM0_MEMORY_MAX_TOKENS", "4096")))),
    )
    # Construct the native SDK on its owning thread, not on the CLI thread.
    l3 = SerialMemoryAccess(lambda: Mem0LongTermMemory(memory_config, adapter_config),
                            read_timeout=args.memory_read_timeout)
    consolidator = SessionConsolidator(
        store=store,
        long_term=l3,
        summary_provider=summary_provider,
        memory_provider_name=args.memory_provider,
        memory_token_counter=memory_counter,
        config=ConsolidatorConfig(
            carryover_messages=args.carryover_messages,
            local_chunk_tokens=args.local_chunk_tokens,
            summary_max_tokens=args.summary_max_tokens,
            max_attempts=args.memory_max_attempts,
            retry_base_seconds=args.memory_retry_base_seconds,
        ),
    )
    return BumiMemory(
        store=store,
        long_term=l3,
        chat_provider=chat_provider,
        consolidator=consolidator,
        chat_token_counter=chat_counter,
        config=BumiMemoryConfig(
            max_messages=args.max_messages,
            max_l1_tokens=args.max_l1_tokens,
            top_k=args.top_k,
            search_threshold=args.search_threshold,
            max_answer_tokens=args.max_answer_tokens,
            memory_read_timeout=args.memory_read_timeout,
        ),
        auto_start=False,
    )


def _memory_text(item: dict[str, Any]) -> str:
    return str(item.get("memory", item.get("text", item.get("data", ""))))


def _print_memories(rows: list[dict[str, Any]]) -> None:
    if not rows:
        print("  （没有结果）")
        return
    for index, row in enumerate(rows, 1):
        score = row.get("score")
        suffix = f" score={score:.4f}" if isinstance(score, (int, float)) else ""
        print(f"  {index}. [{row.get('id', '-')}] {_memory_text(row)}{suffix}")


def _print_submission(receipt: ConsolidationSubmission, *, debug: bool) -> None:
    if debug:
        print(f"[后台整理] job={receipt.job_id} status={receipt.status} new_session={receipt.new_session_id}")


def _print_jobs(engine: BumiMemory, args: argparse.Namespace) -> None:
    jobs = engine.memory_jobs(user_id=args.user_id, agent_id=args.agent_id)
    if not jobs:
        print("（没有后台任务）")
    for job in jobs:
        print(f"job={job.job_id} kind={job.kind} status={job.status} attempts={job.attempts}/{job.max_attempts}")
        if job.last_error:
            print(f"  error: {job.last_error}")


def _help() -> None:
    print("""命令：
  /help                  显示帮助
  /memories              查看当前 user/agent 的 L3
  /search <query>        手动检索 L3
  /session               查看当前 L1 和后台任务
  /jobs                  查看后台任务状态和错误
  /retry <job-id>        重试失败任务（仅限当前 user/agent）
  /retry <job-id> --allow-replay
                         核查 L3 后，显式允许重放结果不确定的分块（再次确认）
  /recall on|off         控制回答前的 L3 检索
  /debug on|off          显示或隐藏内部调试信息
  /new                   保存整理任务并立即开始新会话
  /discard               丢弃当前 L1，不影响之前已经提交的任务
  /forget-all            屏蔽当前 user/agent 的 L3，取消旧写入并后台清理
  /quit                  退出；save 只保证记录/任务落盘，不保证 L3 已整理完
""")


def run_interactive(engine: BumiMemory, args: argparse.Namespace) -> None:
    try:
        engine.start()
        _run_interactive(engine, args)
    finally:
        engine.close()


def _run_interactive(engine: BumiMemory, args: argparse.Namespace) -> None:
    debug = bool(args.debug)
    recall_enabled = True
    session = engine.get_or_create_session(user_id=args.user_id, agent_id=args.agent_id)
    print("\nBumi_Mem L1/L2/L3 交互测试（持久化后台整理）")
    print(f"user-id: {args.user_id}  agent-id: {args.agent_id}")
    print(f"session-id: {session.session_id}")
    print(f"Chat Provider: {args.chat_provider}  Memory Provider: {args.memory_provider}")
    print("输入 /help 查看命令；/jobs 查看后台整理状态。\n")
    while True:
        try:
            text = input("你> ").strip()
            if not text:
                continue
            if text == "/help":
                _help()
                continue
            if text == "/memories":
                if engine.store.reads_blocked(args.user_id, args.agent_id):
                    print("L3 正在等待删除确认，暂不展示；请用 /jobs 查看。")
                else:
                    _print_memories(engine.get_memories(user_id=args.user_id, agent_id=args.agent_id))
                continue
            if text.startswith("/search "):
                _print_memories(engine.search_long_term(user_id=args.user_id, agent_id=args.agent_id,
                                                        query=text.removeprefix("/search ").strip()))
                continue
            if text in ("/session", "/jobs"):
                if text == "/session":
                    stats = engine.session_stats(user_id=args.user_id, agent_id=args.agent_id)
                    print(f"session={stats.session_id} status={stats.status} "
                          f"messages={stats.effective_messages}/{args.max_messages} "
                          f"tokens={stats.effective_tokens}/{args.max_l1_tokens} carryover={stats.context_only_messages}")
                _print_jobs(engine, args)
                continue
            if text.startswith("/retry "):
                parts = text.split()
                if len(parts) not in (2, 3) or (len(parts) == 3 and parts[2] != "--allow-replay"):
                    print("用法：/retry <job-id> [--allow-replay]")
                    continue
                allow_replay = len(parts) == 3
                if allow_replay and input("重放可能产生重复记忆。核查 L3 后输入 REPLAY 确认：").strip() != "REPLAY":
                    print("已取消。")
                    continue
                job = engine.retry_memory_job(parts[1], user_id=args.user_id, agent_id=args.agent_id,
                                              allow_replay=allow_replay)
                print(f"任务已重新排队：{job.job_id}")
                continue
            if text.startswith("/recall ") or text.startswith("/debug "):
                command, value = text.split(maxsplit=1)
                value = value.strip().lower()
                if value not in ("on", "off"):
                    print(f"用法：{command} on 或 off")
                elif command == "/recall":
                    recall_enabled = value == "on"
                    print(f"L3 检索已{'开启' if recall_enabled else '关闭'}。")
                else:
                    debug = value == "on"
                    print(f"调试显示已{'开启' if debug else '关闭'}。")
                continue
            if text == "/new":
                receipt = engine.end_session(user_id=args.user_id, agent_id=args.agent_id)
                _print_submission(receipt, debug=debug)
                print("已开始新会话；上一段已保存，后台整理不影响继续聊天。")
                continue
            if text == "/discard":
                if input("输入 DISCARD 确认放弃当前 L1（不撤销之前的后台任务）：").strip() == "DISCARD":
                    engine.discard_session(user_id=args.user_id, agent_id=args.agent_id)
                    print("当前会话已丢弃，已开始新会话。")
                else:
                    print("已取消。")
                continue
            if text == "/forget-all":
                confirm = input(f"输入完整 user-id {args.user_id!r} 确认清理当前 user/agent 的全部 L3：").strip()
                if confirm == args.user_id:
                    job = engine.forget_all(user_id=args.user_id, agent_id=args.agent_id)
                    print(f"L3 已屏蔽，旧写入已取消，删除任务已保存：{job.job_id}。用 /jobs 确认物理清理结果。")
                else:
                    print("已取消。")
                continue
            if text == "/quit":
                stats = engine.session_stats(user_id=args.user_id, agent_id=args.agent_id)
                if stats.effective_messages:
                    action = input("输入 save / discard / cancel（save 保存任务，下次启动可继续整理）：").strip().lower()
                    if action == "save":
                        # Do not let the worker start the new quit-save job as we exit.
                        engine.worker.stop(timeout=0.0)
                        receipt = engine.end_session(user_id=args.user_id, agent_id=args.agent_id, reason="quit_save")
                        _print_submission(receipt, debug=debug)
                    elif action == "discard":
                        engine.discard_session(user_id=args.user_id, agent_id=args.agent_id)
                    else:
                        print("已取消退出。")
                        continue
                print("已退出；未完成整理任务已落盘，下次启动恢复。正在写入时中断的任务可能需要核查后手动重试。")
                return
            if text.startswith("/"):
                print("未知命令；输入 /help 查看帮助。")
                continue
            result = engine.chat(user_id=args.user_id, agent_id=args.agent_id, text=text,
                                 recall_enabled=recall_enabled, on_answer=lambda answer: print(f"Bumi> {answer}"))
            if debug:
                print(f"[L3 检索] status={result.recall_status}")
                _print_memories(result.recalled)
                if result.auto_consolidation is not None:
                    _print_submission(result.auto_consolidation, debug=True)
        except (EOFError, KeyboardInterrupt):
            print("\n已退出；L1 和已提交任务均已保存。当前未提交的 L1 将在下次启动继续，不自动打包。")
            return
        except Exception as exc:
            print(f"[错误] {exc}；记录未自动丢弃，可用 /session 或 /jobs 检查。", file=sys.stderr)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bumi_Mem layered companion-memory CLI")
    parser.add_argument("--user-id", default="demo_user")
    parser.add_argument("--agent-id", default="bumi_companion")
    parser.add_argument("--chat-provider", choices=["deepseek", "local"], default=os.getenv("CHAT_PROVIDER", "deepseek"))
    parser.add_argument("--memory-provider", choices=["deepseek", "local"], default=os.getenv("MEMORY_PROVIDER", "deepseek"))
    parser.add_argument("--chat-model", default=None)
    parser.add_argument("--memory-model", default=None)
    parser.add_argument("--chat-base-url", default=None)
    parser.add_argument("--memory-base-url", default=None)
    parser.add_argument("--db-path", default=None)
    parser.add_argument("--collection-name", default=os.getenv("BUMI_COLLECTION_NAME", "bumi_mem_l3"))
    parser.add_argument("--embed-device", choices=["cpu", "cuda"], default=os.getenv("MEM0_EMBED_DEVICE", "cpu"))
    parser.add_argument("--max-messages", type=int, default=20)
    parser.add_argument("--max-l1-tokens", type=int, default=8000)
    parser.add_argument("--carryover-messages", type=int, default=4)
    parser.add_argument("--local-chunk-tokens", type=int, default=3500)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--search-threshold", type=float, default=0.1)
    parser.add_argument("--max-answer-tokens", type=int, default=512)
    parser.add_argument("--summary-max-tokens", type=int, default=512)
    parser.add_argument("--memory-max-tokens", type=int, default=None)
    parser.add_argument("--memory-read-timeout", type=float, default=0.2)
    parser.add_argument("--memory-max-attempts", type=int, default=3)
    parser.add_argument("--memory-retry-base-seconds", type=float, default=5.0)
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        engine = build_engine(args)
    except Exception as exc:
        print(f"初始化失败：{exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    run_interactive(engine, args)


if __name__ == "__main__":
    main()

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
from bumi_mem.paths import DATA_ROOT, MODEL_ROOT, PROJECT_ROOT, configure_runtime
from bumi_mem.providers import OpenAICompatibleProvider, ProviderConfig
from bumi_mem.schemas import ConsolidationResult, ProviderConfigurationError, SessionBlockedError
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
    l3 = Mem0LongTermMemory(
        memory_config,
        Mem0AdapterConfig(
            data_dir=DATA_ROOT / "bumi_mem_l3",
            embed_model=MODEL_ROOT / "Qwen3-Embedding-0.6B",
            embed_device=args.embed_device,
            collection_name=args.collection_name,
            memory_max_tokens=(
                args.memory_max_tokens
                if args.memory_max_tokens is not None
                else (512 if args.memory_provider == "local" else int(os.getenv("MEM0_MEMORY_MAX_TOKENS", "4096")))
            ),
        ),
    )
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
        ),
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


def _print_consolidation(result: ConsolidationResult, *, debug: bool) -> None:
    if not result.success:
        print(f"[会话整合失败] 消息已安全保存在 L1：{result.error}", file=sys.stderr)
        print("请使用 /new 重试，或使用 /discard 放弃当前会话。", file=sys.stderr)
        return
    if not debug:
        return
    counts = result.event_counts
    print(f"[会话边界] reason={result.reason} attempt={result.attempt}")
    print(f"[L2] status={result.summary_status} summary={result.summary}")
    print(
        "[L3] "
        + " ".join(f"{name}={count}" for name, count in sorted(counts.items()))
        if counts
        else "[L3] 没有新增或变更"
    )
    print(f"[新会话] {result.new_session_id}")


def _help() -> None:
    print(
        """命令：
  /help                  显示帮助
  /memories              查看当前用户的全部 L3 记忆
  /search <内容>         手动检索 L3
  /session               查看当前 L1 状态
  /recall on|off         控制回答前是否检索 L3
  /debug on|off          显示或隐藏内部调试信息
  /new                   整合当前会话并开始新会话
  /discard               不写 L3，丢弃当前会话并开始新会话
  /forget-all            删除当前用户全部 L3，需要再次输入 user-id
  /quit                  退出；有未整合消息时选择 save/discard/cancel
"""
    )


def run_interactive(engine: BumiMemory, args: argparse.Namespace) -> None:
    debug = bool(args.debug)
    recall_enabled = True
    session = engine.get_or_create_session(user_id=args.user_id, agent_id=args.agent_id)
    print("\nBumi_Mem L1/L2/L3 交互测试")
    print(f"user-id: {args.user_id}")
    print(f"agent-id: {args.agent_id}")
    print(f"session-id: {session.session_id}")
    print(f"Chat Provider: {args.chat_provider}")
    print(f"Memory Provider: {args.memory_provider}")
    print("输入 /help 查看命令。\n")

    while True:
        try:
            text = input("你> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n已退出；当前 L1 已保存在 SQLite，尚未自动整合。")
            return
        if not text:
            continue
        if text == "/help":
            _help()
            continue
        if text == "/memories":
            print("[L3 长期记忆]")
            _print_memories(engine.get_memories(user_id=args.user_id, agent_id=args.agent_id))
            continue
        if text.startswith("/search "):
            query = text.removeprefix("/search ").strip()
            print("[L3 手动检索]")
            _print_memories(engine.search_long_term(user_id=args.user_id, agent_id=args.agent_id, query=query))
            continue
        if text == "/session":
            stats = engine.session_stats(user_id=args.user_id, agent_id=args.agent_id)
            print(
                f"session={stats.session_id} status={stats.status} "
                f"messages={stats.effective_messages}/{args.max_messages} "
                f"tokens={stats.effective_tokens}/{args.max_l1_tokens} "
                f"carryover={stats.context_only_messages} "
                f"chat_provider={args.chat_provider} memory_provider={args.memory_provider}"
            )
            continue
        if text.startswith("/recall "):
            value = text.removeprefix("/recall ").strip().lower()
            if value not in {"on", "off"}:
                print("用法：/recall on 或 /recall off")
            else:
                recall_enabled = value == "on"
                print(f"L3 检索已{'开启' if recall_enabled else '关闭'}。")
            continue
        if text.startswith("/debug "):
            value = text.removeprefix("/debug ").strip().lower()
            if value not in {"on", "off"}:
                print("用法：/debug on 或 /debug off")
            else:
                debug = value == "on"
                print(f"Debug 已{'开启' if debug else '关闭'}。")
            continue
        if text == "/new":
            result = engine.end_session(user_id=args.user_id, agent_id=args.agent_id)
            _print_consolidation(result, debug=debug)
            if result.success and not debug:
                print("新会话已开始。")
            continue
        if text == "/discard":
            confirm = input("输入 DISCARD 确认放弃当前 L1：").strip()
            if confirm == "DISCARD":
                session = engine.discard_session(user_id=args.user_id, agent_id=args.agent_id)
                print(f"当前会话已丢弃；新 session：{session.session_id}")
            else:
                print("已取消。")
            continue
        if text == "/forget-all":
            confirm = input(f"输入完整 user-id {args.user_id!r} 确认删除全部 L3：").strip()
            if confirm == args.user_id:
                engine.forget_all(user_id=args.user_id, agent_id=args.agent_id)
                print("当前用户的全部 L3 已删除。")
            else:
                print("已取消。")
            continue
        if text == "/quit":
            stats = engine.session_stats(user_id=args.user_id, agent_id=args.agent_id)
            if stats.effective_messages == 0:
                print("已退出。")
                return
            action = input("当前 L1 尚未整合。输入 save / discard / cancel：").strip().lower()
            if action == "save":
                result = engine.end_session(user_id=args.user_id, agent_id=args.agent_id, reason="quit_save")
                _print_consolidation(result, debug=debug)
                if result.success:
                    print("已保存并退出。")
                    return
            elif action == "discard":
                engine.discard_session(user_id=args.user_id, agent_id=args.agent_id)
                print("已丢弃当前会话并退出。")
                return
            else:
                print("已取消退出。")
            continue
        if text.startswith("/"):
            print("未知命令；输入 /help 查看帮助。")
            continue

        try:
            result = engine.chat(
                user_id=args.user_id,
                agent_id=args.agent_id,
                text=text,
                recall_enabled=recall_enabled,
                on_answer=lambda answer: print(f"Bumi> {answer}"),
            )
        except SessionBlockedError as exc:
            print(f"[会话被阻止] {exc}", file=sys.stderr)
            continue
        except Exception as exc:
            print(f"[错误] 本轮失败：{exc}", file=sys.stderr)
            continue
        if debug:
            print("[L3 检索]")
            _print_memories(result.recalled)
        if result.auto_consolidation is not None:
            _print_consolidation(result.auto_consolidation, debug=debug)


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

#!/usr/bin/env python3
"""生成记忆服务的 service.env（仓库之外、权限 600）。

把 DeepSeek Key 留在已有的 secrets 文件里，只从那里读；本脚本只输出生成的文件路径与
脱敏摘要，不打印任何密钥。已存在时拒绝覆盖，除非显式 --force。

用法：
    MEM0_PYTHON=$HOME/.venvs/bumi-cuda/bin/python \
    scripts/make_service_env.py --qdrant-url http://127.0.0.1:6333 --entity-boost off
"""

from __future__ import annotations

import argparse
import os
import secrets
import socket
import stat
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def port_in_use(port: int, host: str = "127.0.0.1") -> bool:
    """端口是主机级资源：同机多实例必须换端口，否则 bind 失败或问到别人的服务。"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.5)
        return sock.connect_ex((host, port)) == 0


def read_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.is_file():
        return values
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def main() -> int:
    workspace = Path(os.environ.get("MEM0_WORKSPACE_ROOT", REPO.parent.parent)).expanduser()
    runtime = Path(os.environ.get("MEM0_RUNTIME_ROOT", workspace / "mem0-runtime")).expanduser()

    parser = argparse.ArgumentParser(description="生成记忆服务 service.env")
    parser.add_argument("--out", default=str(runtime / "memory-service" / "service.env"))
    parser.add_argument("--deepseek-secrets",
                        default=str(runtime / "secrets" / "deepseek.env"))
    parser.add_argument("--data-dir", default=str(runtime / "memory-service" / "data-test"))
    parser.add_argument("--embed-model", default=str(workspace / "models" / "Qwen3-Embedding-0.6B"))
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--agent-id", default="bumi_robot_001")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default="8088")
    parser.add_argument("--embed-device", default="cuda", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--qdrant-url", default=None, help="设置后走 Qdrant Server")
    parser.add_argument("--entity-boost", default="on", choices=["on", "off"],
                        help="默认 on（保守，等于代码默认）；off 能把检索 P90 压进 60ms，"
                             "但要先过真实语料召回回归（见 docs/latency/REPORT.md §8.2/8.6）")
    parser.add_argument("--memory-model", default="deepseek-chat")
    parser.add_argument("--memory-base-url", default="https://api.deepseek.com/v1")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    out = Path(args.out).expanduser()
    if out.exists() and not args.force:
        print(f"{out} 已存在；如需重建请加 --force（会轮换服务密钥）", file=sys.stderr)
        return 2

    if port_in_use(int(args.port)):
        print(f"警告：{args.host}:{args.port} 目前有服务在监听——可能是同机其他用户或其他实例"
              f"（包括你自己的另一个实例）。"
              f"同机多实例必须用不同 BUMI_PORT；本机 8088 已归 sunzhixing 的实例，"
              f"我们自己的验收实例建议用 8099。", file=sys.stderr)

    secrets_map = read_env_file(Path(args.deepseek_secrets).expanduser())
    memory_key = secrets_map.get("DEEPSEEK_API_KEY", "").strip()
    if not memory_key:
        print(f"在 {args.deepseek_secrets} 里找不到 DEEPSEEK_API_KEY", file=sys.stderr)
        return 2

    service_key = secrets.token_urlsafe(32)
    lines = [
        "# 由 scripts/make_service_env.py 生成；不要提交到 Git。",
        f'BUMI_HOST="{args.host}"',
        f'BUMI_PORT="{args.port}"',
        f'BUMI_AGENT_ID="{args.agent_id}"',
        f'BUMI_DATA_DIR="{args.data_dir}"',
        f'BUMI_EMBED_MODEL_PATH="{args.embed_model}"',
        f'BUMI_PYTHON="{args.python}"',
        f'MEM0_EMBED_DEVICE="{args.embed_device}"',
        f'BUMI_SERVICE_API_KEY="{service_key}"',
        'BUMI_MEMORY_PROVIDER="deepseek"',
        f'BUMI_MEMORY_MODEL="{args.memory_model}"',
        f'BUMI_MEMORY_BASE_URL="{args.memory_base_url}"',
        f'BUMI_MEMORY_API_KEY="{memory_key}"',
        'BUMI_MEMORY_READ_TIMEOUT="0.2"',
        'BUMI_EXPLICIT_RECALL_TIMEOUT="1.0"',
        'BUMI_MEMORY_WARMUP_TIMEOUT="120"',
        'BUMI_MAX_MESSAGES="20"',
        'BUMI_MAX_L1_TOKENS="8000"',
        'BUMI_CARRYOVER_MESSAGES="4"',
        'BUMI_TOP_K="5"',
        'BUMI_SEARCH_THRESHOLD="0.1"',
        '# 后台整理在有前台活动时让路（实测消除约 5% 的检索被阻塞；0 = 旧行为）',
        'BUMI_CONSOLIDATION_IDLE_SECONDS="2"',
        'BUMI_CONSOLIDATION_MAX_DEFER_SECONDS="60"',
    ]
    if args.qdrant_url:
        lines.append(f'BUMI_QDRANT_URL="{args.qdrant_url}"')
    lines.append(f'BUMI_ENTITY_BOOST="{args.entity_boost}"')

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    out.chmod(stat.S_IRUSR | stat.S_IWUSR)

    print(f"已生成：{out}（权限 600）")
    print(f"  BUMI_DATA_DIR={args.data_dir}")
    print(f"  BUMI_EMBED_MODEL_PATH={args.embed_model}")
    print(f"  BUMI_PYTHON={args.python}")
    print(f"  MEM0_EMBED_DEVICE={args.embed_device}  BUMI_QDRANT_URL={args.qdrant_url or '(本地 path client)'}"
          f"  BUMI_ENTITY_BOOST={args.entity_boost}")
    print(f"  BUMI_SERVICE_API_KEY=<已生成 {len(service_key)} 字符>  BUMI_MEMORY_API_KEY=<来自 secrets，未打印>")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

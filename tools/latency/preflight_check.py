#!/usr/bin/env python3
"""验收前环境预检：GPU / 嵌入设备 / Qdrant 后端 / 实体加成策略 / 模型路径。

把“这次到底跑在什么环境上”变成可执行的检查，避免验收时才发现其实是 CPU 或本地
path client 在跑。硬性检查失败会返回非零退出码，可直接接进部署脚本或 CI。

用法：
    BUMI_QDRANT_URL=http://127.0.0.1:6333 BUMI_ENTITY_BOOST=off \
    tools/latency/preflight_check.py --require-gpu --require-server \
        --collection bumi_latency_bench
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))


def reexec_with_mem0_python() -> None:
    """和压测脚本一致：MEM0_PYTHON 指向项目 venv 时先切过去，避免系统 python 缺依赖。"""
    python = os.environ.get("MEM0_PYTHON")
    marker = os.environ.get("BUMI_PREFLIGHT_PYTHON")
    if python and os.access(python, os.X_OK) and marker != python:
        env = dict(os.environ)
        env["BUMI_PREFLIGHT_PYTHON"] = python
        os.execve(python, [python, str(Path(__file__).resolve()), *sys.argv[1:]], env)


def check(name: str, ok: bool, detail: str, *, hard: bool = True) -> dict:
    return {"name": name, "ok": bool(ok), "hard": hard, "detail": detail}


def http_json(url: str, timeout: float = 5.0):
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def collect(args) -> list[dict]:
    results: list[dict] = []

    # 1. Python / torch / CUDA
    results.append(check("python", True, f"{sys.executable} ({sys.version.split()[0]})"))
    try:
        import torch

        cuda = bool(torch.cuda.is_available())
        devices = ([torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
                   if cuda else [])
        results.append(check("torch", True, f"{torch.__version__} cuda_build={torch.version.cuda}"))
        results.append(check("torch_cuda_available", cuda,
                             f"devices={devices}", hard=args.require_gpu))
    except Exception as exc:  # noqa: BLE001
        results.append(check("torch", False, f"{type(exc).__name__}: {exc}", hard=args.require_gpu))
        cuda = False

    # 2. 请求/实际生效的编码设备
    requested = args.embed_device or os.environ.get("MEM0_EMBED_DEVICE", "auto")
    try:
        from bumi_mem.long_term import resolve_embed_device

        resolved = resolve_embed_device(requested)
        results.append(check("embed_device", resolved == requested or requested == "auto",
                             f"requested={requested} resolved={resolved}",
                             hard=args.require_gpu))
        if args.require_gpu:
            results.append(check("embed_device_is_cuda", resolved == "cuda",
                                 f"resolved={resolved}", hard=True))
    except Exception as exc:  # noqa: BLE001
        results.append(check("embed_device", False, f"{type(exc).__name__}: {exc}",
                             hard=args.require_gpu))

    # 3. 嵌入模型路径
    model_root = Path(os.environ.get("MEM0_MODEL_ROOT", REPO.parent.parent / "models"))
    model_dir = model_root / "Qwen3-Embedding-0.6B"
    results.append(check("embed_model", model_dir.is_dir(), str(model_dir)))

    # 4. Qdrant 后端
    url = os.environ.get("BUMI_QDRANT_URL", "").strip()
    if url:
        results.append(check("vector_backend", True, f"qdrant-server({url})"))
        try:
            payload = http_json(f"{url.rstrip('/')}/collections")
            names = [item["name"] for item in payload["result"]["collections"]]
            results.append(check("qdrant_reachable", True, f"collections={len(names)}"))
            if args.collection:
                present = args.collection in names
                if present:
                    info = http_json(f"{url.rstrip('/')}/collections/{args.collection}")
                    detail = f"{args.collection}: points={info['result']['points_count']}"
                else:
                    detail = f"{args.collection}: 不存在"
                results.append(check("qdrant_collection", present, detail,
                                     hard=args.require_collection))
        except (urllib.error.URLError, OSError, ValueError, KeyError) as exc:
            results.append(check("qdrant_reachable", False, f"{type(exc).__name__}: {exc}",
                                 hard=args.require_server))
    else:
        results.append(check("vector_backend", not args.require_server,
                             "qdrant-local-path（本地精确扫描，仅开发/小数据）",
                             hard=args.require_server))

    # 5. 实体加成策略
    boost = os.environ.get("BUMI_ENTITY_BOOST", "on").strip().lower()
    results.append(check("entity_boost", boost in {"on", "off"},
                         boost, hard=True))
    if args.require_entity_boost_off:
        results.append(check("entity_boost_off", boost == "off",
                             f"BUMI_ENTITY_BOOST={boost}", hard=True))
    return results


def main() -> int:
    reexec_with_mem0_python()
    parser = argparse.ArgumentParser(description="Bumi 记忆检索验收前环境预检")
    parser.add_argument("--require-gpu", action="store_true", help="要求 CUDA 可用且实际用 GPU")
    parser.add_argument("--require-server", action="store_true", help="要求走 Qdrant Server")
    parser.add_argument("--require-collection", action="store_true", help="要求目标集合已存在")
    parser.add_argument("--require-entity-boost-off", action="store_true",
                        help="要求 BUMI_ENTITY_BOOST=off（达标配置）")
    parser.add_argument("--collection", default=None, help="要检查的 Qdrant 集合名")
    parser.add_argument("--embed-device", default=None, choices=["auto", "cpu", "cuda"])
    parser.add_argument("--json", dest="json_out", default=None)
    args = parser.parse_args()

    results = collect(args)
    print("=" * 78)
    print("验收前环境预检")
    print("=" * 78)
    failed = 0
    for item in results:
        mark = "PASS" if item["ok"] else ("FAIL" if item["hard"] else "WARN")
        if not item["ok"] and item["hard"]:
            failed += 1
        print(f"{mark:<5} {item['name']:<26} {item['detail']}")
    print("-" * 78)
    print(f"硬性检查：{'全部通过' if failed == 0 else f'{failed} 项失败'}")
    if args.json_out:
        Path(args.json_out).expanduser().write_text(
            json.dumps({"results": results, "failed": failed}, ensure_ascii=False, indent=2),
            encoding="utf-8")
        print(f"原始数据：{args.json_out}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

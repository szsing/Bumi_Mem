"""运行时路径解析：基于 Git 仓库结构，通过环境变量可覆盖默认路径。

所有路径常量在模块加载时一次性解析（expanduser + resolve），
确保后续使用的是绝对路径。环境变量优先级高于默认值。
"""

from __future__ import annotations

import os
from pathlib import Path

# 项目根目录：src/bumi_mem/ 的上两级（即 Git 仓库根目录）
PROJECT_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = Path(os.getenv("BUMI_WORKSPACE_ROOT", str(PROJECT_ROOT.parent))).expanduser().resolve()
RUNTIME_ROOT = Path(os.getenv("BUMI_RUNTIME_ROOT", str(WORKSPACE_ROOT / "graphiti-runtime"))).expanduser().resolve()
MODEL_ROOT = Path(os.getenv("BUMI_MODEL_ROOT", str(WORKSPACE_ROOT / "models"))).expanduser().resolve()
DATA_ROOT = Path(os.getenv("BUMI_DATA_DIR", os.getenv("BUMI_DATA_ROOT", str(RUNTIME_ROOT / "data")))).expanduser().resolve()
RESULTS_ROOT = Path(os.getenv("BUMI_RESULTS_ROOT", str(RUNTIME_ROOT / "results"))).expanduser().resolve()
LOG_ROOT = Path(os.getenv("BUMI_LOG_ROOT", str(RUNTIME_ROOT / "logs"))).expanduser().resolve()
SECRETS_ROOT = Path(os.getenv("BUMI_SECRETS_ROOT", str(RUNTIME_ROOT / "secrets"))).expanduser().resolve()
CACHE_ROOT = RUNTIME_ROOT / "cache"
HOME_ROOT = RUNTIME_ROOT / "home"
TMP_ROOT = RUNTIME_ROOT / "tmp"
CONDA_ROOT = RUNTIME_ROOT / "conda"


def configure_runtime() -> None:
    """配置运行时环境变量并创建所需目录。

    将所有路径常量写入环境变量，确保下游库（HuggingFace、PyTorch、
    pip、Conda 等）的缓存和下载都落在受控目录下。同时禁用 Mem0 遥测
    和 tokenizers 并行化，避免多进程死锁。
    """
    # 各工具的缓存/数据目录映射：将环境变量指向受控路径
    paths = {
        "XDG_CACHE_HOME": CACHE_ROOT,
        "PIP_CACHE_DIR": CACHE_ROOT / "pip",
        "HF_HOME": CACHE_ROOT / "huggingface",
        "HUGGINGFACE_HUB_CACHE": CACHE_ROOT / "huggingface" / "hub",
        "TORCH_HOME": CACHE_ROOT / "torch",
        "VLLM_CACHE_ROOT": CACHE_ROOT / "vllm",
        "TRITON_CACHE_DIR": CACHE_ROOT / "triton",
        "UV_CACHE_DIR": CACHE_ROOT / "uv",
        "CONDA_PKGS_DIRS": CONDA_ROOT / "pkgs",
        "CONDA_ENVS_PATH": CONDA_ROOT / "envs",
        "TMPDIR": TMP_ROOT,
    }
    os.environ.setdefault("BUMI_RUNTIME_ROOT", str(RUNTIME_ROOT))
    os.environ.setdefault("BUMI_MODEL_ROOT", str(MODEL_ROOT))
    os.environ.setdefault("BUMI_DATA_ROOT", str(DATA_ROOT))
    os.environ.setdefault("BUMI_RESULTS_ROOT", str(RESULTS_ROOT))
    os.environ.setdefault("BUMI_LOG_ROOT", str(LOG_ROOT))
    os.environ.setdefault("BUMI_SECRETS_ROOT", str(SECRETS_ROOT))
    os.environ["GRAPHITI_TELEMETRY_ENABLED"] = "false"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    # 写入各工具的环境变量并自动创建目录
    for name, path in paths.items():
        os.environ[name] = str(path)
        path.mkdir(parents=True, exist_ok=True)
    # 确保核心数据目录存在
    for path in (DATA_ROOT, RESULTS_ROOT, LOG_ROOT, SECRETS_ROOT):
        path.mkdir(parents=True, exist_ok=True)

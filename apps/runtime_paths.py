"""共享路径配置：代码仓库和外部运行时状态的路径常量。

所有路径通过环境变量可覆盖，确保运行时产物不落入 Git 仓库内。
"""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = Path(
    os.environ.get("MEM0_WORKSPACE_ROOT", str(PROJECT_ROOT.parent))
).expanduser().resolve()
RUNTIME_ROOT = Path(
    os.environ.get("MEM0_RUNTIME_ROOT", str(WORKSPACE_ROOT / "mem0-runtime"))
).expanduser().resolve()
MODEL_ROOT = Path(
    os.environ.get("MEM0_MODEL_ROOT", str(WORKSPACE_ROOT / "models"))
).expanduser().resolve()
DATA_ROOT = Path(os.environ.get("MEM0_DATA_ROOT", str(RUNTIME_ROOT / "data"))).expanduser().resolve()
RESULTS_ROOT = Path(os.environ.get("MEM0_RESULTS_ROOT", str(RUNTIME_ROOT / "results"))).expanduser().resolve()
LOG_ROOT = Path(os.environ.get("MEM0_LOG_ROOT", str(RUNTIME_ROOT / "logs"))).expanduser().resolve()
SECRETS_ROOT = Path(os.environ.get("MEM0_SECRETS_ROOT", str(RUNTIME_ROOT / "secrets"))).expanduser().resolve()
CACHE_ROOT = RUNTIME_ROOT / "cache"
HOME_ROOT = RUNTIME_ROOT / "home"
TMP_ROOT = RUNTIME_ROOT / "tmp"
CONDA_ROOT = RUNTIME_ROOT / "conda"


def configure_environment(root: Path | None = None) -> None:
    """配置运行时环境变量并创建所需目录，确保生成状态不落入 Git 仓库。"""

    del root  # Kept for compatibility with existing callers.
    paths = {
        "HOME": HOME_ROOT,
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
    os.environ["MEM0_TELEMETRY"] = "False"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ.setdefault("MEM0_RUNTIME_ROOT", str(RUNTIME_ROOT))
    os.environ.setdefault("MEM0_MODEL_ROOT", str(MODEL_ROOT))
    os.environ.setdefault("MEM0_DATA_ROOT", str(DATA_ROOT))
    os.environ.setdefault("MEM0_RESULTS_ROOT", str(RESULTS_ROOT))
    os.environ.setdefault("MEM0_LOG_ROOT", str(LOG_ROOT))
    os.environ.setdefault("MEM0_SECRETS_ROOT", str(SECRETS_ROOT))
    for name, path in paths.items():
        os.environ[name] = str(path)
        path.mkdir(parents=True, exist_ok=True)
    for path in (DATA_ROOT, RESULTS_ROOT, LOG_ROOT, SECRETS_ROOT):
        path.mkdir(parents=True, exist_ok=True)
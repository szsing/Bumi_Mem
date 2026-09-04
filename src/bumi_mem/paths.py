"""Runtime path resolution for a clean Git repository."""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = Path(os.getenv("MEM0_WORKSPACE_ROOT", str(PROJECT_ROOT.parent))).expanduser().resolve()
RUNTIME_ROOT = Path(os.getenv("MEM0_RUNTIME_ROOT", str(WORKSPACE_ROOT / "mem0-runtime"))).expanduser().resolve()
MODEL_ROOT = Path(os.getenv("MEM0_MODEL_ROOT", str(WORKSPACE_ROOT / "models"))).expanduser().resolve()
DATA_ROOT = Path(os.getenv("MEM0_DATA_ROOT", str(RUNTIME_ROOT / "data"))).expanduser().resolve()
RESULTS_ROOT = Path(os.getenv("MEM0_RESULTS_ROOT", str(RUNTIME_ROOT / "results"))).expanduser().resolve()
LOG_ROOT = Path(os.getenv("MEM0_LOG_ROOT", str(RUNTIME_ROOT / "logs"))).expanduser().resolve()
SECRETS_ROOT = Path(os.getenv("MEM0_SECRETS_ROOT", str(RUNTIME_ROOT / "secrets"))).expanduser().resolve()
CACHE_ROOT = RUNTIME_ROOT / "cache"
HOME_ROOT = RUNTIME_ROOT / "home"
TMP_ROOT = RUNTIME_ROOT / "tmp"
CONDA_ROOT = RUNTIME_ROOT / "conda"


def configure_runtime() -> None:
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
    os.environ.setdefault("MEM0_RUNTIME_ROOT", str(RUNTIME_ROOT))
    os.environ.setdefault("MEM0_MODEL_ROOT", str(MODEL_ROOT))
    os.environ.setdefault("MEM0_DATA_ROOT", str(DATA_ROOT))
    os.environ.setdefault("MEM0_RESULTS_ROOT", str(RESULTS_ROOT))
    os.environ.setdefault("MEM0_LOG_ROOT", str(LOG_ROOT))
    os.environ.setdefault("MEM0_SECRETS_ROOT", str(SECRETS_ROOT))
    os.environ["MEM0_TELEMETRY"] = "False"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    for name, path in paths.items():
        os.environ[name] = str(path)
        path.mkdir(parents=True, exist_ok=True)
    for path in (DATA_ROOT, RESULTS_ROOT, LOG_ROOT, SECRETS_ROOT):
        path.mkdir(parents=True, exist_ok=True)

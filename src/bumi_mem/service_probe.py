"""服务预检：配置/设备检查和一次小型 LLM JSON 请求。

Does not create a memory database, download embedding weights or open a service
port. The actual embedding/Neo4j warmup is checked when the service starts.
"""
from __future__ import annotations

import argparse
import os
import sys
import json
import importlib.util
from datetime import datetime

from bumi_mem.long_term import resolve_embed_device
from bumi_mem.providers import provider_request_options
from bumi_mem.service_settings import ServiceSettings, memory_provider_config


def probe_llm(settings):
    from openai import OpenAI
    config = memory_provider_config(settings)
    with OpenAI(api_key=config.api_key, base_url=config.base_url, timeout=30, max_retries=0) as client:
        result = client.chat.completions.create(model=config.model,
            messages=[{'role': 'user', 'content': 'Return only this JSON object: {"ok": true}'}],
            response_format={'type': 'json_object'}, **provider_request_options(config))
        if json.loads(result.choices[0].message.content).get('ok') is not True:
            raise ValueError('LLM did not return the requested JSON')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--skip-llm', action='store_true', help='Do not make the small external API request')
    args = parser.parse_args()
    try:
        from bumi_mem import __version__
        print(f"Bumi_Mem version: {__version__}; Python: {sys.executable}", flush=True)
        settings = ServiceSettings.from_env()
        required = ('fastapi', 'uvicorn', 'filelock', 'openai', 'graphiti_core', 'neo4j', 'torch', 'sentence_transformers')
        missing = [name for name in required if importlib.util.find_spec(name) is None]
        if missing:
            print('Missing Python modules: ' + ', '.join(missing))
            raise RuntimeError('activate/install a compatible service environment before probing the paid API')
        if not settings.embed_model_path.is_dir():
            raise ValueError('embedding directory is missing')
        device = resolve_embed_device(settings.embed_device)
        if device == 'cuda':
            import torch
            torch.ones(1, device='cuda').sum().item()
        print('Configuration valid; embedding directory exists.', flush=True)
        print(f'Embedding device selected: {device}; local time: {datetime.now().astimezone().isoformat()}', flush=True)
        if device == 'cpu':
            print('CPU selected: this does not verify the prior GPU/60 ms performance target.', flush=True)
        if args.skip_llm:
            print('External LLM check skipped; API key/model access NOT verified.', flush=True)
        else:
            print(f'Checking memory LLM {settings.memory_model} with one small JSON request (billable).', flush=True)
            probe_llm(settings)
            print('Memory LLM JSON request passed.', flush=True)
        print('Next: start the service and check /health/ready. Full embedding warmup and real memory quality are not proven by this probe.', flush=True)
    except Exception as exc:
        # 不打印原始供应商错误体/凭据
        status = getattr(exc, 'status_code', None)
        print(f'Preflight failed: {type(exc).__name__}; HTTP status={status}. Check environment, model path, CUDA and LLM credentials/model access. No raw key was printed.')
        raise SystemExit(1) from None


if __name__ == '__main__':
    main()
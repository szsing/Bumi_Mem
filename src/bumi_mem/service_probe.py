"""Server preflight: configuration/device checks and one small LLM JSON request.

Does not create a memory database, download embedding weights or open a service
port. The actual embedding/Qdrant warmup is checked when the service starts.
"""
from __future__ import annotations

import argparse
import os
import sys
import json
import importlib.util
from datetime import datetime

from bumi_mem.long_term import resolve_embed_device
from bumi_mem.providers import bind_provider_request_options, ensure_explicit_mem0_route
from bumi_mem.service_settings import ServiceSettings, memory_provider_config


def probe_llm(settings):
    ensure_explicit_mem0_route()
    os.environ["MEM0_TELEMETRY"] = "False"
    from mem0.utils.factory import LlmFactory
    config = memory_provider_config(settings)
    llm = LlmFactory.create('openai', {
        'model': config.model, 'api_key': config.api_key,
        'openai_base_url': config.base_url.rstrip('/'), 'temperature': 0.0, 'max_tokens': 64,
    })
    llm.client = llm.client.with_options(timeout=30.0, max_retries=0)
    generate = bind_provider_request_options(llm.generate_response, config)
    try:
        raw = generate(messages=[{'role': 'user', 'content': 'Return only this JSON object: {"ok": true}'}],
                       response_format={'type': 'json_object'})
        if not isinstance(raw, str) or json.loads(raw).get('ok') is not True:
            raise ValueError('LLM did not return the requested JSON')
    finally:
        llm.client.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--skip-llm', action='store_true', help='Do not make the small external API request')
    args = parser.parse_args()
    try:
        from bumi_mem import __version__
        print(f"Bumi_Mem version: {__version__}; Python: {sys.executable}", flush=True)
        settings = ServiceSettings.from_env()
        ensure_explicit_mem0_route()
        required = ('fastapi', 'uvicorn', 'filelock', 'openai', 'mem0', 'torch', 'sentence_transformers')
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
        # Do not print raw vendor error bodies/credentials.
        status = getattr(exc, 'status_code', None)
        print(f'Preflight failed: {type(exc).__name__}; HTTP status={status}. Check environment, model path, CUDA and LLM credentials/model access. No raw key was printed.')
        if os.getenv('OPENROUTER_API_KEY'):
            print('OPENROUTER_API_KEY is set: unset it in this service shell to prevent SDK route override.')
        raise SystemExit(1) from None


if __name__ == '__main__':
    main()

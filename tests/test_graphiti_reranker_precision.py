"""Legacy precision settings remain parseable; RRF must never load a reranker."""
import asyncio
import logging
import sys
from types import SimpleNamespace

import pytest

from bumi_mem.graphiti_memory import GraphitiConfig, GraphitiLongTermMemory
from bumi_mem.providers import ProviderConfig
from bumi_mem.service_settings import ServiceSettings


@pytest.mark.parametrize('value,expected', [
    (None, True), ('on', True), ('true', True), ('off', False), ('false', False),
])
def test_service_reads_reranker_switch(tmp_path, monkeypatch, value, expected):
    for key, item in {
        'BUMI_DATA_DIR':str(tmp_path), 'BUMI_EMBED_MODEL_PATH':str(tmp_path),
        'BUMI_RERANKER_MODEL_PATH':str(tmp_path), 'BUMI_NEO4J_PASSWORD':'test-only',
        'BUMI_SERVICE_API_KEY':'test-service-key-at-least-32-characters',
        'BUMI_MEMORY_API_KEY':'test-only',
    }.items():
        monkeypatch.setenv(key,item)
    monkeypatch.delenv('BUMI_RERANKER_FP16',raising=False)
    if value is not None:
        monkeypatch.setenv('BUMI_RERANKER_FP16',value)
    assert ServiceSettings.from_env().reranker_fp16 is expected


def test_invalid_cli_switch_fails_before_runtime_side_effects(monkeypatch):
    from bumi_mem import cli
    monkeypatch.setenv('BUMI_RERANKER_FP16','maybe')
    monkeypatch.setattr(cli,'configure_runtime',lambda:pytest.fail('invalid config must fail first'))
    with pytest.raises(ValueError,match='BUMI_RERANKER_FP16'):
        cli.build_engine(SimpleNamespace())


@pytest.mark.parametrize('device,actual_device,enabled,expected_dtype', [
    ('auto','cuda',True,'float16'), ('cuda','cuda',False,'float32'),
    ('auto','cpu',True,'float32'), ('cpu','cpu',True,'float32'),
    ('auto','mps',True,'float32'),
])
def test_backend_precision_uses_loaded_device(tmp_path,monkeypatch,caplog,
        device,actual_device,enabled,expected_dtype):
    class Model:
        device = SimpleNamespace(type=actual_device)
        dtype = 'float32'
        def half(self):
            self.dtype = 'float16'
        def float(self):
            self.dtype = 'float32'
    class CrossEncoder:
        def __init__(self,path,**kwargs):
            assert kwargs['device'] == (None if device=='auto' else device)
            assert kwargs['local_files_only'] is True
            pytest.fail('RRF must not construct a BGE CrossEncoder')
    class SentenceTransformer:
        def __init__(self,*args,**kwargs):
            pass
        def get_sentence_embedding_dimension(self):
            return 1024
    class LLM:
        def __init__(self,**kwargs):
            self.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **kw:None)))
    class Graphiti:
        def __init__(self,*args,**kwargs):
            self.cross_encoder = kwargs['cross_encoder']
    modules = {
        'graphiti_core':SimpleNamespace(Graphiti=Graphiti),
        'graphiti_core.llm_client.config':SimpleNamespace(LLMConfig=SimpleNamespace),
        'graphiti_core.embedder.client':SimpleNamespace(EmbedderClient=object,EmbedderConfig=SimpleNamespace),
        'graphiti_core.cross_encoder.client':SimpleNamespace(CrossEncoderClient=object),
        'bumi_mem.graphiti_llm':SimpleNamespace(ValidatedGraphitiClient=LLM),
        'sentence_transformers':SimpleNamespace(CrossEncoder=CrossEncoder,SentenceTransformer=SentenceTransformer),
    }
    for name,module in modules.items():
        monkeypatch.setitem(sys.modules,name,module)
    config = GraphitiConfig('bolt://test','neo4j','test-only',tmp_path,tmp_path,tmp_path,
        device=device,reranker_fp16=enabled)
    provider = ProviderConfig('deepseek','test-model','https://example.invalid/v1','test-only')
    with caplog.at_level(logging.INFO,logger='bumi_mem.graphiti_memory'):
        backend = GraphitiLongTermMemory(provider,config)
    assert not hasattr(backend.client.cross_encoder,'model')
    with pytest.raises(RuntimeError,match='disabled'):
        asyncio.run(backend.client.cross_encoder.rank('query',['passage']))
    assert 'BGE reranker loaded' not in caplog.text
    assert 'test-only' not in caplog.text

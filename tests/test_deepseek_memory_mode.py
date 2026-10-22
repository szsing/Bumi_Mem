"""测试 DeepSeek 记忆模式：验证 DeepSeek 提供者的配置和路由逻辑。"""
from types import SimpleNamespace
import json
import ast
import os
from pathlib import Path
import typing
import types
import sys
import pytest

from bumi_mem.providers import ProviderConfig, OpenAICompatibleProvider, bind_provider_request_options, provider_request_options
from bumi_mem.service_settings import ServiceSettings, memory_provider_config
from bumi_mem.long_term import Mem0AdapterConfig, Mem0LongTermMemory
from bumi_mem.cli import _provider_config


def test_service_deepseek_memory_disables_thinking_and_uses_current_default(tmp_path):
    settings = ServiceSettings(data_dir=tmp_path, embed_model_path=tmp_path/'model',
                               api_key='test-service-key-with-at-least-32-chars', memory_api_key='secret')
    assert settings.memory_model == 'deepseek-flash'
    assert memory_provider_config(settings).thinking_mode == 'disabled'
    assert 'secret' not in repr(memory_provider_config(settings))


def test_cli_memory_opt_in_does_not_change_chat_thinking_policy(monkeypatch):
    monkeypatch.setenv('BUMI_MEMORY_API_KEY', 'test')
    memory = _provider_config('deepseek', purpose='memory', model_override='deepseek-flash', base_url_override=None, max_tokens=4096)
    chat = _provider_config('deepseek', purpose='chat', model_override='chat-model', base_url_override=None, max_tokens=100)
    assert memory.thinking_mode == 'disabled' and chat.thinking_mode is None


def test_options_are_not_leaked_to_other_providers():
    assert provider_request_options(ProviderConfig('local', 'model', 'http://local', 'test')) == {}
    fn = lambda **kwargs: kwargs
    config = ProviderConfig('deepseek', 'deepseek-flash', 'http://example', 'test', thinking_mode='disabled')
    result = bind_provider_request_options(fn, config)(extra_body={'custom': 1, 'thinking': {'type':'enabled'}})
    assert result['extra_body'] == {'custom': 1, 'thinking': {'type': 'disabled'}}
    assert fn(extra_body={}) == {'extra_body': {}}  # No global function mutation.


def test_summary_forwards_non_thinking_request_options_without_network():
    captured = []
    def create(**kwargs):
        captured.append(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content='ok'))])
    provider = object.__new__(OpenAICompatibleProvider)
    provider.config = ProviderConfig('deepseek','deepseek-flash','https://example.test','test',thinking_mode='disabled')
    provider.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    assert provider.generate([{'role':'user','content':'hi'}]) == 'ok'
    assert captured[0]['extra_body']['thinking'] == {'type':'disabled'}


def test_pinned_mem0_generation_source_forwards_options(monkeypatch):
    monkeypatch.delenv('OPENROUTER_API_KEY', raising=False)
    root = Path(__file__).resolve().parents[1] / 'src/mem0-new/mem0/llms'
    if not (root / 'base.py').is_file():
        pytest.skip('Optional historical Mem0 submodule is absent; Graphiti does not depend on it')
    methods = []
    selected = {'_is_reasoning_model','_uses_max_completion_tokens','_get_supported_params',
                '_get_common_params','_parse_response','generate_response'}
    for filename in ('base.py','openai.py'):
        tree = ast.parse((root / filename).read_text(encoding='utf8'))
        cls = next(n for n in tree.body if isinstance(n,ast.ClassDef))
        methods.extend(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name in selected and
                       not (filename=='base.py' and n.name=='generate_response'))
    node = ast.ClassDef(name='PinnedLLM',bases=[],keywords=[],body=methods,decorator_list=[])
    namespace = dict(Dict=typing.Dict,List=typing.List,Optional=typing.Optional,os=os,json=json)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])), '<pinned Mem0 source>', 'exec'),namespace)
    captured=[]
    def create(**kwargs):
        captured.append(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content='{"ok":true}'))])
    llm=namespace['PinnedLLM']()
    llm.config=SimpleNamespace(model='deepseek-flash',temperature=0.0,top_p=0.1,max_tokens=64,
                               is_reasoning_model=False,store=None,response_callback=None)
    llm.client=SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    cfg=ProviderConfig('deepseek','deepseek-flash','https://example.test','test',thinking_mode='disabled')
    generate=bind_provider_request_options(llm.generate_response,cfg)
    assert json.loads(generate(messages=[{'role':'user','content':'Return JSON'}],response_format={'type':'json_object'}))['ok']
    assert captured[0]['extra_body']['thinking']['type']=='disabled'
    assert captured[0]['response_format']=={'type':'json_object'}


def test_mem0_adapter_binds_only_its_configured_instance(tmp_path, monkeypatch):
    monkeypatch.delenv('OPENROUTER_API_KEY', raising=False)
    calls = []
    original = lambda **kwargs: calls.append(kwargs) or '[]'
    fake = SimpleNamespace(llm=SimpleNamespace(generate_response=original))
    module=types.ModuleType('mem0')
    module.Memory=SimpleNamespace(from_config=lambda config: fake)
    monkeypatch.setitem(sys.modules,'mem0',module)
    config = ProviderConfig('deepseek','deepseek-flash','https://example.test','test',thinking_mode='disabled')
    adapter = Mem0LongTermMemory(config, Mem0AdapterConfig(data_dir=tmp_path,embed_model=tmp_path/'model',embed_device='cpu'))
    adapter.memory.llm.generate_response(messages=[])
    assert calls[0]['extra_body']['thinking']['type'] == 'disabled'
    original(messages=[])
    assert 'extra_body' not in calls[1]


def test_ambient_router_key_cannot_redirect_memory_calls(monkeypatch):
    from bumi_mem.providers import ensure_explicit_mem0_route
    from bumi_mem.schemas import ProviderConfigurationError
    monkeypatch.setenv('OPENROUTER_API_KEY','not-a-real-key')
    with pytest.raises(ProviderConfigurationError, match='override'):
        ensure_explicit_mem0_route()


def test_probe_uses_bounded_json_call_and_closes_client(tmp_path, monkeypatch):
    from bumi_mem.service_probe import probe_llm
    monkeypatch.delenv('OPENROUTER_API_KEY', raising=False)
    calls = []
    class Client:
        closed = False
        def __init__(self, **kwargs):
            assert kwargs['timeout'] == 30 and kwargs['max_retries'] == 0
            def create(**kw):
                calls.append(kw)
                return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content='{"ok":true}'))])
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=create))
        def __enter__(self):
            return self
        def __exit__(self, *args):
            self.closed = True
    client = Client(timeout=30, max_retries=0)
    module = types.ModuleType('openai')
    module.OpenAI = lambda **kwargs: client
    monkeypatch.setitem(sys.modules, 'openai', module)
    settings = ServiceSettings(data_dir=tmp_path, embed_model_path=tmp_path/'model',
                               api_key='test-service-key-with-at-least-32-chars', memory_api_key='secret')
    probe_llm(settings)
    assert calls[0]['extra_body']['thinking'] == {'type':'disabled'}
    assert calls[0]['response_format'] == {'type':'json_object'}
    assert client.closed


def test_probe_fails_before_network_when_dependencies_missing(tmp_path, monkeypatch, capsys):
    import bumi_mem.service_probe as probe
    monkeypatch.delenv('OPENROUTER_API_KEY', raising=False)
    settings = ServiceSettings(data_dir=tmp_path, embed_model_path=tmp_path/'model',
                               api_key='test-service-key-with-at-least-32-chars', memory_api_key='do-not-print-me')
    monkeypatch.setattr(probe.ServiceSettings, 'from_env', lambda: settings)
    monkeypatch.setattr(probe.importlib.util, 'find_spec', lambda name: None)
    monkeypatch.setattr(sys, 'argv', ['probe'])
    monkeypatch.setattr(probe, 'probe_llm', lambda _: pytest.fail('must not call external API'))
    with pytest.raises(SystemExit) as exc:
        probe.main()
    assert exc.value.code == 1
    output = capsys.readouterr().out
    assert 'Missing Python modules' in output and 'do-not-print-me' not in output
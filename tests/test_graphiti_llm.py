"""Transport-shape failures use one upstream retry budget, never write retries."""
import asyncio
import json
from types import SimpleNamespace

import pytest

pytest.importorskip('graphiti_core')
from graphiti_core.llm_client.client import LLMClient
from graphiti_core.llm_client.config import LLMConfig
from graphiti_core.prompts.models import Message
from graphiti_core.prompts.dedupe_edges import EdgeDuplicate
from tenacity import wait_none
from bumi_mem.graphiti_llm import ValidatedGraphitiClient, StructuredResponseError


def client_for(tmp_path, monkeypatch, bodies):
    monkeypatch.setattr(LLMClient._generate_response_with_retry.retry, 'wait', wait_none())
    requests = []
    async def create(**kwargs):
        requests.append(kwargs)
        body = bodies[min(len(requests)-1, len(bodies)-1)]
        if isinstance(body, Exception):
            raise body
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(body)))])
    sdk = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    client = ValidatedGraphitiClient(config=LLMConfig(api_key='not-a-real-key'), client=sdk,
        structured_output_mode='json_object', diagnostic_dir=tmp_path)
    return client, requests


def test_schema_echo_retries_only_generation_and_preserves_prompt(tmp_path, monkeypatch):
    good = {'duplicate_facts': [], 'contradicted_facts': [2]}
    client, requests = client_for(tmp_path, monkeypatch, [EdgeDuplicate.model_json_schema(), good])
    result = asyncio.run(client.generate_response(
        [Message(role='system',content='native system'),Message(role='user',content='native fact')],
        response_model=EdgeDuplicate))
    assert result == good
    assert len(requests) == 2
    assert requests[0] == requests[1]  # No corrective prompt or new parameters.
    records = [json.loads(p.read_text()) for p in tmp_path.glob('*.json')]
    failed = next(r for r in records if r['status']=='failed')
    assert failed['parsed_response'] == EdgeDuplicate.model_json_schema()
    assert {e['loc'][0] for e in failed['validation_errors']} == {'duplicate_facts','contradicted_facts'}
    assert all(p.stat().st_mode & 0o777 == 0o600 for p in tmp_path.glob('*.json'))


@pytest.mark.parametrize('body', [EdgeDuplicate.model_json_schema(),
    {'duplicate_facts': [], 'contradicted_facts': 'wrong type'}, ['wrong root']])
def test_persistent_invalid_shape_stops_after_upstream_four_attempts(tmp_path, monkeypatch, body):
    client, requests = client_for(tmp_path, monkeypatch, [body])
    with pytest.raises(StructuredResponseError):
        asyncio.run(client.generate_response([Message(role='user',content='fact')],response_model=EdgeDuplicate))
    assert len(requests) == 4
    assert len(list(tmp_path.glob('*.json'))) == 4


def test_unrelated_failures_are_not_retried_or_logged_with_secret(tmp_path, monkeypatch):
    client, requests = client_for(tmp_path, monkeypatch, [ValueError('vendor body with secret')])
    with pytest.raises(ValueError):
        asyncio.run(client.generate_response([Message(role='user',content='fact')], response_model=EdgeDuplicate))
    assert len(requests) == 1
    assert 'vendor body with secret' not in next(tmp_path.glob('*.json')).read_text()


def test_response_without_schema_is_returned_unchanged(tmp_path, monkeypatch):
    client, requests = client_for(tmp_path, monkeypatch, [{'value': 1}])
    assert asyncio.run(client.generate_response([Message(role='user',content='fact')])) == {'value':1}
    assert len(requests) == 1

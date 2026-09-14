"""Validate preparation artifacts, not future retrieval/policy behavior."""
import io
import json
from pathlib import Path
import sys
import pytest
from pydantic import ValidationError

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from tools.stage0.contracts import Scope, SourceSpan, SeedCase, MemoryView, MemoryBundle
from tools.stage0 import stage0

CASES=json.loads((ROOT/"tests/fixtures/stage0/cases.json").read_text(encoding="utf-8-sig"))

@pytest.mark.parametrize("case",CASES,ids=[c["case_id"] for c in CASES])
def test_seed_case_schema_only(case):
    parsed=SeedCase.model_validate(case)
    assert parsed.setup["synthetic"] is True
    assert 1 <= parsed.expected.target_phase <= 6


def test_fixture_ids_unique():
    assert len(CASES)==12
    assert len({c["case_id"] for c in CASES})==12
    for case in CASES:
        obs=case["observations"]+[case["query"]]
        assert len({o["observation_id"] for o in obs})==len(obs)


def test_verified_scope_requires_actor():
    scope=dict(CASES[0]["query"]["scope"],actor_id=None)
    with pytest.raises(ValidationError): Scope.model_validate(scope)


def test_unknown_scope_cannot_claim_actor():
    scope=dict(CASES[0]["query"]["scope"],identity_status="unknown")
    with pytest.raises(ValidationError): Scope.model_validate(scope)


def test_contract_rejects_unrecognized_permission_field():
    scope=dict(CASES[0]["query"]["scope"],read_all_private_memories=True)
    with pytest.raises(ValidationError): Scope.model_validate(scope)


@pytest.mark.parametrize("start,end",[(3,3),(4,2),(-1,3)])
def test_span_order_and_nonnegative_offsets(start,end):
    with pytest.raises(ValidationError): SourceSpan(observation_id="obs",revision=1,start=start,end=end)


def test_bundle_preserves_real_degraded_status():
    bundle=MemoryBundle(request_id="r",scope=Scope.model_validate(CASES[0]["query"]["scope"]),retrieval_status="degraded",persistence_status="failed")
    assert bundle.retrieval_status=="degraded"
    assert not bundle.context_events


def test_run_path_rejects_live_data_directory():
    with pytest.raises(ValueError): stage0.run_path(stage0.REPO.parent/"mem0-runtime/data")


def test_api_smoke_without_key_never_opens_network(tmp_path,monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY",raising=False)
    def forbidden(*a,**k): raise AssertionError("network should not be attempted")
    monkeypatch.setattr(stage0.urllib.request,"build_opener",forbidden)
    assert stage0.api_smoke(tmp_path,"deepseek-flash")==2


def test_api_smoke_mocked_request_no_secret_persisted(tmp_path,monkeypatch):
    key="SYNTHETIC_TEST_CREDENTIAL_NOT_REAL"
    monkeypatch.setenv("DEEPSEEK_API_KEY",key)
    seen=[]
    class FakeOpener:
        def open(self,req,timeout):
            seen.append(req)
            body=json.loads(req.data)
            assert body["thinking"]=={"type":"disabled"}
            assert body["max_tokens"]==64
            return io.StringIO(json.dumps({"model":"mock-model","choices":[{"message":{"content":"STAGE0_OK"}}],"usage":{"total_tokens":2}}))
    monkeypatch.setattr(stage0.urllib.request,"build_opener",lambda *a:FakeOpener())
    (tmp_path/"reports").mkdir()
    assert stage0.api_smoke(tmp_path,"deepseek-flash")==0
    report=(tmp_path/"reports/deepseek_smoke.json").read_text()
    assert key not in report and "Authorization" not in report
    assert len(seen)==1


def test_unmeasured_timing_is_null_and_nonfinite_rejected():
    from tools.stage0.contracts import TimingBreakdown
    assert TimingBreakdown().memory_bundle_ms is None
    with pytest.raises(ValidationError): TimingBreakdown(memory_bundle_ms=float("nan"))

> 0.6.0只改变单次HTTP响应的表示：长期/短期按实际数据行结构化返回，不改变下述记忆所有权和生命周期语义。schema 9迁移见 `../structured_turn_response/README.md`。

> v0.4 adds schema 7 and an explicit Agent turn/event protocol; the v0.3 memory semantics below remain. See `../bumi_agent_integration/README.md` for the actual Pipecat 1.8 Agent integration.

# Companion memory contract (v0.3, SQLite schema 6)

Implemented against the product decisions confirmed on 2026-09-21. This document describes the current main Bumi/Mem0 path, **not** automatic integration of the separate `typed` library.

## 1. Identity, shared L1, and personal L3

- One `agent_id` is one shared room. The HTTP service remains single-process/single-worker.
- A trusted upstream assigns a stable known `user_id`. The reserved `__guest__` ID is the ONLY unknown-speaker category. Multiple strangers may share it; it does not imply they are one person.
- Guests can use shared L1, including personal L3 references already published by their owners. They cannot directly search/list/read/write personal L3, receive personal core state, or generate their own L3 consolidation. Their L1 headers contain `identity_status=unconfirmed`.
- Known A may retrieve only A's L3 for that agent. Successful recall is attached to A's current user message in `shared_memory_refs`, after that owner gate. B and guests may see the reference in the shared window. There is intentionally no additional public-disclosure gate.
- References are separate DATA, not user/assistant speech. They carry owner, source ID/version, original time and reference/fact status. They are never part of extraction snapshots. Seeing A's fact does not make it B's fact.
- A reference follows its original L1 anchor, including carry-over. It is not another dialogue message and does not renew merely because the prompt is rebuilt or its retrieval score changes. A later fresh retrieval after expiry (including token-budget clipping of the old anchor) may publish a new reference.
- The default window remains 20 effective messages plus up to 4 carry-over messages, shared across users. Token selection can shorten it. Referenced data has a 6,000-character prompt budget (newest fitting records first; never truncate a fact mid-sentence). Stored references expire with anchors, not an extra clock.
- Ordinary assistant retellings are not independent evidence of user facts. Source/owner prompt rules remain necessary: actual LLM extraction quality is not proven by fake-provider tests.

## 2. Time: preserve speech, normalize long-term input

Every L1 message is rendered with its original offset-aware `created_at`. Original content is immutable.

The context request optionally accepts:

```json
{
  "user_id": "A",
  "previous_ai_reply": null,
  "query": "昨天我去了一趟游乐场",
  "query_occurred_at": "2026-09-21T14:30:00+08:00"
}
```

`previous_reply_occurred_at` likewise describes the **actually spoken** previous reply. Require ISO 8601 timestamps with an offset; timestamp fields require the associated nonempty message. Network retries must preserve the full request body, including timestamps. If absent, time is the server's local-offset receipt time, **not a claim that actual playback or speech time is known**. Configure the device/server clock and local timezone; pass upstream timestamps for delayed delivery or replay.

Extraction views include original source ID, original content and observed time. Simple `昨天/今天/明天` and `yesterday/today/tomorrow` become absolute calendar dates relative to each message's local date. For the example above the normalized body is `2026-09-20我去了一趟游乐场`. The LLM extraction instructions require durable prose using the absolute date and the correct owner.

Explicit `remember()` writes also normalize simple relative dates; callers may supply its optional `observed_at`, otherwise server receipt time is used and original contents are retained in metadata.

Vague `最近/recently`, unknown historical dates and unsupported relative expressions retain their wording and source anchor. No guessed deadline or blanket event date is assigned. `created_at`/`updated_at` on a vector record are storage timestamps, not event time. Retrieval rendering includes available observed/event time and labels update time explicitly. No independent temporal expiry state machine is added: the answering LLM reasons from the preserved evidence.

Legacy pending extraction chunks are matched against immutable original snapshots before retry. Missing/mismatched provenance fails closed for explicit source repair instead of inventing dates. Each extraction chunk also has a separate Mem0 `run_id` within its deletion generation, preventing the SDK's previous-input cache from resurrecting withdrawn sources under a new chunk's metadata.

## 3. Confirmed core and semantic state

`semantic_sources`, immutable `semantic_versions`, and `semantic_current` implement a small, directly readable per-user state layer, independent of embedding availability. Updates and reference invalidation/context-epoch revocation commit together. Reusing a source with different text/time is rejected.

The first version is deliberately a **bounded exact-form recognizer**, not a general NL contradiction solver. It makes no extra LLM call on the critical path. Supported examples:

| Input | Result |
|---|---|
| `以后叫我林老师` / `请叫我小林` | Confirmed preferred name |
| `别再叫我宝宝，我不喜欢` | Forbidden form of address |
| `请不要评价我的外貌` / `请不要对我开玩笑` | Interaction boundary |
| `今天别和我聊工作的事` / `最近别聊工作` | Boundary with original temporal qualifier |
| `我喜欢咖啡` / `我不喜欢咖啡` | Explicit finite-object preference; history retained |
| `我现在不喝咖啡了` | Explicit drinking-state update; `现在/now` denotes a current update rather than a separate conditional key |
| `我现在住在上海` / `我在北京工作` | Narrow explicit residence/work-location fact |
| `确认计划：公园；状态：计划中` / `确认计划：公园；状态：已取消` | Source-backed plan status |
| `确认我的承诺：练习；状态：已完成` | Explicit user-promise status, not inferred completion |
| Actually spoken assistant `我答应你陪你练习` | Robot promise with assistant provenance, status `promised` |
| `确认机器人承诺：陪你练习；状态：已完成` | Explicit user confirmation of the matching promise status, with user provenance |

Selected English equivalents are supported; see `semantic_state.py` and its tests for the exact grammar. Quotes, reported speech, questions, explicit jokes, ambiguous/multi-clause/unrecognized requests do not silently update core. Unsupported wording remains in L1 and may follow the ordinary L3 extraction path, but is **not** a confirmed core write. Do not assume every natural-language request is recognized.

Limits: source text 512 characters, value 240 characters, 64 current state keys (including retraction/status tombstones). Capacity rejects new keys without evicting existing boundaries; existing keys can still update. Rejections preserve previously readable core rules and return a failed receipt.

Plans/promises track explicit states only; dates passing do not complete them. No scheduler, task execution, reminder delivery or fulfilment guarantee is implemented. Actually spoken promises are recorded, never treated as user assertions. There is no automatic general-language robot-promise completion inference.

A clear recognized update creates a new version, retains history, invalidates narrowly recognized conflicting L3 references, and prefers current state for ordinary recall. Explicit history questions can read semantic versions (the prompt includes up to 16 recent versions) and retrieve conflicting L3 as `historical_superseded`. Cross-language/paraphrased/complex conflicts may remain unresolved; the current state and timestamps are available to the answer model, but this is not a universal contradiction resolver.

## 4. Write acknowledgement and response contract

`POST /v1/memory/context` is still the only conversational endpoint. Responses add:

```json
{"core_write": {"status": "persisted", "changes": [], "source": {}}}
```

The example shows shape only; a real persisted result contains actual changes. `persisted` is emitted only after the core transaction commits and current state can be read. `not_requested` means no recognized core mutation, not implicit success. `failed` means persistence/read confirmation could not be completed. Generic queued L3 consolidation remains unrelated to this receipt.

Agent integration MUST:

1. Replace the old memory-provided messages with the current returned messages; do not append duplicate query/history or keep unbounded copies of expired shared references.
2. Only make a durable-memory acknowledgement for a confirmed `core_write` change or an explicit `remember()` read-back receipt.
3. On a failed core change, obey the current explicit request within the conversation, preserve unrelated existing rules, and briefly explain that long-term saving was not completed.
4. Not promise proactive reminders merely because a plan/promise record exists.

## 5. Deletion and stale replies

- `forget_all()` advances the owner's generation, suppresses reads immediately, cancels old jobs, excludes old messages from new extraction, removes semantic sources/history/current state, invalidates published references, and queues verified backend deletion.
- Context epochs are per agent because a cached B prompt may contain A's published reference. Deletion/correction invalidates affected cached contexts via the agent epoch. A retry of stale context returns HTTP 409 `shared_memory_context_changed`; do not regenerate or play an old prompt.
- Already spoken responses can still be recorded as shared speech. A delayed response based on a revoked prompt is marked `memory_excluded`, even when its owner's personal generation did not change.
- `forget_memory()` (trusted Python API and CLI `/forget <memory-id>`) performs **logical, source-aware withdrawal**, not physical erasure. It tombstones the record, invalidates references, suppresses source-linked core and future reads, cancels affected pending jobs, and excludes the evidence set from later extraction. Late writes carrying those old source IDs are hidden even under new vector IDs.
- Extraction lineage is currently chunk-level. Withdrawing one extracted record can therefore conservatively suppress other records sharing that evidence set; it does not pretend to have per-fact source-span precision.
- Legacy records without reliable source lineage require `forget_all()` instead of an unsafe targeted delete. Explicit `remember()` records can be individually withdrawn. Use full deletion for backend physical erasure. Raw already-spoken L1/history and copies held by external Agents/listeners are not automatically erased by reference withdrawal.
- Management deletion is not exposed as an unauthenticated new HTTP endpoint, nor is arbitrary ambiguous chat text automatically turned into a delete command.

## 6. Timing and deployment

Normal serial memory-read deadline remains `BUMI_MEMORY_READ_TIMEOUT=0.2`. Explicit recall phrases can use `BUMI_EXPLICIT_RECALL_TIMEOUT=1.0`; both are configurable wait/degradation policies, **not measured retrieval latency**. The user's target of retrieval below 60 ms requires verification on the actual embedding/Qdrant deployment. Core reads do not depend on that queue. SQLite overhead, reference rendering, network and model generation are separate metrics. 实测与差距见 [时延压测报告](../latency/REPORT.md)。

Schema 6 adds tables and nullable/defaulted columns, preserving existing messages, sessions and jobs. Before upgrading a real database, stop the service and take a consistent SQLite backup (and the L3 backend backup according to its deployment procedure). The older binary must not open a schema-6 database; rollback means restoring a pre-upgrade backup, not lowering `user_version`. Test only on a copied database before production use.

This change does not upgrade the pinned Mem0 submodule, deploy the service, download models, or claim real GPU/DeepSeek/Qdrant acceptance. The independent `typed` foundation is not automatically wired into this path.

## 7. Offline acceptance

Run `python -m pytest -q` from the repository's configured environment. New tests cover shared guest access without direct L3 access, window expiry/no renewal, dated delayed delivery, core restart/write failure, clear correction/history, source withdrawal, cross-user stale cache/reply handling, plan state, actual assistant-promise provenance, and timestamp request validation. Existing lease/retry/migration tests remain.

Real model extraction/answer quality, real retrieval latency, voice identity accuracy, playback-order integration, crash/power-loss behavior and production rollout must be separately accepted. Passing fake-provider tests does not establish those results.

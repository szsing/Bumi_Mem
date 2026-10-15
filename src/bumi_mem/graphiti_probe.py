"""Server-only Graphiti integration checks (real model calls, no fake success)."""
import argparse
import json
import time
import uuid

from bumi_mem.service_settings import ServiceSettings, build_service_engine


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--query', default='小星喜欢什么动物？')
    parser.add_argument('--user-id', default=None, help='Reuse this ID with --recall-only after restart')
    parser.add_argument('--recall-only', action='store_true')
    args = parser.parse_args()
    settings = ServiceSettings.from_env()
    user = args.user_id or 'graphiti_probe_' + uuid.uuid4().hex[:12]
    engine = build_service_engine(settings)
    try:
        engine.start()
        print('Probe user:', user, flush=True)
        if not args.recall_only:
            # This does not exercise the chat model: supplied assistant text
            # settles real completed turns through the production turn protocol.
            for index in range(settings.extraction_turns):
                query = '我叫小星，我最喜欢三角龙。' if index == 0 else f'我们继续聊三角龙，这是第{index+1}轮。'
                engine.process_turn_context(turn_id='probe_' + uuid.uuid4().hex,
                    user_id=user, agent_id=settings.agent_id, query=query,
                    previous_ai_reply=None if index == 0 else '我们继续聊你喜欢的恐龙。')
            engine.process_turn_context(turn_id='probe_' + uuid.uuid4().hex,
                user_id=user, agent_id=settings.agent_id, query=None,
                previous_ai_reply='下次我们还可以聊三角龙。')
            deadline = time.monotonic() + 600
            while time.monotonic() < deadline:
                jobs = engine.memory_jobs(user_id=user, agent_id=settings.agent_id)
                if jobs and all(j.status == 'succeeded' for j in jobs):
                    break
                if any(j.status == 'failed' for j in jobs):
                    raise RuntimeError('Ingestion failed; inspect /v1/memory/debug and jobs; do not blindly replay')
                time.sleep(1)
            else:
                raise TimeoutError('Ingestion did not finish within 600 seconds')
        # Direct backend retrieval deliberately bypasses L1 and semantic_state.
        rows = engine.search_long_term(user_id=user, agent_id=settings.agent_id,
                                      query=args.query, read_timeout=120)
        print(json.dumps(rows, ensure_ascii=False, indent=2))
        print(json.dumps(engine.long_term.debug(user_id=user, agent_id=settings.agent_id), ensure_ascii=False, indent=2))
        if not rows:
            raise RuntimeError('No Graphiti recall results; investigate extraction and reranker scores')
        print('Graph retrieval returned records. Manually check they support the query; nonempty is not a quality score.')
        print(f'Restart probe: python -m bumi_mem.graphiti_probe --recall-only --user-id {user}')
    finally:
        engine.close()


if __name__ == '__main__':
    main()

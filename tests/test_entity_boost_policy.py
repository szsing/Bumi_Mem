"""实体加成开关：默认 on 不动 SDK；off 时只短路 mem0 的内部方法，且 SDK 不匹配要显式报错。"""
from __future__ import annotations

import pytest

from bumi_mem.long_term import _apply_entity_boost_policy


class FakeMemory:
    def _compute_entity_boosts(self, query_entities, filters):
        return {"m1": 0.4}


class FakeMemoryWithoutHook:
    pass


def test_default_mode_is_on_and_keeps_sdk_behaviour() -> None:
    memory = FakeMemory()
    assert _apply_entity_boost_policy(memory, mode="on") == "on"
    assert memory._compute_entity_boosts([], {}) == {"m1": 0.4}


def test_off_short_circuits_the_hook() -> None:
    memory = FakeMemory()
    assert _apply_entity_boost_policy(memory, mode="off") == "off"
    assert memory._compute_entity_boosts([("person", "小林")], {}) == {}


def test_off_fails_loudly_when_sdk_hook_is_gone() -> None:
    with pytest.raises(RuntimeError, match="_compute_entity_boosts"):
        _apply_entity_boost_policy(FakeMemoryWithoutHook(), mode="off")

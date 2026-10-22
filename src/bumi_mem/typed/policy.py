"""纯授权决策：源/主体/受众/用途相互独立，不含 I/O 或副作用。"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class Requirement:
    """访问需求：描述一次记忆访问所需的记忆 ID、所有者、主体和属性。"""
    memory_id: str
    owner_id: str
    subjects: frozenset[str]
    attributes: frozenset[str]


@dataclass(frozen=True)
class Rule:
    """授权规则：定义某主体对某受众在某用途下对某类别的允许/拒绝策略。

    属性:
        subject_id:        授权主体 ID
        audience:          授权受众集合
        effect:            效果（'allow'/'deny'）
        category:          记忆类别（'identity'/'behavior'/'episode'/'record_index'/'*'）
        attributes:        涉及的属性集合
        purposes:          允许的用途集合
        target_memory_id:  目标记忆 ID（None 表示适用于所有记忆）
        block_inference:   是否阻止基于此记忆的推断
    """
    subject_id: str
    audience: frozenset[str]
    effect: str
    category: str
    attributes: frozenset[str]
    purposes: frozenset[str]
    target_memory_id: str | None
    block_inference: bool


class PolicyEngine:
    """策略引擎：基于规则集合判定访问是否被允许。

    决策逻辑：
    - 遍历所有需求和所有权威主体
    - 对每个主体-受众对，收集匹配的规则
    - 若存在 deny 规则且属性匹配（或 block_inference），则拒绝
    - 若为共享访问，还需确认所有属性都被 allow 规则显式授权
    - 仅私有 owner-only 读取可获得隐式允许
    """
    @staticmethod
    def permits(*, requirements: Iterable[Requirement], rules: Iterable[Rule], recipients: set[str], purpose: str, category: str, shared: bool) -> bool:
        requirements, rules = tuple(requirements), tuple(rules)
        if not requirements or not recipients:
            return False
        for requirement in requirements:
            authorities = set(requirement.subjects) | {requirement.owner_id}
            for subject in authorities:
                for recipient in recipients:
                    relevant = [r for r in rules if r.subject_id == subject
                                and recipient in r.audience and purpose in r.purposes
                                and r.category in (category, '*')
                                and r.target_memory_id in (None, requirement.memory_id)]
                    if any(r.effect == 'deny' and (r.block_inference or '*' in r.attributes
                           or bool(requirement.attributes & r.attributes)) for r in relevant):
                        return False
                    # 仅私有所有者独享读取获得隐式允许。共享仍需要
                    # 所有者和每个涉及的主体授予
                    if shared:
                        allowed = set().union(*(r.attributes for r in relevant if r.effect == 'allow'))
                        if not requirement.attributes <= allowed:
                            return False
        return True
"""Pure authorization decisions. Source/subject/audience/purpose are independent."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class Requirement:
    memory_id: str
    owner_id: str
    subjects: frozenset[str]
    attributes: frozenset[str]


@dataclass(frozen=True)
class Rule:
    subject_id: str
    audience: frozenset[str]
    effect: str
    category: str
    attributes: frozenset[str]
    purposes: frozenset[str]
    target_memory_id: str | None
    block_inference: bool


class PolicyEngine:
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
                    # Only private owner-only reads get an implicit allow. Sharing
                    # still requires the owner and every involved subject to grant it.
                    if shared:
                        allowed = set().union(*(r.attributes for r in relevant if r.effect == 'allow'))
                        if not requirement.attributes <= allowed:
                            return False
        return True

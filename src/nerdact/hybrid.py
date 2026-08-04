"""Deterministic overlap resolution for model and rule PII spans."""

from __future__ import annotations

from dataclasses import dataclass

from .schema import Entity

_DETECTOR_LABEL_PRIORITY = {
    "URL": 100,
    "EMAIL_ADDRESS": 90,
    "CREDENTIAL": 80,
    "GOVERNMENT_ID": 70,
    "FINANCIAL_ID": 70,
    "DEVICE_ID": 70,
    "IP_ADDRESS": 60,
    "PHONE_NUMBER": 60,
}


@dataclass(frozen=True, slots=True)
class ResolutionResult:
    entities: tuple[Entity, ...]
    rejected: tuple[Entity, ...]


def _is_detector(entity: Entity) -> bool:
    return any(source.startswith("detector:") for source in entity.provenance)


def _overlaps(left: Entity, right: Entity) -> bool:
    return left.start < right.end and right.start < left.end


def _candidate_priority(entity: Entity) -> tuple[int, int, float, int, int, int, str]:
    detector = _is_detector(entity)
    return (
        int(detector),
        _DETECTOR_LABEL_PRIORITY.get(entity.label, 0) if detector else 0,
        float(entity.score or 0),
        entity.end - entity.start,
        -entity.start,
        -entity.end,
        entity.label,
    )


def _merge_exact(entities: list[Entity]) -> list[Entity]:
    merged: dict[tuple[int, int, str], Entity] = {}
    for entity in entities:
        previous = merged.get(entity.key())
        if previous is None:
            merged[entity.key()] = entity
            continue
        scores = [score for score in (previous.score, entity.score) if score is not None]
        merged[entity.key()] = Entity(
            entity.start,
            entity.end,
            entity.label,
            entity.text,
            max(scores) if scores else None,
            tuple(sorted(set(previous.provenance + entity.provenance))),
        )
    return list(merged.values())


def resolve_pii_overlaps(entities: list[Entity]) -> ResolutionResult:
    """Resolve a flat redaction set while retaining every rejected conflict for audit."""
    accepted: list[Entity] = []
    rejected: list[Entity] = []
    for candidate in sorted(_merge_exact(entities), key=_candidate_priority, reverse=True):
        if any(_overlaps(candidate, existing) for existing in accepted):
            rejected.append(candidate)
        else:
            accepted.append(candidate)
    return ResolutionResult(tuple(sorted(accepted)), tuple(sorted(rejected)))

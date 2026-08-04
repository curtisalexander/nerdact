"""Deterministic typed-placeholder redaction."""

from collections import defaultdict

from .schema import Entity, validate_entities


def redact(
    text: str, entities: list[Entity] | tuple[Entity, ...]
) -> tuple[str, list[tuple[Entity, str]]]:
    """Assign placeholders in reading order, reusing identical label/text values."""
    ordered = sorted(entities)
    validate_entities(text, ordered)
    if any(a.end > b.start for a, b in zip(ordered, ordered[1:], strict=False)):
        raise ValueError("cannot redact overlapping entities")
    counts: dict[str, int] = defaultdict(int)
    assigned: dict[tuple[str, str], str] = {}
    replacements: list[tuple[Entity, str]] = []
    for entity in ordered:
        key = (entity.label, entity.text)
        if key not in assigned:
            counts[entity.label] += 1
            assigned[key] = f"[{entity.label}_{counts[entity.label]}]"
        replacements.append((entity, assigned[key]))
    output = text
    for entity, placeholder in reversed(replacements):
        output = output[: entity.start] + placeholder + output[entity.end :]
    return output, replacements

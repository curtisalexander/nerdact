"""Core schema and strict JSONL loading."""

from __future__ import annotations

import json
from collections.abc import Collection
from dataclasses import dataclass
from pathlib import Path
from typing import Any

LABELS = ("PERSON", "ORGANIZATION", "LOCATION", "MISCELLANEOUS")
PII_LABELS = (
    "PERSON",
    "ADDRESS",
    "PHONE_NUMBER",
    "EMAIL_ADDRESS",
    "GOVERNMENT_ID",
    "FINANCIAL_ID",
    "CREDENTIAL",
    "DATE",
    "USERNAME",
    "IP_ADDRESS",
    "DEVICE_ID",
    "URL",
)


@dataclass(frozen=True, slots=True, order=True)
class Entity:
    start: int
    end: int
    label: str
    text: str
    score: float | None = None
    provenance: tuple[str, ...] = ()

    def key(self) -> tuple[int, int, str]:
        return (self.start, self.end, self.label)


@dataclass(frozen=True, slots=True)
class Example:
    id: str
    text: str
    entities: tuple[Entity, ...]
    note: str = ""


def validate_entities(text: str, entities: Collection[Entity]) -> None:
    """Reject entities that do not describe exact spans in their source text."""
    for entity in entities:
        if not isinstance(entity.start, int) or isinstance(entity.start, bool):
            raise ValueError("entity start must be an integer")
        if not isinstance(entity.end, int) or isinstance(entity.end, bool):
            raise ValueError("entity end must be an integer")
        if not (0 <= entity.start < entity.end <= len(text)):
            raise ValueError(
                f"invalid entity span [{entity.start}, {entity.end}) for text length {len(text)}"
            )
        if not isinstance(entity.label, str) or not entity.label:
            raise ValueError("entity label must be a non-empty string")
        if entity.text != text[entity.start : entity.end]:
            raise ValueError(
                f"entity text {entity.text!r} does not match {text[entity.start : entity.end]!r}"
            )


def _entity(data: dict[str, Any], text: str, example_id: str, labels: Collection[str]) -> Entity:
    if not isinstance(data, dict):
        raise ValueError(f"{example_id}: entity must be an object")
    try:
        start, end, label = data["start"], data["end"], data["label"]
    except KeyError as error:
        raise ValueError(f"{example_id}: malformed entity {data!r}") from error
    if (
        not isinstance(start, int)
        or isinstance(start, bool)
        or not isinstance(end, int)
        or isinstance(end, bool)
        or not isinstance(label, str)
    ):
        raise ValueError(f"{example_id}: malformed entity {data!r}")
    if label not in labels:
        raise ValueError(f"{example_id}: unsupported label {label!r}")
    if not (0 <= start < end <= len(text)):
        raise ValueError(f"{example_id}: invalid span [{start}, {end}) for text length {len(text)}")
    actual = text[start:end]
    supplied = data.get("text")
    if supplied is not None and not isinstance(supplied, str):
        raise ValueError(f"{example_id}: entity text must be a string")
    if supplied is not None and supplied != actual:
        raise ValueError(f"{example_id}: entity text {supplied!r} does not match {actual!r}")
    return Entity(start, end, label, actual)


def load_jsonl(path: str | Path, labels: Collection[str] = LABELS) -> list[Example]:
    """Load examples while validating IDs, offsets, text, labels, and overlap."""
    examples: list[Example] = []
    ids: set[str] = set()
    with Path(path).open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                data = json.loads(line)
                if not isinstance(data, dict):
                    raise ValueError("example must be an object")
                example_id, text = data["id"], data["text"]
                note = data.get("note", "")
                if not isinstance(example_id, str) or not isinstance(text, str):
                    raise ValueError("id and text must be strings")
                if not isinstance(note, str):
                    raise ValueError("note must be a string")
                raw_entities = data.get("entities", [])
                if not isinstance(raw_entities, list):
                    raise ValueError("entities must be a list")
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
                raise ValueError(f"line {line_number}: invalid example: {error}") from error
            if not example_id or example_id in ids:
                raise ValueError(f"line {line_number}: empty or duplicate id {example_id!r}")
            entities = tuple(
                sorted(_entity(item, text, example_id, labels) for item in raw_entities)
            )
            if any(
                left.end > right.start for left, right in zip(entities, entities[1:], strict=False)
            ):
                raise ValueError(f"{example_id}: entities overlap")
            ids.add(example_id)
            examples.append(Example(example_id, text, entities, note))
    return examples

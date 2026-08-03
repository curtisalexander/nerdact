"""Core schema and strict JSONL loading."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

LABELS = ("PERSON", "ORGANIZATION", "LOCATION", "MISCELLANEOUS")


@dataclass(frozen=True, slots=True, order=True)
class Entity:
    start: int
    end: int
    label: str
    text: str
    score: float | None = None

    def key(self) -> tuple[int, int, str]:
        return (self.start, self.end, self.label)


@dataclass(frozen=True, slots=True)
class Example:
    id: str
    text: str
    entities: tuple[Entity, ...]
    note: str = ""


def _entity(data: dict[str, Any], text: str, example_id: str) -> Entity:
    try:
        start, end, label = int(data["start"]), int(data["end"]), str(data["label"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"{example_id}: malformed entity {data!r}") from error
    if label not in LABELS:
        raise ValueError(f"{example_id}: unsupported label {label!r}")
    if not (0 <= start < end <= len(text)):
        raise ValueError(f"{example_id}: invalid span [{start}, {end}) for text length {len(text)}")
    actual = text[start:end]
    supplied = data.get("text")
    if supplied is not None and supplied != actual:
        raise ValueError(f"{example_id}: entity text {supplied!r} does not match {actual!r}")
    return Entity(start, end, label, actual)


def load_jsonl(path: str | Path) -> list[Example]:
    """Load examples while validating IDs, offsets, text, labels, and overlap."""
    examples: list[Example] = []
    ids: set[str] = set()
    with Path(path).open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                data = json.loads(line)
                example_id, text = str(data["id"]), str(data["text"])
                raw_entities = data.get("entities", [])
                if not isinstance(raw_entities, list):
                    raise ValueError("entities must be a list")
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
                raise ValueError(f"line {line_number}: invalid example: {error}") from error
            if not example_id or example_id in ids:
                raise ValueError(f"line {line_number}: empty or duplicate id {example_id!r}")
            entities = tuple(sorted(_entity(item, text, example_id) for item in raw_entities))
            if any(
                left.end > right.start for left, right in zip(entities, entities[1:], strict=False)
            ):
                raise ValueError(f"{example_id}: entities overlap")
            ids.add(example_id)
            examples.append(Example(example_id, text, entities, str(data.get("note", ""))))
    return examples

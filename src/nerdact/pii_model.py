"""GLiNER2-PII adapter with a checked canonical label mapping."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Any

from .schema import PII_LABELS, Entity

DEFAULT_PII_MODEL = "fastino/gliner2-privacy-filter-PII-multi"
DEFAULT_PII_REVISION = "59894c087cb2923b01f337d4ee72f6ff84d5bdd6"


@dataclass(frozen=True, slots=True)
class PIILabel:
    type: str
    placeholder: str
    model_labels: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class PIILabelSchema:
    name: str
    model: str
    labels: tuple[PIILabel, ...]

    def requested_labels(self) -> list[str]:
        return [model_label for label in self.labels for model_label in label.model_labels]

    def model_label_map(self) -> dict[str, str]:
        return {
            model_label.casefold(): label.type
            for label in self.labels
            for model_label in label.model_labels
        }


def load_pii_schema(path: str | Path) -> PIILabelSchema:
    """Load and validate the model-to-canonical PII label contract."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    try:
        labels = tuple(
            PIILabel(
                type=str(item["type"]),
                placeholder=str(item["placeholder"]),
                model_labels=tuple(str(label) for label in item["model_labels"]),
            )
            for item in data["labels"]
        )
        schema = PIILabelSchema(str(data["schema"]), str(data["model"]), labels)
    except (KeyError, TypeError) as error:
        raise ValueError("malformed PII label schema") from error
    types = [label.type for label in labels]
    model_labels = [label.casefold() for label in labels for label in label.model_labels]
    if len(types) != len(PII_LABELS) or set(types) != set(PII_LABELS):
        raise ValueError(f"PII schema must define each canonical label exactly once: {PII_LABELS}")
    if any(label.placeholder != label.type for label in labels):
        raise ValueError("placeholder mappings must preserve canonical typed placeholders")
    if any(not label.strip() for label in model_labels) or len(model_labels) != len(
        set(model_labels)
    ):
        raise ValueError("model labels must be non-empty and map to only one canonical label")
    return schema


def gliner2_predictions_to_entities(
    text: str,
    result: Mapping[str, Any],
    label_map: Mapping[str, str],
) -> list[Entity]:
    """Normalize GLiNER2's grouped native output and retain the best exact duplicate."""
    grouped = result.get("entities")
    if not isinstance(grouped, Mapping):
        raise ValueError("GLiNER2 result must contain an entities mapping")
    entities: dict[tuple[int, int, str], Entity] = {}
    for model_label, predictions in grouped.items():
        canonical = label_map.get(str(model_label).casefold())
        if canonical is None or not isinstance(predictions, list):
            continue
        for item in predictions:
            if not isinstance(item, Mapping):
                continue
            start, end = item["start"], item["end"]
            score = float(item["confidence"])
            if not math.isfinite(score) or not 0 <= score <= 1:
                raise ValueError("prediction score must be finite and between 0 and 1")
            if (
                not isinstance(start, int)
                or isinstance(start, bool)
                or not isinstance(end, int)
                or isinstance(end, bool)
                or not (0 <= start < end <= len(text))
            ):
                raise ValueError("prediction offsets must be valid integer source offsets")
            entity = Entity(start, end, canonical, text[start:end], score)
            key = entity.key()
            previous = entities.get(key)
            if previous is None or score > float(previous.score or 0):
                entities[key] = entity
    return sorted(entities.values())


class GLiNER2PIIAdapter:
    """Lazy adapter using GLiNER2's overlapping long-document extraction path."""

    def __init__(
        self,
        schema: PIILabelSchema,
        model: str = DEFAULT_PII_MODEL,
        revision: str | None = DEFAULT_PII_REVISION,
        threshold: float = 0.5,
        chunk_size: int = 384,
        chunk_overlap: int = 64,
    ) -> None:
        if model != schema.model:
            raise ValueError(f"PII schema is for {schema.model}, not {model}")
        if not 0 <= threshold <= 1:
            raise ValueError("threshold must be between 0 and 1")
        if chunk_size <= 0 or not 0 <= chunk_overlap < chunk_size:
            raise ValueError("chunk overlap must be non-negative and smaller than chunk size")
        self.schema = schema
        self.model_name = model
        self.revision = revision
        self.threshold = threshold
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._model: Any = None

    def _load(self) -> Any:
        if self._model is None:
            AutoExtractor = import_module("gliner2").AutoExtractor
            self._model = AutoExtractor.from_pretrained(self.model_name, revision=self.revision)
        return self._model

    def predict(self, text: str) -> list[Entity]:
        extractor = self._load()
        # The pinned GLiNER2 revision's extract_entities_long convenience wrapper
        # references an undefined overlap_policy. These are the same two public
        # calls that wrapper is intended to make before delegating to extract_long.
        extraction_schema = extractor.create_schema().entities(self.schema.requested_labels())
        result = extractor.extract_long(
            text,
            extraction_schema,
            threshold=self.threshold,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            include_confidence=True,
            include_spans=True,
            overlap_policy=None,
        )
        return [
            Entity(
                entity.start,
                entity.end,
                entity.label,
                entity.text,
                entity.score,
                (f"model:{self.model_name}",),
            )
            for entity in gliner2_predictions_to_entities(
                text, result, self.schema.model_label_map()
            )
        ]

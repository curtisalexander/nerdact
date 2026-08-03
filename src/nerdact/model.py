"""Lazy Hugging Face adapter with character spans as its public contract."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from .schema import Entity

DEFAULT_MODEL = "dslim/bert-base-NER"
DEFAULT_REVISION = "d1a3e8f13f8c3566299d95fcfc9a8d2382a9affc"
LABEL_MAP = {
    "PER": "PERSON",
    "PERSON": "PERSON",
    "ORG": "ORGANIZATION",
    "ORGANIZATION": "ORGANIZATION",
    "LOC": "LOCATION",
    "LOCATION": "LOCATION",
    "MISC": "MISCELLANEOUS",
    "MISCELLANEOUS": "MISCELLANEOUS",
}


def predictions_to_entities(
    text: str, predictions: Iterable[Mapping[str, Any]], threshold: float = 0.5
) -> list[Entity]:
    """Validate and normalize aggregated pipeline predictions."""
    entities: list[Entity] = []
    for item in predictions:
        score = float(item["score"])
        if score < threshold:
            continue
        start, end = int(item["start"]), int(item["end"])
        raw_label = (
            str(item.get("entity_group", item.get("entity", "")))
            .removeprefix("B-")
            .removeprefix("I-")
        )
        label = LABEL_MAP.get(raw_label)
        if label is None or not (0 <= start < end <= len(text)):
            continue
        entities.append(Entity(start, end, label, text[start:end], score))
    return sorted(entities)


class HuggingFaceNER:
    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        revision: str | None = DEFAULT_REVISION,
        threshold: float = 0.5,
    ) -> None:
        self.model_name = model
        self.revision = revision
        self.threshold = threshold
        self._pipeline: Any = None

    def _load(self) -> Any:
        if self._pipeline is None:
            from transformers import pipeline

            self._pipeline = pipeline(
                "token-classification",
                model=self.model_name,
                revision=self.revision,
                # Resolve WordPiece disagreements using the first token for each
                # visible word, then aggregate BIO labels into complete entities.
                aggregation_strategy="first",
            )
        return self._pipeline

    def predict(self, text: str) -> list[Entity]:
        return predictions_to_entities(text, self._load()(text), self.threshold)

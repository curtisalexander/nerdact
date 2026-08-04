"""GLiNER adapter for runtime-selected labels and stable character spans."""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Any

from .schema import LABELS, Entity

DEFAULT_GLINER_MODEL = "urchade/gliner_small-v2.1"
DEFAULT_GLINER_REVISION = "4e091416cf7c3481db542c2a3d26156916f3a47f"


@dataclass(frozen=True, slots=True)
class RuntimeLabel:
    type: str
    placeholder: str
    concise: str
    descriptive: str


@dataclass(frozen=True, slots=True)
class RuntimeLabelSchema:
    name: str
    labels: tuple[RuntimeLabel, ...]

    def prompts(self, wording: str) -> list[str]:
        if wording not in ("concise", "descriptive"):
            raise ValueError("wording must be 'concise' or 'descriptive'")
        return [getattr(label, wording) for label in self.labels]

    def prompt_map(self, wording: str) -> dict[str, str]:
        return {
            prompt.casefold(): label.type
            for prompt, label in zip(self.prompts(wording), self.labels, strict=True)
        }


def load_runtime_schema(path: str | Path) -> RuntimeLabelSchema:
    """Load and validate the checked-in runtime-label and placeholder contract."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    try:
        labels = tuple(RuntimeLabel(**item) for item in data["labels"])
        name = str(data["schema"])
    except (KeyError, TypeError) as error:
        raise ValueError("malformed runtime label schema") from error
    types = [label.type for label in labels]
    prompts = [
        prompt.casefold() for label in labels for prompt in (label.concise, label.descriptive)
    ]
    if set(types) != set(LABELS) or len(types) != len(LABELS):
        raise ValueError(f"runtime schema must define each canonical label exactly once: {LABELS}")
    if any(label.placeholder != label.type for label in labels):
        raise ValueError("placeholder mappings must preserve canonical typed placeholders")
    if any(not prompt.strip() for prompt in prompts) or len(prompts) != len(set(prompts)):
        raise ValueError("runtime label descriptions must be non-empty and unique")
    return RuntimeLabelSchema(name, labels)


def gliner_predictions_to_entities(
    text: str,
    predictions: Iterable[Mapping[str, Any]],
    label_map: Mapping[str, str],
) -> list[Entity]:
    """Validate GLiNER's native span output and normalize requested labels."""
    entities: dict[tuple[int, int, str], Entity] = {}
    for item in predictions:
        label = label_map.get(str(item["label"]).casefold())
        if label is None:
            continue
        start, end = item["start"], item["end"]
        score = float(item["score"])
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
        entity = Entity(start, end, label, text[start:end], score)
        previous = entities.get(entity.key())
        if previous is None or score > float(previous.score or 0):
            entities[entity.key()] = entity
    return sorted(entities.values())


class GLiNERAdapter:
    """Lazy adapter around GLiNER's label-conditioned span decoder."""

    def __init__(
        self,
        schema: RuntimeLabelSchema,
        model: str = DEFAULT_GLINER_MODEL,
        revision: str | None = DEFAULT_GLINER_REVISION,
        threshold: float = 0.5,
        wording: str = "concise",
        *,
        flat_ner: bool = True,
        multi_label: bool = False,
    ) -> None:
        if not 0 <= threshold <= 1:
            raise ValueError("threshold must be between 0 and 1")
        schema.prompts(wording)
        self.schema = schema
        self.model_name = model
        self.revision = revision
        self.threshold = threshold
        self.wording = wording
        self.flat_ner = flat_ner
        self.multi_label = multi_label
        self._model: Any = None

    def _load(self) -> Any:
        if self._model is None:
            GLiNER = import_module("gliner").GLiNER
            self._model = GLiNER.from_pretrained(self.model_name, revision=self.revision)
        return self._model

    def predict(self, text: str) -> list[Entity]:
        prompts = self.schema.prompts(self.wording)
        predictions = self._load().predict_entities(
            text,
            prompts,
            threshold=self.threshold,
            flat_ner=self.flat_ner,
            multi_label=self.multi_label,
        )
        return gliner_predictions_to_entities(
            text, predictions, self.schema.prompt_map(self.wording)
        )

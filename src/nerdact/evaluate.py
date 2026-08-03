"""Exact-span and character-coverage evaluation."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from typing import Any

from .schema import LABELS, Entity, Example


def _ratio(numerator: int | float, denominator: int | float) -> float:
    return numerator / denominator if denominator else 0.0


def _scores(tp: int, fp: int, fn: int) -> dict[str, float | int]:
    precision, recall = _ratio(tp, tp + fp), _ratio(tp, tp + fn)
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": _ratio(2 * precision * recall, precision + recall),
    }


def evaluate(
    examples: Sequence[Example], predictions: Sequence[Sequence[Entity]]
) -> dict[str, Any]:
    if len(examples) != len(predictions):
        raise ValueError("examples and predictions must have equal lengths")
    counts: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])
    errors: list[dict[str, Any]] = []
    gold_chars = predicted_chars = leaked_chars = extra_chars = total_chars = 0
    for example, predicted in zip(examples, predictions, strict=True):
        gold_keys = {entity.key(): entity for entity in example.entities}
        pred_keys = {entity.key(): entity for entity in predicted}
        for key in gold_keys.keys() & pred_keys.keys():
            counts[key[2]][0] += 1
        for key in pred_keys.keys() - gold_keys.keys():
            counts[key[2]][1] += 1
        for key in gold_keys.keys() - pred_keys.keys():
            counts[key[2]][2] += 1
        gold_mask = {i for entity in example.entities for i in range(entity.start, entity.end)}
        pred_mask = {i for entity in predicted for i in range(entity.start, entity.end)}
        gold_chars += len(gold_mask)
        predicted_chars += len(pred_mask)
        leaked_chars += len(gold_mask - pred_mask)
        extra_chars += len(pred_mask - gold_mask)
        total_chars += len(example.text)
        errors.append(
            {
                "id": example.id,
                "fp": [entity for key, entity in pred_keys.items() if key not in gold_keys],
                "fn": [entity for key, entity in gold_keys.items() if key not in pred_keys],
            }
        )
    per_label = {label: _scores(*counts[label]) for label in LABELS}
    totals = [sum(counts[label][index] for label in LABELS) for index in range(3)]
    macro = {
        metric: sum(float(per_label[label][metric]) for label in LABELS) / len(LABELS)
        for metric in ("precision", "recall", "f1")
    }
    return {
        "micro": _scores(*totals),
        "macro": macro,
        "per_label": per_label,
        "characters": {
            "gold_entity_characters": gold_chars,
            "predicted_entity_characters": predicted_chars,
            "non_gold_characters": total_chars - gold_chars,
            "leaked_gold_characters": leaked_chars,
            "over_redacted_characters": extra_chars,
            "leakage_rate": _ratio(leaked_chars, gold_chars),
            "over_redaction_rate": _ratio(extra_chars, total_chars - gold_chars),
        },
        "errors": errors,
    }

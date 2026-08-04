"""Exact-span and character-coverage evaluation."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Collection, Sequence
from typing import Any

from .schema import LABELS, Entity, Example, validate_entities


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
    examples: Sequence[Example],
    predictions: Sequence[Sequence[Entity]],
    labels: Collection[str] = LABELS,
) -> dict[str, Any]:
    if len(examples) != len(predictions):
        raise ValueError("examples and predictions must have equal lengths")
    label_order = tuple(labels)
    if not label_order or len(label_order) != len(set(label_order)):
        raise ValueError("labels must be non-empty and unique")
    counts: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])
    errors: list[dict[str, Any]] = []
    gold_chars = predicted_chars = leaked_chars = extra_chars = total_chars = 0
    transcripts_with_leaks = 0
    for example, predicted in zip(examples, predictions, strict=True):
        validate_entities(example.text, example.entities)
        validate_entities(example.text, predicted)
        gold_key_list = [entity.key() for entity in example.entities]
        pred_key_list = [entity.key() for entity in predicted]
        if len(gold_key_list) != len(set(gold_key_list)):
            raise ValueError(f"{example.id}: duplicate gold entities")
        if len(pred_key_list) != len(set(pred_key_list)):
            raise ValueError(f"{example.id}: duplicate predicted entities")
        observed_labels = {entity.label for entity in example.entities} | {
            entity.label for entity in predicted
        }
        unknown_labels = observed_labels - set(label_order)
        if unknown_labels:
            raise ValueError(
                f"entities contain labels outside the evaluation inventory: {unknown_labels}"
            )
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
        transcripts_with_leaks += bool(gold_mask - pred_mask)
        total_chars += len(example.text)
        errors.append(
            {
                "id": example.id,
                "fp": [entity for key, entity in pred_keys.items() if key not in gold_keys],
                "fn": [entity for key, entity in gold_keys.items() if key not in pred_keys],
            }
        )
    per_label = {label: _scores(*counts[label]) for label in label_order}
    totals = [sum(counts[label][index] for label in label_order) for index in range(3)]
    macro = {
        metric: sum(float(per_label[label][metric]) for label in label_order) / len(label_order)
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
        "transcripts": {
            "count": len(examples),
            "with_any_leak": transcripts_with_leaks,
            "any_leak_rate": _ratio(transcripts_with_leaks, len(examples)),
        },
        "errors": errors,
    }

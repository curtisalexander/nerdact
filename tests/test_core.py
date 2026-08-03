import json

import pytest

from nerdact.evaluate import evaluate
from nerdact.model import _aggregation_strategy, predictions_to_entities
from nerdact.redact import redact
from nerdact.report import build_results, render_html, write_comparison
from nerdact.schema import Entity, Example, load_jsonl


def test_fake_pipeline_conversion_threshold_and_labels():
    text = "Ada joined Acme in Rome"
    fake = [
        {"entity_group": "PER", "start": 0, "end": 3, "score": 0.99},
        {"entity_group": "ORG", "start": 11, "end": 15, "score": 0.8},
        {"entity_group": "LOC", "start": 19, "end": 23, "score": 0.2},
    ]
    assert predictions_to_entities(text, fake, 0.5) == [
        Entity(0, 3, "PERSON", "Ada", 0.99),
        Entity(11, 15, "ORGANIZATION", "Acme", 0.8),
    ]


def test_aggregation_matches_the_models_label_contract():
    assert _aggregation_strategy(["O", "B-PER", "I-PER"]) == "first"
    assert _aggregation_strategy(["O", "PER", "ORG"]) == "simple"


def test_redaction_reuses_values_and_replaces_right_to_left():
    text = "Ada met Ada in Rome."
    entities = [
        Entity(0, 3, "PERSON", "Ada"),
        Entity(8, 11, "PERSON", "Ada"),
        Entity(15, 19, "LOCATION", "Rome"),
    ]
    assert redact(text, entities)[0] == "[PERSON_1] met [PERSON_1] in [LOCATION_1]."


def test_exact_and_character_metrics():
    example = Example(
        "x", "Ada in Rome", (Entity(0, 3, "PERSON", "Ada"), Entity(7, 11, "LOCATION", "Rome"))
    )
    predicted = [[Entity(0, 3, "PERSON", "Ada", 0.9), Entity(7, 10, "LOCATION", "Rom", 0.8)]]
    metrics = evaluate([example], predicted)
    assert metrics["micro"] == {
        "tp": 1,
        "fp": 1,
        "fn": 1,
        "precision": 0.5,
        "recall": 0.5,
        "f1": 0.5,
    }
    assert metrics["characters"]["leaked_gold_characters"] == 1
    assert metrics["characters"]["over_redacted_characters"] == 0


def test_loader_validates_entity_text(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text(
        json.dumps(
            {
                "id": "x",
                "text": "Ada",
                "entities": [{"start": 0, "end": 3, "label": "PERSON", "text": "Eve"}],
            }
        )
        + "\n"
    )
    with pytest.raises(ValueError, match="does not match"):
        load_jsonl(path)


def test_report_escapes_all_content():
    text = "<script>alert(1)</script> Ada"
    example = Example("<bad>", text, (Entity(26, 29, "PERSON", "Ada"),), "<img src=x>")
    prediction = Entity(26, 29, "PERSON", "Ada", 0.9)
    page = render_html(build_results([example], [[prediction]], "<model>", "<rev>"))
    assert "<script>" not in page and "<img src=x>" not in page
    assert "&lt;script&gt;" in page and "&lt;model&gt;" in page


def test_comparison_supports_multiple_linked_model_profiles(tmp_path):
    example = Example("x", "Ada", (Entity(0, 3, "PERSON", "Ada"),))
    result = build_results([example], [[Entity(0, 3, "PERSON", "Ada", 0.9)]], "model", "rev")
    rows = [
        {
            "name": "Small <model>",
            "model": "owner/model",
            "revision": "abc",
            "parameters": "66M",
            "introduced": "2019",
            "report": "small.html",
            "domain": result,
            "benchmark": result,
            "performance": {
                "warm_latency_median_ms": 12.3,
                "examples_per_second": 80.0,
                "cached_snapshot_bytes": 1024**2,
                "peak_memory_bytes": 2 * 1024**2,
                "peak_memory_kind": "process RSS",
                "peak_rss_bytes": 2 * 1024**2,
                "timed_repeats": 3,
            },
            "environment": {
                "processor": "Test CPU",
                "machine": "test64",
                "os": "Test OS",
                "device": "cpu",
                "python": "3.11",
                "torch": "2.0",
                "transformers": "4.0",
            },
        },
        {
            "name": "Large model",
            "model": "owner/large",
            "revision": "def",
            "parameters": "340M",
            "introduced": "2018",
            "report": "large.html",
            "domain": result,
            "benchmark": result,
            "performance": {
                "warm_latency_median_ms": 45.6,
                "examples_per_second": 20.0,
                "cached_snapshot_bytes": 3 * 1024**2,
                "peak_memory_bytes": 4 * 1024**2,
                "peak_memory_kind": "process RSS",
                "peak_rss_bytes": 4 * 1024**2,
                "timed_repeats": 3,
            },
            "environment": {
                "processor": "Test CPU",
                "machine": "test64",
                "os": "Test OS",
                "device": "cpu",
                "python": "3.11",
                "torch": "2.0",
                "transformers": "4.0",
            },
        },
    ]
    output = tmp_path / "comparison.html"
    write_comparison(rows, output, "validation", 1)
    page = output.read_text()
    assert 'href="small.html"' in page and 'href="large.html"' in page
    assert "Small &lt;model&gt;" in page and "66M" in page
    assert "12.3 ms" in page and "1 MiB" in page and "Test CPU" in page
    assert "same 1 fictional call transcript" in page
    assert "Quality versus cost conclusion" in page


def test_checked_in_data_is_valid():
    examples = load_jsonl("data/transcripts.jsonl")
    assert len(examples) == 20
    assert any(not example.entities for example in examples)
    assert any(entity.start == 0 for example in examples for entity in example.entities)
    assert any(
        entity.end == len(example.text) for example in examples for entity in example.entities
    )

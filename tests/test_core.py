import json

import pytest

from nerdact.evaluate import evaluate
from nerdact.model import predictions_to_entities
from nerdact.redact import redact
from nerdact.report import build_results, render_html
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


def test_checked_in_data_is_valid():
    examples = load_jsonl("data/transcripts.jsonl")
    assert len(examples) == 10
    assert any(not example.entities for example in examples)

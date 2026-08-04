import json
from pathlib import Path

import pytest

from nerdact.cli import (
    _long_context_examples,
    _model_profiles,
    _pii_system_results,
    _select_recall_threshold,
    _validate_pii_splits,
    load_benchmark_manifest,
)
from nerdact.detectors import detect_structured_pii
from nerdact.evaluate import evaluate
from nerdact.gliner_model import (
    GLiNERAdapter,
    gliner_predictions_to_entities,
    load_runtime_schema,
)
from nerdact.hybrid import resolve_pii_overlaps
from nerdact.model import (
    HuggingFaceNER,
    _aggregation_strategy,
    _validate_labels,
    predictions_to_entities,
)
from nerdact.pii_model import (
    GLiNER2PIIAdapter,
    gliner2_predictions_to_entities,
    load_pii_schema,
)
from nerdact.provenance import build_provenance
from nerdact.redact import redact
from nerdact.report import (
    build_results,
    render_html,
    write_comparison,
    write_landing_page,
    write_learning_summary,
    write_pii_comparison,
)
from nerdact.schema import PII_LABELS, Entity, Example, load_jsonl


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


def test_prediction_offsets_exclude_tokenizer_separator_whitespace():
    text = "Ada met Mira Sol today"
    fake = [{"entity_group": "PER", "start": 7, "end": 16, "score": 0.9}]
    assert predictions_to_entities(text, fake) == [Entity(8, 16, "PERSON", "Mira Sol", 0.9)]


def test_hugging_face_conversion_rejects_bad_offsets_and_deduplicates_windows():
    text = "Ada joined Acme"
    duplicate_predictions = [
        {"entity_group": "PER", "start": 0, "end": 3, "score": 0.8},
        {"entity_group": "PER", "start": 0, "end": 3, "score": 0.9},
    ]
    assert predictions_to_entities(text, duplicate_predictions) == [
        Entity(0, 3, "PERSON", "Ada", 0.9)
    ]
    with pytest.raises(ValueError, match="offsets"):
        predictions_to_entities(
            text, [{"entity_group": "PER", "start": -1, "end": 3, "score": 0.9}]
        )
    with pytest.raises(ValueError, match="finite"):
        predictions_to_entities(
            text, [{"entity_group": "PER", "start": 0, "end": 3, "score": float("nan")}]
        )


def test_fixed_adapter_rejects_invalid_thresholds():
    for threshold in (-0.1, 1.1, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="threshold"):
            HuggingFaceNER(threshold=threshold)


def test_aggregation_matches_the_models_label_contract():
    assert _aggregation_strategy(["O", "B-PER", "I-PER"]) == "first"
    assert _aggregation_strategy(["O", "PER", "ORG"]) == "simple"


def test_checkpoint_labels_must_match_all_four_labels_exactly():
    _validate_labels(
        ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "B-MISC", "I-MISC"]
    )
    with pytest.raises(ValueError, match="map exactly"):
        _validate_labels(["O", "B-PER", "I-PER", "B-DATE", "I-DATE"])
    with pytest.raises(ValueError, match="map exactly"):
        _validate_labels(["O", "PER", "ORG", "LOC"])


def test_adapter_uses_overlap_to_recover_a_512_token_boundary_span():
    example = _long_context_examples()[0]
    entity = example.entities[0]
    assert len(example.text[: entity.start].split()) == 509

    class BoundaryPipeline:
        def __call__(self, text, *, stride):
            assert text == example.text
            assert stride == 64
            return [
                {
                    "entity_group": "PER",
                    "start": entity.start,
                    "end": entity.end,
                    "score": 0.99,
                }
            ]

    adapter = HuggingFaceNER()
    adapter._pipeline = BoundaryPipeline()
    assert adapter.predict(example.text) == [
        Entity(entity.start, entity.end, "PERSON", "Mira Sol", 0.99)
    ]


def test_checked_in_runtime_schema_has_described_typed_mappings():
    schema = load_runtime_schema("data/runtime-labels.json")
    assert schema.prompts("concise") == [
        "person",
        "organization",
        "location",
        "miscellaneous named entity",
    ]
    assert schema.prompt_map("descriptive")["named individual person"] == "PERSON"
    assert [label.placeholder for label in schema.labels] == [label.type for label in schema.labels]


def test_benchmark_manifest_is_the_profile_and_dataset_source_of_truth():
    manifest = load_benchmark_manifest()
    classic = _model_profiles(manifest, "classic")
    modern = _model_profiles(manifest, "modern")

    assert manifest["datasets"]["conll2003"] == {
        "id": "BramVanroy/conll2003",
        "revision": "4ffbd53d9e0b92b473b9b7dcff12f53e7c17ce0c",
        "allowed_splits": ["validation", "test"],
    }
    assert [profile.model for profile in classic] == [
        "dslim/distilbert-NER",
        "dslim/bert-base-NER",
        "dslim/bert-large-NER",
        "Jean-Baptiste/roberta-large-ner-english",
    ]
    assert [profile.report for profile in classic] == [
        "distilbert.html",
        "baseline.html",
        "bert-large.html",
        "roberta-large.html",
    ]
    assert modern[0].decoder.endswith("64-token overlap")
    assert modern[1].context_length == 8192


def test_benchmark_manifest_rejects_adapter_revision_drift(tmp_path):
    manifest = json.loads(Path("data/benchmark-manifest.json").read_text())
    manifest["checkpoints"]["bert-base"]["revision"] = "mutable-main"
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="does not match its adapter default"):
        load_benchmark_manifest(path)


def test_gliner_adapter_uses_native_span_decoder_contract():
    schema = load_runtime_schema("data/runtime-labels.json")

    class FakeGLiNER:
        def predict_entities(self, text, labels, *, threshold, flat_ner, multi_label):
            assert text == "Mira joined Northstar"
            assert labels == schema.prompts("descriptive")
            assert threshold == 0.4
            assert flat_ner is False
            assert multi_label is True
            return [
                {"start": 0, "end": 4, "text": "Mira", "label": labels[0], "score": 0.91},
                {
                    "start": 12,
                    "end": 21,
                    "text": "Northstar",
                    "label": labels[1],
                    "score": 0.82,
                },
            ]

    adapter = GLiNERAdapter(
        schema, threshold=0.4, wording="descriptive", flat_ner=False, multi_label=True
    )
    adapter._model = FakeGLiNER()
    assert adapter.predict("Mira joined Northstar") == [
        Entity(0, 4, "PERSON", "Mira", 0.91),
        Entity(12, 21, "ORGANIZATION", "Northstar", 0.82),
    ]


def test_gliner_conversion_preserves_competing_and_overlapping_spans():
    text = "Northstar Relay"
    predictions = [
        {"start": 0, "end": 15, "label": "organization", "score": 0.8},
        {"start": 0, "end": 9, "label": "location", "score": 0.7},
        {"start": 0, "end": 15, "label": "unknown", "score": 0.9},
    ]
    assert gliner_predictions_to_entities(
        text, predictions, {"organization": "ORGANIZATION", "location": "LOCATION"}
    ) == [
        Entity(0, 9, "LOCATION", "Northstar", 0.7),
        Entity(0, 15, "ORGANIZATION", "Northstar Relay", 0.8),
    ]


def test_checked_in_pii_schema_maps_only_documented_model_labels():
    schema = load_pii_schema("data/pii-labels.json")
    assert set(schema.model_label_map().values()) == set(PII_LABELS) - {"DEVICE_ID", "URL"}
    assert schema.model_label_map()["email"] == "EMAIL_ADDRESS"
    assert schema.model_label_map()["card_number"] == "FINANCIAL_ID"
    assert schema.requested_labels().count("phone_number") == 1


def test_gliner2_pii_adapter_uses_long_span_extraction_contract():
    schema = load_pii_schema("data/pii-labels.json")

    class FakeSchemaBuilder:
        def entities(self, labels):
            assert labels == schema.requested_labels()
            return "built-schema"

    class FakeGLiNER2:
        def create_schema(self):
            return FakeSchemaBuilder()

        def extract_long(self, text, extraction_schema, **options):
            assert text == "Email mira@example.test from 192.0.2.1"
            assert extraction_schema == "built-schema"
            assert options == {
                "threshold": 0.4,
                "chunk_size": 384,
                "chunk_overlap": 64,
                "include_confidence": True,
                "include_spans": True,
                "overlap_policy": None,
            }
            return {
                "entities": {
                    "email": [
                        {
                            "text": "mira@example.test",
                            "confidence": 0.91,
                            "start": 6,
                            "end": 23,
                        }
                    ],
                    "ip_address": [
                        {"text": "192.0.2.1", "confidence": 0.82, "start": 29, "end": 38}
                    ],
                }
            }

    adapter = GLiNER2PIIAdapter(schema, threshold=0.4)
    adapter._model = FakeGLiNER2()
    assert adapter.predict("Email mira@example.test from 192.0.2.1") == [
        Entity(
            6,
            23,
            "EMAIL_ADDRESS",
            "mira@example.test",
            0.91,
            ("model:fastino/gliner2-privacy-filter-PII-multi",),
        ),
        Entity(
            29,
            38,
            "IP_ADDRESS",
            "192.0.2.1",
            0.82,
            ("model:fastino/gliner2-privacy-filter-PII-multi",),
        ),
    ]


def test_gliner2_pii_conversion_deduplicates_broad_and_specific_labels():
    text = "Card 4111 1111 1111 1111"
    result = {
        "entities": {
            "payment_card": [{"text": text[5:], "confidence": 0.8, "start": 5, "end": len(text)}],
            "card_number": [{"text": text[5:], "confidence": 0.9, "start": 5, "end": len(text)}],
            "unsupported": [{"text": "Card", "confidence": 1.0, "start": 0, "end": 4}],
        }
    }
    label_map = {"payment_card": "FINANCIAL_ID", "card_number": "FINANCIAL_ID"}
    assert gliner2_predictions_to_entities(text, result, label_map) == [
        Entity(5, len(text), "FINANCIAL_ID", text[5:], 0.9)
    ]


def test_structured_detectors_preserve_rule_provenance_and_exact_spans():
    text = (
        "Email mira@example.test, call +1 (202) 555-0147, then open "
        "https://example.test/reset/demo. Login came from 192.0.2.44."
    )
    entities = detect_structured_pii(text)
    assert [(entity.label, entity.text, entity.provenance) for entity in entities] == [
        ("EMAIL_ADDRESS", "mira@example.test", ("detector:email",)),
        ("PHONE_NUMBER", "+1 (202) 555-0147", ("detector:phone",)),
        ("URL", "https://example.test/reset/demo", ("detector:url",)),
        ("IP_ADDRESS", "192.0.2.44", ("detector:ip-address",)),
    ]


def test_url_detector_preserves_balanced_delimiters_and_trims_prose_delimiters():
    text = (
        "See https://example.test/a(b), https://example.test/x[y], and "
        "(https://example.test/plain)."
    )
    assert [entity.text for entity in detect_structured_pii(text)] == [
        "https://example.test/a(b)",
        "https://example.test/x[y]",
        "https://example.test/plain",
    ]


def test_rules_only_pii_results_resolve_nested_structured_findings():
    text = "Open https://192.0.2.44/reset"
    example = Example("nested-rules", text, (Entity(5, len(text), "URL", text[5:]),))
    systems = _pii_system_results([example], [[]], "model", "revision")

    assert systems["rules"]["examples"][0]["predictions"] == [
        {
            "start": 5,
            "end": len(text),
            "label": "URL",
            "text": text[5:],
            "score": 1.0,
            "provenance": ("detector:url",),
        }
    ]


def test_checked_in_pii_rules_cover_supported_structures_without_negative_hits():
    examples = load_jsonl("data/pii-calibration.jsonl", PII_LABELS) + load_jsonl(
        "data/pii-evaluation.jsonl", PII_LABELS
    )
    predictions = {example.id: detect_structured_pii(example.text) for example in examples}
    for example in examples:
        assert all(entity.provenance for entity in predictions[example.id])
        if not example.entities:
            assert predictions[example.id] == []

    detected = {
        (example.id, entity.start, entity.end, entity.label)
        for example in examples
        for entity in predictions[example.id]
    }
    gold = {
        (example.id, entity.start, entity.end, entity.label)
        for example in examples
        for entity in example.entities
    }
    assert detected <= gold
    assert {label for _, _, _, label in detected} == {
        "CREDENTIAL",
        "DEVICE_ID",
        "EMAIL_ADDRESS",
        "FINANCIAL_ID",
        "GOVERNMENT_ID",
        "IP_ADDRESS",
        "PHONE_NUMBER",
        "URL",
    }


def test_hybrid_resolution_merges_sources_and_prefers_rules_over_model_conflicts():
    text = "Open https://192.0.2.44/reset"
    findings = [
        Entity(5, 31, "URL", text[5:31], 0.96, ("model:pii",)),
        Entity(5, 31, "URL", text[5:31], 1.0, ("detector:url",)),
        Entity(13, 23, "IP_ADDRESS", "192.0.2.44", 1.0, ("detector:ip-address",)),
        Entity(5, 23, "ADDRESS", text[5:23], 0.99, ("model:pii",)),
    ]
    result = resolve_pii_overlaps(findings)
    assert result.entities == (
        Entity(5, 31, "URL", text[5:31], 1.0, ("detector:url", "model:pii")),
    )
    assert {entity.label for entity in result.rejected} == {"ADDRESS", "IP_ADDRESS"}


def test_hybrid_resolution_uses_confidence_for_model_only_conflicts_and_keeps_adjacency():
    findings = [
        Entity(0, 4, "PERSON", "Mira", 0.8, ("model:pii",)),
        Entity(0, 9, "USERNAME", "Mira Vale", 0.7, ("model:pii",)),
        Entity(4, 9, "PERSON", " Vale", 0.6, ("model:pii",)),
    ]
    result = resolve_pii_overlaps(findings)
    assert result.entities == (
        Entity(0, 4, "PERSON", "Mira", 0.8, ("model:pii",)),
        Entity(4, 9, "PERSON", " Vale", 0.6, ("model:pii",)),
    )
    assert result.rejected == (Entity(0, 9, "USERNAME", "Mira Vale", 0.7, ("model:pii",)),)


def test_redaction_reuses_values_and_replaces_right_to_left():
    text = "Ada met Ada in Rome."
    entities = [
        Entity(0, 3, "PERSON", "Ada"),
        Entity(8, 11, "PERSON", "Ada"),
        Entity(15, 19, "LOCATION", "Rome"),
    ]
    assert redact(text, entities)[0] == "[PERSON_1] met [PERSON_1] in [LOCATION_1]."


def test_redaction_rejects_stale_or_out_of_range_entities():
    with pytest.raises(ValueError, match="does not match"):
        redact("Ada", [Entity(0, 3, "PERSON", "Eve")])
    with pytest.raises(ValueError, match="invalid entity span"):
        redact("Ada", [Entity(-1, 3, "PERSON", "Ada")])


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
    assert metrics["transcripts"] == {"count": 1, "with_any_leak": 1, "any_leak_rate": 1.0}


def test_evaluation_rejects_duplicate_predictions():
    example = Example("x", "Ada", (Entity(0, 3, "PERSON", "Ada"),))
    duplicate = Entity(0, 3, "PERSON", "Ada", 0.9)
    with pytest.raises(ValueError, match="duplicate predicted"):
        evaluate([example], [[duplicate, duplicate]])


def test_evaluation_uses_an_explicit_pii_label_inventory():
    example = Example(
        "pii",
        "mira@example.test",
        (Entity(0, 17, "EMAIL_ADDRESS", "mira@example.test"),),
    )
    metrics = evaluate([example], [[]], PII_LABELS)
    assert metrics["micro"]["fn"] == 1
    assert metrics["per_label"]["EMAIL_ADDRESS"]["fn"] == 1
    assert set(metrics["per_label"]) == set(PII_LABELS)

    with pytest.raises(ValueError, match="outside the evaluation inventory"):
        evaluate([example], [[]])


def test_recall_threshold_selection_uses_only_declared_safety_tiebreakers():
    def run(threshold, recall, leakage, any_leak, over_redaction):
        return {
            "threshold": threshold,
            "systems": {
                "hybrid": {
                    "metrics": {
                        "micro": {"recall": recall},
                        "characters": {
                            "leakage_rate": leakage,
                            "over_redaction_rate": over_redaction,
                        },
                        "transcripts": {"any_leak_rate": any_leak},
                    }
                }
            },
        }

    runs = [
        run(0.1, 1.0, 0.02, 0.25, 0.08),
        run(0.2, 1.0, 0.01, 0.25, 0.09),
        run(0.3, 1.0, 0.01, 0.25, 0.04),
        run(0.4, 0.9, 0.0, 0.0, 0.0),
    ]
    assert _select_recall_threshold(runs) == 0.3


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


@pytest.mark.parametrize(
    "record",
    [
        {"id": None, "text": "Ada"},
        {"id": "x", "text": ["Ada"]},
        {
            "id": "x",
            "text": "Ada",
            "entities": [{"start": 0.0, "end": 3, "label": "PERSON"}],
        },
        {
            "id": "x",
            "text": "Ada",
            "entities": [{"start": False, "end": 3, "label": "PERSON"}],
        },
    ],
)
def test_loader_rejects_coercible_non_json_schema_types(tmp_path, record):
    path = tmp_path / "bad-types.jsonl"
    path.write_text(json.dumps(record) + "\n")
    with pytest.raises(ValueError, match="invalid example|malformed entity"):
        load_jsonl(path)


def test_pii_splits_must_be_independent(tmp_path):
    calibration_path = tmp_path / "calibration.jsonl"
    evaluation_path = tmp_path / "evaluation.jsonl"
    calibration_path.write_text("")
    evaluation_path.write_text("")
    calibration = [Example("cal", "Mira called", ())]

    with pytest.raises(ValueError, match="different files"):
        _validate_pii_splits(calibration_path, calibration_path, calibration, [])
    with pytest.raises(ValueError, match="IDs overlap"):
        _validate_pii_splits(
            calibration_path, evaluation_path, calibration, [Example("cal", "Other text", ())]
        )
    with pytest.raises(ValueError, match="duplicate normalized text"):
        _validate_pii_splits(
            calibration_path,
            evaluation_path,
            calibration,
            [Example("eval", "  MIRA   called ", ())],
        )


def test_report_escapes_all_content():
    text = "<script>alert(1)</script> Ada"
    example = Example("<bad>", text, (Entity(26, 29, "PERSON", "Ada"),), "<img src=x>")
    prediction = Entity(26, 29, "PERSON", "Ada", 0.9)
    page = render_html(build_results([example], [[prediction]], "<model>", "<rev>"))
    assert "<script>" not in page and "<img src=x>" not in page
    assert "&lt;script&gt;" in page and "&lt;model&gt;" in page
    assert "How token predictions become evaluated redactions" in page
    assert all(
        f'id="{section}"' in page
        for section in ("ner", "spans", "evaluation", "results", "redaction", "limits", "reproduce")
    )
    assert 'class="case"' in page and "Recreate this report" in page
    assert "Results on this entire page" in page


def test_report_supports_scoreless_predictions_and_escapes_reproduction_command():
    example = Example("x", "Ada", (Entity(0, 3, "PERSON", "Ada"),))
    results = build_results([example], [[Entity(0, 3, "PERSON", "Ada")]], "model", "revision")
    results["reproduce_command"] = 'nerdact report --model "<model>"'

    page = render_html(results)

    assert "score unavailable" not in page
    assert "nerdact report --model &quot;&lt;model&gt;&quot;" in page


def test_provenance_fingerprints_inputs_options_and_surfaces_run_id(tmp_path):
    input_path = tmp_path / "input.txt"
    input_path.write_text("fictional input")
    root = Path(__file__).resolve().parents[1]
    first = build_provenance(root, [input_path], {"threshold": 0.5})
    second = build_provenance(root, [input_path], {"threshold": 0.5})

    assert first["run_id"] == second["run_id"]
    assert first["inputs_sha256"][str(input_path)]
    assert first["source_sha256"]
    assert first["dependencies"]["nerdact"] == "0.1.0"

    example = Example("x", "Ada", (Entity(0, 3, "PERSON", "Ada"),))
    results = build_results([example], [[]], "model", "revision")
    results["provenance"] = first
    page = render_html(results)
    assert first["run_id"] in page and "fingerprints are recorded" in page


def test_pii_report_escapes_originals_and_metadata(tmp_path):
    text = "<script>alert(1)</script> Ada"
    example = Example("<bad>", text, (Entity(26, 29, "PERSON", "Ada"),), "<img src=x>")
    result = build_results([example], [[]], "model", "revision", PII_LABELS)
    systems = {
        "model": result,
        "rules": result,
        "hybrid": result,
        "rejected_conflicts": [{"id": "<bad>", "entities": []}],
    }
    artifact = {
        "model": "<model>",
        "revision": "<revision>",
        "selection_policy": "test",
        "operating_threshold": 0.5,
        "calibration_runs": [{"threshold": 0.5, "systems": systems}],
        "evaluation": systems,
    }
    output = tmp_path / "pii.html"
    write_pii_comparison(artifact, output)
    page = output.read_text()
    assert "<script>" not in page and "<img src=x>" not in page
    assert "&lt;script&gt;" in page and "&lt;model&gt;" in page
    assert "What makes this system “hybrid”?" in page


def test_learning_summary_connects_results_to_examples(tmp_path):
    benchmark = {
        "models": [
            {
                "report": "roberta-large.html",
                "model": "owner/roberta",
                "domain": {
                    "metrics": {
                        "micro": {"f1": 0.88, "precision": 0.87},
                        "characters": {
                            "leakage_rate": 0.02,
                            "leaked_gold_characters": 9,
                            "gold_entity_characters": 475,
                        },
                    },
                    "examples": [
                        {"id": "lowercase", "text": "ada works at acme"},
                        {"id": "boundary-slashes", "text": "Ada/at Acme"},
                    ],
                },
            }
        ]
    }
    model_metrics = {
        "micro": {"f1": 0.78},
        "characters": {"leakage_rate": 0.27},
    }
    hybrid_metrics = {
        "micro": {"f1": 0.92, "recall": 0.88},
        "characters": {"leakage_rate": 0.15},
    }
    pii = {
        "model": "owner/gliner2",
        "evaluation": {
            "model": {"metrics": model_metrics},
            "hybrid": {
                "metrics": hybrid_metrics,
                "examples": [
                    {"id": "pii-eval-contact", "text": "Ada, ada@example.test"},
                    {"id": "pii-eval-unicode-address", "text": "Ada at 8 Rue Exemple"},
                    {"id": "pii-eval-password-spaces", "text": "correct horse demo"},
                ],
            },
        },
    }
    output = tmp_path / "conclusion.html"

    write_learning_summary(benchmark, pii, output)

    page = output.read_text()
    assert "RoBERTa NER: strong, narrow, and still fallible" in page
    assert "The GLiNER2 model is only one half" in page
    assert "examples/roberta_ner" in page and "examples/gliner2_pii_hybrid" in page
    assert "Do not compare the two headline F1 scores as a race" in page


def test_landing_page_offers_learning_answer_and_code_paths(tmp_path):
    output = tmp_path / "index.html"

    write_landing_page(output)

    page = output.read_text()
    assert 'href="baseline.html"' in page
    assert 'href="conclusion.html"' in page
    assert "Open examples" in page
    assert 'aria-current="page">Home</a>' in page


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
    assert "same 1 fictional transcript" in page
    assert "Quality versus cost conclusion" in page


def test_checked_in_data_is_valid():
    examples = load_jsonl("data/transcripts.jsonl")
    assert len(examples) == 20
    assert any(not example.entities for example in examples)
    assert any(entity.start == 0 for example in examples for entity in example.entities)
    assert any(
        entity.end == len(example.text) for example in examples for entity in example.entities
    )


def test_checked_in_pii_corpus_is_valid_split_and_covers_every_label():
    calibration = load_jsonl("data/pii-calibration.jsonl", PII_LABELS)
    evaluation = load_jsonl("data/pii-evaluation.jsonl", PII_LABELS)

    assert len(calibration) == 8
    assert len(evaluation) == 16
    assert {example.id for example in calibration}.isdisjoint(example.id for example in evaluation)
    assert any(not example.entities for example in calibration)
    assert any(not example.entities for example in evaluation)
    for examples in (calibration, evaluation):
        assert {entity.label for example in examples for entity in example.entities} == set(
            PII_LABELS
        )

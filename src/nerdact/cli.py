"""Command-line interface."""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing
import re
import shlex
import unicodedata
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .detectors import detect_structured_pii
from .gliner_model import (
    DEFAULT_GLINER_MODEL,
    DEFAULT_GLINER_REVISION,
    GLiNERAdapter,
    load_runtime_schema,
)
from .hybrid import resolve_pii_overlaps
from .model import DEFAULT_MODEL, DEFAULT_REVISION, HuggingFaceNER
from .performance import measure_warm_inference
from .pii_model import (
    DEFAULT_PII_MODEL,
    DEFAULT_PII_REVISION,
    GLiNER2PIIAdapter,
    load_pii_schema,
)
from .provenance import build_provenance
from .redact import redact
from .report import (
    build_results,
    write_comparison,
    write_learning_summary,
    write_pii_comparison,
    write_results,
    write_runtime_comparison,
)
from .schema import LABELS, PII_LABELS, Entity, Example, load_jsonl

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA = ROOT / "data" / "transcripts.jsonl"
DEFAULT_RUNTIME_SCHEMA = ROOT / "data" / "runtime-labels.json"
DEFAULT_PII_CALIBRATION_DATA = ROOT / "data" / "pii-calibration.jsonl"
DEFAULT_PII_EVALUATION_DATA = ROOT / "data" / "pii-evaluation.jsonl"
DEFAULT_PII_SCHEMA = ROOT / "data" / "pii-labels.json"
BENCHMARK_MANIFEST_REFERENCE = Path("data/benchmark-manifest.json")
DEFAULT_BENCHMARK_MANIFEST = ROOT / BENCHMARK_MANIFEST_REFERENCE


@dataclass(frozen=True)
class ModelProfile:
    name: str
    model: str
    revision: str
    parameters: str
    introduced: str
    report: str
    license: str
    training_data: str
    label_schema: tuple[str, ...]
    context_length: int
    decoder: str


def load_benchmark_manifest(path: str | Path = DEFAULT_BENCHMARK_MANIFEST) -> dict[str, Any]:
    """Load the benchmark metadata contract and reject incomplete or drifting records."""
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        datasets = data["datasets"]
        checkpoints = data["checkpoints"]
        experiments = data["experiments"]
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as error:
        raise ValueError(f"malformed benchmark manifest: {error}") from error
    if data.get("schema_version") != 1:
        raise ValueError("benchmark manifest schema_version must be 1")
    if (
        not isinstance(datasets, dict)
        or not isinstance(checkpoints, dict)
        or not isinstance(experiments, dict)
    ):
        raise ValueError(
            "benchmark manifest datasets, checkpoints, and experiments must be objects"
        )

    checkpoint_fields = {
        "model": str,
        "revision": str,
        "license": str,
        "parameters": str,
        "introduced": str,
        "training_data": str,
        "label_schema": list,
        "context_length": int,
        "context_unit": str,
        "decoder": str,
    }
    identities: set[tuple[str, str]] = set()
    for name, checkpoint in checkpoints.items():
        if not isinstance(name, str) or not isinstance(checkpoint, dict):
            raise ValueError("benchmark checkpoint names and records must be objects")
        for field, expected_type in checkpoint_fields.items():
            if not isinstance(checkpoint.get(field), expected_type) or not checkpoint[field]:
                raise ValueError(f"checkpoint {name!r} has invalid {field!r}")
        if not all(isinstance(label, str) and label for label in checkpoint["label_schema"]):
            raise ValueError(f"checkpoint {name!r} has invalid label_schema")
        identity = (checkpoint["model"], checkpoint["revision"])
        if identity in identities:
            raise ValueError(f"duplicate checkpoint identity {identity!r}")
        identities.add(identity)

    try:
        conll = datasets["conll2003"]
        if not all(isinstance(conll[field], str) and conll[field] for field in ("id", "revision")):
            raise ValueError("CoNLL id and revision must be non-empty strings")
        if not isinstance(conll["allowed_splits"], list) or not conll["allowed_splits"]:
            raise ValueError("CoNLL allowed_splits must be a non-empty list")
        if not all(isinstance(split, str) and split for split in conll["allowed_splits"]):
            raise ValueError("CoNLL allowed_splits must contain non-empty strings")
        for experiment_name in ("classic", "modern"):
            experiment = experiments[experiment_name]
            if experiment["dataset"] not in datasets:
                raise ValueError(f"experiment {experiment_name!r} references an unknown dataset")
            if experiment["split"] not in datasets[experiment["dataset"]]["allowed_splits"]:
                raise ValueError(f"experiment {experiment_name!r} uses an unsupported split")
            if not isinstance(experiment["profiles"], list) or not experiment["profiles"]:
                raise ValueError(f"experiment {experiment_name!r} must define profiles")
            reports: set[str] = set()
            for profile in experiment["profiles"]:
                if profile["checkpoint"] not in checkpoints:
                    raise ValueError(f"experiment {experiment_name!r} references a checkpoint")
                if not all(
                    isinstance(profile[field], str) and profile[field]
                    for field in ("name", "report")
                ):
                    raise ValueError(f"experiment {experiment_name!r} has an invalid profile")
                if profile["report"] in reports:
                    raise ValueError(f"experiment {experiment_name!r} repeats a report path")
                reports.add(profile["report"])
                labels = checkpoints[profile["checkpoint"]]["label_schema"]
                if labels != list(LABELS):
                    raise ValueError(
                        f"experiment {experiment_name!r} requires the fixed label schema"
                    )
                if checkpoints[profile["checkpoint"]]["context_unit"] != "tokens":
                    raise ValueError(f"experiment {experiment_name!r} requires token context")
                if "decoder" in profile and not isinstance(profile["decoder"], str):
                    raise ValueError(f"experiment {experiment_name!r} has an invalid decoder")
        for experiment_name in ("runtime-labels", "practical-pii"):
            if experiments[experiment_name]["checkpoint"] not in checkpoints:
                raise ValueError(f"experiment {experiment_name!r} references a checkpoint")
    except (KeyError, TypeError) as error:
        raise ValueError(f"malformed benchmark experiment: {error}") from error

    defaults = {
        "bert-base": (DEFAULT_MODEL, DEFAULT_REVISION),
        "gliner-small": (DEFAULT_GLINER_MODEL, DEFAULT_GLINER_REVISION),
        "gliner2-pii": (DEFAULT_PII_MODEL, DEFAULT_PII_REVISION),
    }
    for name, expected in defaults.items():
        checkpoint = checkpoints.get(name)
        if not isinstance(checkpoint, dict):
            raise ValueError(f"manifest must define adapter checkpoint {name!r}")
        actual = (checkpoint["model"], checkpoint["revision"])
        if actual != expected:
            raise ValueError(f"manifest checkpoint {name!r} does not match its adapter default")
    if not isinstance(checkpoints["gliner-small"].get("maximum_span_width"), int):
        raise ValueError("manifest GLiNER checkpoint must define maximum_span_width")
    return data


def _model_profiles(manifest: dict[str, Any], experiment_name: str) -> tuple[ModelProfile, ...]:
    checkpoints = manifest["checkpoints"]
    profiles = []
    for record in manifest["experiments"][experiment_name]["profiles"]:
        checkpoint = checkpoints[record["checkpoint"]]
        profiles.append(
            ModelProfile(
                name=record["name"],
                model=checkpoint["model"],
                revision=checkpoint["revision"],
                parameters=checkpoint["parameters"],
                introduced=checkpoint["introduced"],
                report=record["report"],
                license=checkpoint["license"],
                training_data=checkpoint["training_data"],
                label_schema=tuple(checkpoint["label_schema"]),
                context_length=checkpoint["context_length"],
                decoder=record.get("decoder", checkpoint["decoder"]),
            )
        )
    return tuple(profiles)


def _compare_model(
    profile: ModelProfile,
    domain_examples: list[Example],
    benchmark_examples: list[Example],
    threshold: float,
    repeats: int,
) -> dict[str, Any]:
    """Evaluate one model in an isolated process so its peak RSS is meaningful."""
    adapter = HuggingFaceNER(profile.model, profile.revision, threshold)
    measured = measure_warm_inference(
        adapter, [example.text for example in domain_examples], repeats
    )
    domain = build_results(
        domain_examples,
        measured.pop("predictions"),
        profile.model,
        profile.revision,
    )
    benchmark = build_results(
        benchmark_examples,
        [adapter.predict(example.text) for example in benchmark_examples],
        profile.model,
        profile.revision,
    )
    return {
        "name": profile.name,
        "model": profile.model,
        "revision": profile.revision,
        "parameters": profile.parameters,
        "introduced": profile.introduced,
        "report": profile.report,
        "license": profile.license,
        "training_data": profile.training_data,
        "label_schema": list(profile.label_schema),
        "context_length": profile.context_length,
        "decoder": profile.decoder,
        "domain": domain,
        "benchmark": benchmark,
        **measured,
    }


def _model_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Hugging Face model ID or path")
    parser.add_argument("--revision", default=None, help="model revision (default model is pinned)")
    parser.add_argument("--threshold", type=float, default=0.5)


def _revision(args: argparse.Namespace) -> str | None:
    return (
        args.revision
        if args.revision is not None
        else (DEFAULT_REVISION if args.model == DEFAULT_MODEL else None)
    )


def _cli_path(path: str | Path) -> str:
    """Use portable repository-relative paths in generated reproduction commands."""
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(ROOT).as_posix()
    except ValueError:
        return str(resolved)


def _run_corpus(args: argparse.Namespace, make_report: bool) -> None:
    examples = load_jsonl(args.data)
    revision = _revision(args)
    adapter = HuggingFaceNER(args.model, revision, args.threshold)
    predictions = [adapter.predict(example.text) for example in examples]
    results = build_results(examples, predictions, args.model, revision)
    results["provenance"] = build_provenance(
        ROOT,
        [Path(args.data), DEFAULT_BENCHMARK_MANIFEST],
        {
            "command": "report" if make_report else "demo",
            "model": args.model,
            "revision": revision,
            "threshold": args.threshold,
        },
    )
    if make_report:
        command = [
            "uv",
            "run",
            "nerdact",
            "report",
            "--data",
            _cli_path(args.data),
            "--model",
            args.model,
            "--threshold",
            str(args.threshold),
            "--output",
            _cli_path(args.output),
            "--html",
            _cli_path(args.html),
        ]
        if revision is not None:
            command.extend(("--revision", revision))
        results["reproduce_command"] = shlex.join(command)
        write_results(results, Path(args.output), Path(args.html))
        print(f"Wrote {args.output} and {args.html}")
    else:
        for record in results["examples"]:
            print(f"\n[{record['id']}]\n{record['redacted']}")
        print("\nExact micro metrics:", json.dumps(results["metrics"]["micro"], indent=2))


def _conll_examples(
    limit: int, split: str, dataset_id: str, dataset_revision: str
) -> list[Example]:
    from importlib import import_module

    load_dataset = import_module("datasets").load_dataset
    dataset = load_dataset(dataset_id, revision=dataset_revision, split=f"{split}[:{limit}]")
    names = dataset.features["ner_tags"].feature.names
    label_map = {
        "PER": "PERSON",
        "ORG": "ORGANIZATION",
        "LOC": "LOCATION",
        "MISC": "MISCELLANEOUS",
    }
    examples = []
    for index, row in enumerate(dataset):
        text, offsets, cursor = "", [], 0
        for token in row["tokens"]:
            if text:
                text += " "
                cursor += 1
            start = cursor
            text += token
            cursor += len(token)
            offsets.append((start, cursor))
        entities: list[Entity] = []
        active: tuple[int, str] | None = None
        for position, tag_id in enumerate(row["ner_tags"]):
            tag = names[tag_id]
            changed_label = tag != "O" and active and label_map[tag[2:]] != active[1]
            if tag == "O" or tag.startswith("B-") or changed_label:
                if active:
                    start, label = active
                    end = offsets[position - 1][1]
                    entities.append(Entity(start, end, label, text[start:end]))
                    active = None
            if tag.startswith("B-") or (tag.startswith("I-") and active is None):
                active = (offsets[position][0], label_map[tag[2:]])
        if active:
            start, label = active
            entities.append(Entity(start, offsets[-1][1], label, text[start : offsets[-1][1]]))
        examples.append(Example(f"conll-{index}", text, tuple(entities)))
    return examples


def _long_context_examples() -> list[Example]:
    """Build deterministic fictional calls with entities around BERT's 512-token edge."""
    examples = []
    specifications = (
        (
            "long-boundary",
            "Caller " + "okay " * 508,
            "Mira Sol",
            " confirmed the fictional return.",
            "PERSON",
            "The two-token name straddles BERT's first 510-content-token window.",
        ),
        (
            "long-after-boundary",
            "Agent " + "noted " * 560,
            "Northwind Relay",
            " as the fictional employer.",
            "ORGANIZATION",
            "The organization occurs after a one-pass 512-token encoder would truncate.",
        ),
        (
            "long-multiple-windows",
            "Representative " + "confirmed " * 740,
            "Vela Harbor",
            " as the fictional destination.",
            "LOCATION",
            "The location requires a later overlapping window but remains below 8192 tokens.",
        ),
    )
    for example_id, prefix, value, suffix, label, note in specifications:
        text = prefix + value + suffix
        start = len(prefix)
        examples.append(
            Example(example_id, text, (Entity(start, start + len(value), label, value),), note)
        )
    return examples


def _overlap_count(entities: list[Entity]) -> int:
    return sum(
        left.start < right.end and right.start < left.end
        for index, left in enumerate(entities)
        for right in entities[index + 1 :]
    )


def _compare_gliner(
    args: argparse.Namespace, checkpoint: dict[str, Any], manifest_schema_version: int
) -> None:
    examples = load_jsonl(args.data)
    schema = load_runtime_schema(args.schema)
    adapter = GLiNERAdapter(
        schema,
        args.model,
        args.revision,
        wording="concise",
        threshold=args.operating_threshold,
    )
    runs = []
    predictions_by_run: dict[tuple[str, float], list[list[Entity]]] = {}
    for wording in ("concise", "descriptive"):
        adapter.wording = wording
        adapter.threshold = min(args.thresholds)
        candidates = [adapter.predict(example.text) for example in examples]
        for threshold in args.thresholds:
            # GLiNER resolves candidates from highest score downward, so removing
            # candidates below a stricter threshold after decoding is equivalent
            # to rerunning its greedy flat decoder at that stricter threshold.
            predictions = [
                [entity for entity in predicted if float(entity.score or 0) > threshold]
                for predicted in candidates
            ]
            predictions_by_run[(wording, threshold)] = predictions
            runs.append(
                {
                    "wording": wording,
                    "threshold": threshold,
                    "results": build_results(examples, predictions, args.model, args.revision),
                }
            )
    fixed_adapter = HuggingFaceNER(DEFAULT_MODEL, DEFAULT_REVISION, args.operating_threshold)
    fixed = build_results(
        examples,
        [fixed_adapter.predict(example.text) for example in examples],
        DEFAULT_MODEL,
        DEFAULT_REVISION,
    )
    flat_predictions = predictions_by_run[("concise", args.operating_threshold)]
    adapter.wording = "concise"
    adapter.threshold = args.operating_threshold
    adapter.flat_ner = False
    adapter.multi_label = True
    overlap_diagnostics = []
    for example, flat in zip(examples, flat_predictions, strict=True):
        nested = adapter.predict(example.text)
        overlap_diagnostics.append(
            {
                "id": example.id,
                "flat_count": len(flat),
                "nested_count": len(nested),
                "overlap_count": _overlap_count(nested),
                "predictions": [asdict(entity) for entity in nested],
            }
        )
    artifact = {
        "model": args.model,
        "revision": args.revision,
        "license": checkpoint["license"],
        "training_data": checkpoint["training_data"],
        "context_length_words": checkpoint["context_length"],
        "maximum_span_width_words": checkpoint["maximum_span_width"],
        "manifest": str(BENCHMARK_MANIFEST_REFERENCE),
        "manifest_schema_version": manifest_schema_version,
        "schema": {
            "name": schema.name,
            "path": str(args.schema),
            "labels": [asdict(label) for label in schema.labels],
        },
        "example_count": len(examples),
        "operating_threshold": args.operating_threshold,
        "fixed": fixed,
        "runs": runs,
        "overlap_diagnostics": overlap_diagnostics,
    }
    artifact["reproduce_command"] = shlex.join(
        [
            "uv",
            "run",
            "--extra",
            "gliner",
            "nerdact",
            "compare-gliner",
            "--data",
            _cli_path(args.data),
            "--schema",
            _cli_path(args.schema),
            "--model",
            args.model,
            "--revision",
            args.revision,
            "--thresholds",
            *(str(value) for value in args.thresholds),
            "--operating-threshold",
            str(args.operating_threshold),
            "--output",
            _cli_path(args.output),
            "--html",
            _cli_path(args.html),
        ]
    )
    artifact["provenance"] = build_provenance(
        ROOT,
        [Path(args.data), Path(args.schema), DEFAULT_BENCHMARK_MANIFEST],
        {
            "command": "compare-gliner",
            "model": args.model,
            "revision": args.revision,
            "thresholds": args.thresholds,
            "operating_threshold": args.operating_threshold,
        },
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(artifact, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    write_runtime_comparison(artifact, args.html)
    print(f"Wrote {args.output} and {args.html}")


def _pii_system_results(
    examples: list[Example], model_predictions: list[list[Entity]], model: str, revision: str
) -> dict[str, Any]:
    rule_predictions = [detect_structured_pii(example.text) for example in examples]
    resolved_rules = [resolve_pii_overlaps(rules) for rules in rule_predictions]
    resolutions = [
        resolve_pii_overlaps(model + rules)
        for model, rules in zip(model_predictions, rule_predictions, strict=True)
    ]
    return {
        "model": build_results(examples, model_predictions, model, revision, PII_LABELS),
        "rules": build_results(
            examples,
            [list(result.entities) for result in resolved_rules],
            "deterministic-detectors",
            None,
            PII_LABELS,
        ),
        "hybrid": build_results(
            examples,
            [list(result.entities) for result in resolutions],
            f"{model} + deterministic-detectors",
            revision,
            PII_LABELS,
        ),
        "rejected_conflicts": [
            {"id": example.id, "entities": [asdict(entity) for entity in result.rejected]}
            for example, result in zip(examples, resolutions, strict=True)
        ],
    }


def _select_recall_threshold(runs: list[dict[str, Any]]) -> float:
    """Choose from calibration only: exact recall, then character and utility costs."""

    def rank(run: dict[str, Any]) -> tuple[float, float, float, float, float]:
        metrics = run["systems"]["hybrid"]["metrics"]
        return (
            -metrics["micro"]["recall"],
            metrics["characters"]["leakage_rate"],
            metrics["transcripts"]["any_leak_rate"],
            metrics["characters"]["over_redaction_rate"],
            -run["threshold"],
        )

    return min(runs, key=rank)["threshold"]


def _validate_pii_splits(
    calibration_path: Path,
    evaluation_path: Path,
    calibration: list[Example],
    evaluation: list[Example],
) -> None:
    """Reject evaluation data that is not locally independent from calibration."""
    if calibration_path.resolve() == evaluation_path.resolve():
        raise ValueError("calibration and evaluation data must be different files")
    shared_ids = {example.id for example in calibration} & {example.id for example in evaluation}
    if shared_ids:
        raise ValueError(f"calibration and evaluation IDs overlap: {sorted(shared_ids)}")

    def normalized(text: str) -> str:
        return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text)).strip().casefold()

    calibration_texts = {normalized(example.text) for example in calibration}
    duplicate_text_ids = [
        example.id for example in evaluation if normalized(example.text) in calibration_texts
    ]
    if duplicate_text_ids:
        raise ValueError(
            "calibration and evaluation contain duplicate normalized text in evaluation IDs: "
            f"{sorted(duplicate_text_ids)}"
        )


def _compare_pii(
    args: argparse.Namespace, checkpoint: dict[str, Any], manifest_schema_version: int
) -> None:
    calibration = load_jsonl(args.calibration_data, PII_LABELS)
    evaluation = load_jsonl(args.evaluation_data, PII_LABELS)
    _validate_pii_splits(args.calibration_data, args.evaluation_data, calibration, evaluation)
    schema = load_pii_schema(args.schema)
    adapter = GLiNER2PIIAdapter(schema, args.model, args.revision, min(args.thresholds))
    calibration_runs = []
    for threshold in args.thresholds:
        adapter.threshold = threshold
        systems = _pii_system_results(
            calibration,
            [adapter.predict(example.text) for example in calibration],
            args.model,
            args.revision,
        )
        calibration_runs.append({"threshold": threshold, "systems": systems})
    operating_threshold = _select_recall_threshold(calibration_runs)
    adapter.threshold = operating_threshold
    evaluation_systems = _pii_system_results(
        evaluation,
        [adapter.predict(example.text) for example in evaluation],
        args.model,
        args.revision,
    )
    artifact = {
        "model": args.model,
        "revision": args.revision,
        "license": checkpoint["license"],
        "parameters": checkpoint["parameters"],
        "training_data": checkpoint["training_data"],
        "context_length": checkpoint["context_length"],
        "context_unit": checkpoint["context_unit"],
        "decoder": checkpoint["decoder"],
        "manifest": str(BENCHMARK_MANIFEST_REFERENCE),
        "manifest_schema_version": manifest_schema_version,
        "schema": str(args.schema),
        "calibration_data": str(args.calibration_data),
        "evaluation_data": str(args.evaluation_data),
        "selection_policy": (
            "maximize calibration hybrid exact recall; then minimize character leakage, "
            "transcript any-leak rate, and over-redaction; then prefer the higher threshold"
        ),
        "operating_threshold": operating_threshold,
        "calibration_runs": calibration_runs,
        "evaluation": evaluation_systems,
    }
    artifact["reproduce_command"] = shlex.join(
        [
            "uv",
            "run",
            "--extra",
            "pii",
            "nerdact",
            "compare-pii",
            "--calibration-data",
            _cli_path(args.calibration_data),
            "--evaluation-data",
            _cli_path(args.evaluation_data),
            "--schema",
            _cli_path(args.schema),
            "--model",
            args.model,
            "--revision",
            args.revision,
            "--thresholds",
            *(str(value) for value in args.thresholds),
            "--output",
            _cli_path(args.output),
            "--html",
            _cli_path(args.html),
        ]
    )
    artifact["provenance"] = build_provenance(
        ROOT,
        [
            Path(args.calibration_data),
            Path(args.evaluation_data),
            Path(args.schema),
            DEFAULT_BENCHMARK_MANIFEST,
        ],
        {
            "command": "compare-pii",
            "model": args.model,
            "revision": args.revision,
            "thresholds": args.thresholds,
            "operating_threshold": operating_threshold,
        },
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(artifact, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    write_pii_comparison(artifact, args.html)
    print(
        f"Selected threshold {operating_threshold:.2f} on calibration; "
        f"wrote {args.output} and {args.html}"
    )


def main() -> None:
    manifest = load_benchmark_manifest()
    checkpoints = manifest["checkpoints"]
    experiments = manifest["experiments"]
    conll = manifest["datasets"]["conll2003"]
    runtime_checkpoint = checkpoints[experiments["runtime-labels"]["checkpoint"]]
    pii_checkpoint = checkpoints[experiments["practical-pii"]["checkpoint"]]
    parser = argparse.ArgumentParser(prog="nerdact", description="Span-first NER teaching demo")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("demo", "report"):
        command = sub.add_parser(
            name,
            help=f"run the checked-in corpus{' and write the report' if name == 'report' else ''}",
        )
        command.add_argument("--data", type=Path, default=DEFAULT_DATA)
        _model_args(command)
        if name == "report":
            command.add_argument("--output", default="artifacts/results.json")
            command.add_argument("--html", default="docs/index.html")
    text_parser = sub.add_parser("redact", help="redact arbitrary text")
    text_parser.add_argument("text", nargs="?", help="text; reads stdin when omitted")
    _model_args(text_parser)
    benchmark = sub.add_parser("benchmark", help="evaluate a bounded CoNLL-2003 subset")
    benchmark.add_argument("--limit", type=int, default=200)
    benchmark.add_argument("--split", choices=tuple(conll["allowed_splits"]), default="test")
    _model_args(benchmark)
    compare = sub.add_parser("compare", help="compare pinned fixed-label NER checkpoints")
    compare.add_argument("--data", type=Path, default=DEFAULT_DATA)
    compare.add_argument("--limit", type=int, default=200)
    compare.add_argument(
        "--split",
        choices=(experiments["classic"]["split"],),
        default=experiments["classic"]["split"],
    )
    compare.add_argument("--threshold", type=float, default=0.5)
    compare.add_argument("--timing-repeats", type=int, default=3)
    compare.add_argument("--output", type=Path, default=Path("artifacts/benchmark.json"))
    compare.add_argument("--html", type=Path, default=Path("docs/benchmark.html"))
    modern = sub.add_parser(
        "compare-modern", help="compare BERT with a pinned ModernBERT NER checkpoint"
    )
    modern.add_argument("--limit", type=int, default=200)
    modern.add_argument(
        "--split",
        choices=(experiments["modern"]["split"],),
        default=experiments["modern"]["split"],
    )
    modern.add_argument("--threshold", type=float, default=0.5)
    modern.add_argument("--timing-repeats", type=int, default=3)
    modern.add_argument("--output", type=Path, default=Path("artifacts/modern-benchmark.json"))
    modern.add_argument("--html", type=Path, default=Path("docs/modern-encoders.html"))
    gliner = sub.add_parser(
        "compare-gliner", help="sweep GLiNER label wording and confidence thresholds"
    )
    gliner.add_argument("--data", type=Path, default=DEFAULT_DATA)
    gliner.add_argument("--schema", type=Path, default=DEFAULT_RUNTIME_SCHEMA)
    gliner.add_argument("--model", default=runtime_checkpoint["model"])
    gliner.add_argument("--revision", default=runtime_checkpoint["revision"])
    gliner.add_argument(
        "--thresholds", type=float, nargs="+", default=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    )
    gliner.add_argument("--operating-threshold", type=float, default=0.5)
    gliner.add_argument("--output", type=Path, default=Path("artifacts/gliner-benchmark.json"))
    gliner.add_argument("--html", type=Path, default=Path("docs/runtime-labels.html"))
    pii = sub.add_parser(
        "compare-pii", help="calibrate and evaluate the contextual-plus-rule PII system"
    )
    pii.add_argument("--calibration-data", type=Path, default=DEFAULT_PII_CALIBRATION_DATA)
    pii.add_argument("--evaluation-data", type=Path, default=DEFAULT_PII_EVALUATION_DATA)
    pii.add_argument("--schema", type=Path, default=DEFAULT_PII_SCHEMA)
    pii.add_argument("--model", default=pii_checkpoint["model"])
    pii.add_argument("--revision", default=pii_checkpoint["revision"])
    pii.add_argument(
        "--thresholds", type=float, nargs="+", default=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    )
    pii.add_argument("--output", type=Path, default=Path("artifacts/pii-benchmark.json"))
    pii.add_argument("--html", type=Path, default=Path("docs/practical-pii.html"))
    summary = sub.add_parser(
        "summarize", help="write the conclusions page from checked-in benchmark artifacts"
    )
    summary.add_argument(
        "--benchmark", type=Path, default=Path("artifacts/benchmark.json")
    )
    summary.add_argument("--pii", type=Path, default=Path("artifacts/pii-benchmark.json"))
    summary.add_argument("--html", type=Path, default=Path("docs/conclusion.html"))
    args = parser.parse_args()
    scalar_thresholds = [
        value
        for name in ("threshold", "operating_threshold")
        if (value := getattr(args, name, None)) is not None
    ]
    if any(not math.isfinite(value) or not 0 <= value <= 1 for value in scalar_thresholds):
        parser.error("thresholds must be finite and between 0 and 1")
    if args.command in ("compare-gliner", "compare-pii"):
        if any(
            not math.isfinite(value) or not 0 <= value <= 1 for value in args.thresholds
        ):
            parser.error("--thresholds must be finite and between 0 and 1")
        if len(args.thresholds) != len(set(args.thresholds)):
            parser.error("--thresholds must be unique")
    if args.command == "compare-gliner":
        if (args.model, args.revision) != (
            runtime_checkpoint["model"],
            runtime_checkpoint["revision"],
        ):
            parser.error("compare-gliner checkpoint must match the benchmark manifest")
        if args.operating_threshold not in args.thresholds:
            parser.error("--operating-threshold must be included in --thresholds")
        _compare_gliner(args, runtime_checkpoint, manifest["schema_version"])
    elif args.command == "compare-pii":
        if (args.model, args.revision) != (pii_checkpoint["model"], pii_checkpoint["revision"]):
            parser.error("compare-pii checkpoint must match the benchmark manifest")
        _compare_pii(args, pii_checkpoint, manifest["schema_version"])
    elif args.command == "summarize":
        write_learning_summary(
            json.loads(args.benchmark.read_text(encoding="utf-8")),
            json.loads(args.pii.read_text(encoding="utf-8")),
            args.html,
        )
        print(f"Wrote {args.html}")
    elif args.command in ("demo", "report"):
        _run_corpus(args, args.command == "report")
    elif args.command == "redact":
        import sys

        text = args.text if args.text is not None else sys.stdin.read()
        entities = HuggingFaceNER(args.model, _revision(args), args.threshold).predict(text)
        print(redact(text, entities)[0])
    elif args.command == "benchmark":
        if not 1 <= args.limit <= 5000:
            parser.error("--limit must be between 1 and 5000")
        examples = _conll_examples(args.limit, args.split, conll["id"], conll["revision"])
        revision = _revision(args)
        adapter = HuggingFaceNER(args.model, revision, args.threshold)
        results = build_results(
            examples, [adapter.predict(x.text) for x in examples], args.model, revision
        )
        print(json.dumps(results["metrics"], default=lambda value: value.__dict__, indent=2))
    else:
        if not 1 <= args.limit <= 5000:
            parser.error("--limit must be between 1 and 5000")
        if not 1 <= args.timing_repeats <= 100:
            parser.error("--timing-repeats must be between 1 and 100")
        modern_comparison = args.command == "compare-modern"
        domain_examples = _long_context_examples() if modern_comparison else load_jsonl(args.data)
        benchmark_examples = _conll_examples(args.limit, args.split, conll["id"], conll["revision"])
        rows = []
        experiment_name = "modern" if modern_comparison else "classic"
        profiles = _model_profiles(manifest, experiment_name)
        reproduce_command = [
            "uv",
            "run",
            "--extra",
            "benchmark",
            "nerdact",
            args.command,
            "--limit",
            str(args.limit),
            "--split",
            args.split,
            "--threshold",
            str(args.threshold),
            "--timing-repeats",
            str(args.timing_repeats),
            "--output",
            _cli_path(args.output),
            "--html",
            _cli_path(args.html),
        ]
        if not modern_comparison:
            reproduce_command.extend(("--data", _cli_path(args.data)))
        reproduce_command_text = shlex.join(reproduce_command)
        context = multiprocessing.get_context("spawn")
        for profile in profiles:
            with context.Pool(1) as pool:
                row = pool.apply(
                    _compare_model,
                    (
                        profile,
                        domain_examples,
                        benchmark_examples,
                        args.threshold,
                        args.timing_repeats,
                    ),
                )
            row["domain"]["provenance"] = build_provenance(
                ROOT,
                [DEFAULT_BENCHMARK_MANIFEST]
                + ([] if modern_comparison else [Path(args.data)]),
                {
                    "command": args.command,
                    "model": profile.model,
                    "revision": profile.revision,
                    "threshold": args.threshold,
                    "timing_repeats": args.timing_repeats,
                },
            )
            row["domain"]["reproduce_command"] = reproduce_command_text
            rows.append(row)
            artifact_name = (
                "results.json"
                if profile.model == DEFAULT_MODEL and not modern_comparison
                else f"{Path(profile.report).stem}-results.json"
            )
            write_results(
                row["domain"],
                Path("artifacts") / artifact_name,
                Path("docs") / profile.report,
            )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        provenance = build_provenance(
            ROOT,
            [DEFAULT_BENCHMARK_MANIFEST] + ([] if modern_comparison else [Path(args.data)]),
            {
                "command": args.command,
                "experiment": experiment_name,
                "dataset": conll["id"],
                "dataset_revision": conll["revision"],
                "split": args.split,
                "limit": args.limit,
                "threshold": args.threshold,
                "timing_repeats": args.timing_repeats,
            },
        )
        artifact = {
            "manifest": str(BENCHMARK_MANIFEST_REFERENCE),
            "manifest_schema_version": manifest["schema_version"],
            "experiment": experiment_name,
            "dataset": conll["id"],
            "dataset_revision": conll["revision"],
            "split": args.split,
            "limit": args.limit,
            "threshold": args.threshold,
            "timing_corpus": (
                "deterministic long-context fixtures in nerdact.cli"
                if modern_comparison
                else str(args.data)
            ),
            "timing_repeats": args.timing_repeats,
            "models": rows,
            "provenance": provenance,
            "reproduce_command": reproduce_command_text,
        }
        args.output.write_text(
            json.dumps(artifact, indent=2, ensure_ascii=False, allow_nan=False)
            + "\n",
            encoding="utf-8",
        )
        write_comparison(
            rows,
            args.html,
            args.split,
            args.limit,
            "modern" if modern_comparison else "classic",
            provenance,
            reproduce_command_text,
        )
        print(f"Wrote {args.output}, {args.html}, and {len(rows)} transcript reports")

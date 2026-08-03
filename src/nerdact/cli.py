"""Command-line interface."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .model import DEFAULT_MODEL, DEFAULT_REVISION, HuggingFaceNER
from .redact import redact
from .report import build_results, write_comparison, write_results
from .schema import Entity, Example, load_jsonl

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA = ROOT / "data" / "transcripts.jsonl"
DATASET = "BramVanroy/conll2003"
DATASET_REVISION = "4ffbd53d9e0b92b473b9b7dcff12f53e7c17ce0c"
CANDIDATE_MODEL = "Jean-Baptiste/roberta-large-ner-english"
CANDIDATE_REVISION = "8f3abc1ef81ffbbb0e80568d4fed1dd10d459548"


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


def _run_corpus(args: argparse.Namespace, make_report: bool) -> None:
    examples = load_jsonl(args.data)
    revision = _revision(args)
    adapter = HuggingFaceNER(args.model, revision, args.threshold)
    predictions = [adapter.predict(example.text) for example in examples]
    results = build_results(examples, predictions, args.model, revision)
    if make_report:
        write_results(results, Path(args.output), Path(args.html))
        print(f"Wrote {args.output} and {args.html}")
    else:
        for record in results["examples"]:
            print(f"\n[{record['id']}]\n{record['redacted']}")
        print("\nExact micro metrics:", json.dumps(results["metrics"]["micro"], indent=2))


def _conll_examples(limit: int, split: str) -> list[Example]:
    from importlib import import_module

    load_dataset = import_module("datasets").load_dataset
    dataset = load_dataset(DATASET, revision=DATASET_REVISION, split=f"{split}[:{limit}]")
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


def main() -> None:
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
    benchmark.add_argument("--split", choices=("validation", "test"), default="test")
    _model_args(benchmark)
    compare = sub.add_parser("compare", help="compare the baseline with a larger NER model")
    compare.add_argument("--data", type=Path, default=DEFAULT_DATA)
    compare.add_argument("--limit", type=int, default=200)
    compare.add_argument("--split", choices=("validation",), default="validation")
    compare.add_argument("--candidate-model", default=CANDIDATE_MODEL)
    compare.add_argument("--candidate-revision", default=CANDIDATE_REVISION)
    compare.add_argument("--threshold", type=float, default=0.5)
    compare.add_argument("--html", type=Path, default=Path("docs/benchmark.html"))
    args = parser.parse_args()
    if args.command in ("demo", "report"):
        _run_corpus(args, args.command == "report")
    elif args.command == "redact":
        import sys

        text = args.text if args.text is not None else sys.stdin.read()
        entities = HuggingFaceNER(args.model, _revision(args), args.threshold).predict(text)
        print(redact(text, entities)[0])
    elif args.command == "benchmark":
        if not 1 <= args.limit <= 5000:
            parser.error("--limit must be between 1 and 5000")
        examples = _conll_examples(args.limit, args.split)
        revision = _revision(args)
        adapter = HuggingFaceNER(args.model, revision, args.threshold)
        results = build_results(
            examples, [adapter.predict(x.text) for x in examples], args.model, revision
        )
        print(json.dumps(results["metrics"], default=lambda value: value.__dict__, indent=2))
    else:
        if not 1 <= args.limit <= 5000:
            parser.error("--limit must be between 1 and 5000")
        domain_examples = load_jsonl(args.data)
        benchmark_examples = _conll_examples(args.limit, args.split)
        rows = []
        for name, model, revision in (
            ("Baseline · BERT base", DEFAULT_MODEL, DEFAULT_REVISION),
            ("Candidate · RoBERTa large", args.candidate_model, args.candidate_revision),
        ):
            adapter = HuggingFaceNER(model, revision, args.threshold)
            domain = build_results(
                domain_examples, [adapter.predict(x.text) for x in domain_examples], model, revision
            )
            benchmark_results = build_results(
                benchmark_examples,
                [adapter.predict(x.text) for x in benchmark_examples],
                model,
                revision,
            )
            rows.append(
                {
                    "name": name,
                    "model": model,
                    "revision": revision,
                    "domain": domain,
                    "benchmark": benchmark_results,
                }
            )
            if model == DEFAULT_MODEL:
                write_results(domain, Path("artifacts/results.json"), Path("docs/index.html"))
            else:
                write_results(
                    domain,
                    Path("artifacts/roberta-large-results.json"),
                    Path("docs/roberta-large.html"),
                )
        write_comparison(rows, args.html, args.split, args.limit)
        print(f"Wrote {args.html} and both transcript reports")

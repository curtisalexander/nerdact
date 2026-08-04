# NERdact

NERdact is a compact teaching repository for named-entity recognition (NER), exact
character-span evaluation, and prediction-driven redaction. It runs pinned models over
fictional call transcripts so you can inspect where a newswire benchmark does—and does
not—transfer to conversational text.

> **Teaching project, not a privacy system.** The baseline recognizes only `PERSON`,
> `ORGANIZATION`, `LOCATION`, and `MISCELLANEOUS`. It does not detect all personally
> identifiable information (PII), and replacing names with placeholders is
> pseudonymization—not anonymization. Use only synthetic or otherwise non-sensitive text.

## Start with your goal

| I want to… | Start here |
| --- | --- |
| Understand NER, spans, evaluation, and redaction | [Baseline visual lesson](https://curtisalexander.github.io/nerdact/) → [`learning/`](learning/) |
| Reproduce or audit the experiments | [`learning/` reproduction guide](learning/README.md#reproduce-the-lessons) → [`data/benchmark-manifest.json`](data/benchmark-manifest.json) |
| Build from a small implementation | [`examples/`](examples/) |

The implementation examples offer two different scopes:

- [`examples/roberta_ner/`](examples/roberta_ner/) performs conventional four-label NER.
- [`examples/gliner2_pii_hybrid/`](examples/gliner2_pii_hybrid/) combines a broader
  contextual PII model with deterministic structural rules.

They use different taxonomies and evaluation corpora, so their scores are not directly
comparable.

## Quick start

Python 3.11+ and [uv](https://docs.astral.sh/uv/) are required.

```sh
uv sync
uv run nerdact demo
uv run nerdact redact "Mira called from Portland"
uv run nerdact report  # artifacts/results.json + docs/index.html
```

The first inference command downloads pinned model weights into Hugging Face's cache.
Generated JSON and HTML retain their input text; protect them if you use anything other
than the checked-in fictional data.

## The core idea: use source spans everywhere

A token-classification model assigns labels such as `B-PER`, `I-PER`, and `O` to model
tokens. One visible word can become several subword tokens, so those predictions are not
safe replacement instructions by themselves.

NERdact aggregates token output and normalizes every finding to one shared contract:

```text
(start, end, label, text, score)
```

`[start, end)` indexes the exact original Python string. Evaluation and redaction both use
these source-character spans, making boundary errors visible and edits deterministic.

An exact match requires the same **start, end, and label** as the human annotation. Alongside
precision, recall, and F1, NERdact reports:

- **character leakage:** annotated entity characters not covered by any prediction;
- **over-redaction:** non-entity characters covered by predictions.

Redaction assigns typed placeholders in reading order and edits right-to-left. Repeated
occurrences of the same label and exact text reuse a placeholder. A miss remains visible, a
wrong label produces the wrong placeholder type, and retained context may still identify a
person.

The [baseline lesson](https://curtisalexander.github.io/nerdact/) demonstrates this flow with
real predictions before introducing the aggregate metrics.

## What the experiments teach

| Experiment | Question | Main lesson |
| --- | --- | --- |
| [Baseline](https://curtisalexander.github.io/nerdact/) | What fails when newswire NER meets fictional calls? | Casing, boundaries, products, and support language create inspectable errors. |
| [Classic checkpoints](https://curtisalexander.github.io/nerdact/benchmark.html) | How do four fixed-label checkpoints compare? | The strongest measured checkpoint motivated the small RoBERTa example; size alone does not explain quality. |
| [Modern encoders](https://curtisalexander.github.io/nerdact/modern-encoders.html) | Does a longer context window solve long-input NER? | Architectural capacity is not the same as task-specific fine-tuning quality; overlapping BERT windows performed better here. |
| [Runtime labels](https://curtisalexander.github.io/nerdact/runtime-labels.html) | What changes when labels become inference inputs? | Prompt wording and thresholds change behavior without changing weights and require careful evaluation. |
| [Practical PII](https://curtisalexander.github.io/nerdact/practical-pii.html) | Can a broader model plus rules reduce leakage? | The hybrid improved held-out results but still leaked annotated characters in 3 of 16 fictional transcripts. |

Two headline outcomes explain the included examples:

- The RoBERTa checkpoint reached **88.4% exact-span F1** with **1.9% character leakage** on
  twenty fictional four-label transcripts.
- The separately evaluated PII hybrid reached **91.8% exact-span F1** with **14.6% character
  leakage** on sixteen held-out fictional PII transcripts.

These are small, designed evaluations—not estimates of production safety or universal model
rankings. The classic CoNLL columns are also NERdact pipeline measurements: dataset tokens are
joined with spaces, retokenized, and scored as exact character spans. They do not reproduce
standard token-level CoNLL/seqeval scores from model cards. The RoBERTa author trained on the
original CoNLL test split, so NERdact uses the validation split for that comparison.

## Reproduce and inspect

The baseline is lightweight; the optional experiments download additional pinned models and,
for classic/modern comparisons, a bounded CoNLL mirror.

```sh
# Core checks and baseline
uv run ruff check .
uv run ty check
uv run pytest
uv run nerdact report

# Optional experiments
uv run --extra benchmark nerdact compare --limit 200
uv run --extra benchmark nerdact compare-modern --limit 200
uv run --extra gliner nerdact compare-gliner
uv run --extra pii nerdact compare-pii
```

See [`learning/`](learning/README.md#reproduce-the-lessons) for the recommended order and the opt-in
real-model integration test. Each generated report includes its effective reproduction
command. Machine-readable artifacts in `artifacts/` record model revisions, options,
dependency versions, environment metadata, input/source hashes, Git state, and a run ID.

The optional `BramVanroy/conll2003` mirror is pinned and not redistributed here. Its dataset
card describes Reuters corpus restrictions and says access/use requires agreement; review and
accept those terms yourself.

## Trust boundaries and provenance

- [`data/benchmark-manifest.json`](data/benchmark-manifest.json) is the source of truth for
  accepted checkpoint revisions, licenses, training-data statements, label schemas, context
  limits, decoders, and experiment membership.
- [`data/PII-ANNOTATION.md`](data/PII-ANNOTATION.md) defines the separate twelve-label PII
  taxonomy, boundaries, exclusions, and review procedure.
- Repository code and checked-in fictional project data use the MIT license. Downloaded
  models and datasets retain their own licenses and terms.
- Reports escape transcript and model-controlled HTML, but still contain the original text.
  Model caches, logs, shell history, JSON, and surrounding context can also retain or reveal
  information.

A real privacy workflow needs representative labeled data, broader detection coverage,
access and retention controls, and human review. The practical PII report discusses those
limits in more detail; it does not claim to solve them.

## Repository map

```text
learning/      guided curriculum and reproduction order
examples/      two minimal implementation paths
docs/          generated lessons and experiment reports
src/nerdact/   span schema, model adapters, evaluation, redaction, reports, CLI
data/          fictional corpora, label schemas, and benchmark manifest
tests/         network-free unit tests plus one opt-in cached-model integration test
artifacts/     generated machine-readable results (gitignored)
```

Convenience aliases for setup, reports, checks, and experiments are in the `justfile`.

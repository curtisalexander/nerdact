# Learning and benchmarks

This directory is the front door for understanding *why* the examples use their chosen
models and span-handling design. The benchmark implementation remains in `src/nerdact`, the
fictional evaluation data remains in `data`, and generated lessons remain in `docs`; keeping
those existing paths stable makes every published result reproducible.

## Suggested path

Start with the shared foundation:

1. **NER and source spans** — follow sections 1–2 of the
   [published baseline lesson](https://curtisalexander.github.io/nerdact/#ner), then inspect
   `model.py`, `schema.py`, and `redact.py` in [`src/nerdact`](../src/nerdact/).
2. **Exact evaluation and redaction risk** — continue through
   [evaluation, results, and redaction](https://curtisalexander.github.io/nerdact/#evaluation).
3. **Classic model comparison** — compare DistilBERT, BERT-base, BERT-large, and RoBERTa-large
   in the [classic checkpoint report](https://curtisalexander.github.io/nerdact/benchmark.html). This evaluation motivates the
   minimal [RoBERTa example](../examples/roberta_ner/).

Then choose what matters to you:

- **Implement fixed-label NER:** stop at the [RoBERTa example](../examples/roberta_ner/).
- **Study long inputs:** review the [modern encoder lesson](https://curtisalexander.github.io/nerdact/modern-encoders.html),
  including why a newer architecture did not automatically produce a better NER checkpoint.
- **Explore broader PII:** read the [runtime-label lesson](https://curtisalexander.github.io/nerdact/runtime-labels.html) as an
  optional bridge, then review the separate taxonomy, calibration split, deterministic rules,
   model limitations, and conflict resolution in the [practical PII lesson](https://curtisalexander.github.io/nerdact/practical-pii.html). This evaluation motivates the alternative
   [hybrid PII example](../examples/gliner2_pii_hybrid/).

## Reproduce the lessons

```sh
uv sync
uv run nerdact report
uv run --extra benchmark nerdact compare --limit 200
uv run --extra benchmark nerdact compare-modern --limit 200
uv run --extra gliner nerdact compare-gliner
uv run --extra pii nerdact compare-pii
just integration-test  # real cached BERT pipeline across 512-token windows
```

The first command in each model family downloads pinned weights. Public benchmark access may
require accepting the upstream dataset terms described in the root README. Generated JSON is
written to `artifacts/`; generated, self-contained HTML is written to `docs/`.
The integration test is opt-in because it loads the pinned BERT checkpoint; it runs offline
and fails rather than downloading weights when the snapshot is not already cached.

## How to interpret “best”

There is no model-independent winner. On this repository's small designed evaluations:

- `Jean-Baptiste/roberta-large-ner-english` was the strongest **fixed four-label NER model**:
  88.4% exact-span F1 and 1.9% character leakage on 20 fictional call transcripts.
- `fastino/gliner2-privacy-filter-PII-multi` covers a much broader **PII taxonomy**, but the
  model alone reached 78.0% F1 and leaked 27.0% of annotated characters on its separate
  held-out fixture set.
- The **hybrid PII system**, which combines that model with structurally validated rules,
  reached 91.8% F1 and 14.6% character leakage. It is a system result, not a model result.

These are teaching results on small fictional corpora, not production safety estimates.

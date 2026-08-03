# NERdact

NERdact is a compact teaching repository for running named-entity recognition (NER)
over fictional call transcripts, evaluating exact character spans, and replacing model
findings with typed placeholders such as `[PERSON_1]`. It deliberately exposes the gap
between a newswire benchmark and conversational text.

> **Not production-safe and not a general PII detector.** The default model recognizes
> only PERSON, ORGANIZATION, LOCATION, and MISCELLANEOUS. It does **not** recognize street
> addresses, phone numbers, email addresses, account numbers, or all sensitive data.
> Never send sensitive text to infrastructure you do not control, and do not treat this
> output as anonymization.

## Quick start

Python 3.11+ and [uv](https://docs.astral.sh/uv/) are required.

```sh
uv sync
uv run nerdact demo                  # terminal output over checked-in data
uv run nerdact report                # artifacts/results.json + docs/index.html
uv run nerdact redact "Mira called from Portland"
uv run pytest
```

The first inference command downloads model weights into Hugging Face's cache; model
files are not committed. `just setup`, `just demo`, `just report`, and `just check` are
equivalent convenience recipes. To publish the generated report, enable GitHub Pages
from the repository's `/docs` directory.

## From tokens to stable spans

Token classification labels each model token. In **BIO** encoding, `B-PER` begins a
person, `I-PER` continues it, and `O` is outside an entity. BERT tokenization can split
one visible word into subwords, so token predictions do not directly identify what to
replace in the source. Hugging Face's aggregation combines subword predictions and
returns offsets; `nerdact.model` validates those offsets and normalizes CoNLL's
`PER/ORG/LOC/MISC` labels.

Every downstream component uses one stable contract: `(start, end, label, text, score)`,
where `[start, end)` indexes the original Python string. Character spans survive changes
in tokenizer, make exact boundary errors inspectable, and permit deterministic edits.
The JSONL loader rejects invalid, mismatched, unsupported, and overlapping gold spans.

## Evaluation and redaction

An exact true positive requires the same start, end, **and** label. The report shows
TP/FP/FN and precision/recall/F1 per label, micro totals, and an unweighted macro average
across all four labels. This is stricter—and more directly tied to replacement safety—
than token accuracy. Two coverage diagnostics complement exact metrics:

* **character leakage** = gold-entity characters not covered by any prediction / all
  gold-entity characters;
* **over-redaction** = predicted characters outside every gold entity / all non-gold
  transcript characters.

Predicted entities receive numbered placeholders in reading order. The same normalized
`(label, exact text)` value reuses its placeholder within a transcript. Edits happen
right-to-left so earlier offsets remain valid. Placeholders are typed pseudonyms, not a
reversible vault, and equality reuse can itself reveal repetition.

## Benchmark, domain shift, and model choice

The default `dslim/bert-base-NER` revision is pinned for reproducibility. It was trained
on CoNLL-2003 English newswire; call language, casing, invented products, and support
jargon differ. A good benchmark result therefore does not establish transcript safety.
Use the checked-in domain examples and inspect FP/FN sections rather than relying on one
score.

An optional, bounded comparison (default 200 examples) is available:

```sh
uv run --extra benchmark nerdact benchmark --limit 200
```

This downloads the Parquet-backed `BramVanroy/conll2003` mirror at a pinned revision and
does not redistribute it. That mirror is an exact duplicate of `eriktks/conll2003` made
loadable by modern versions of `datasets`. Its card says access/use requires agreement
and describes the original Reuters corpus restrictions; review and accept those terms
yourself.

Swap models with `--model some/model-id --revision COMMIT`. Before doing so, check its
label map and taxonomy, training domain, language and casing, maximum input length,
aggregation/offset behavior, license, pinned revision, calibration at your threshold,
and performance on representative domain data. A custom model must emit labels that map
to this project's four-label taxonomy.

## Privacy limitations

NER is not anonymization. Misses leak data; false positives destroy useful text;
context can re-identify people after names are replaced; and model caches, logs, shell
history, JSON artifacts, and reports can retain input. The HTML generator escapes all
transcript and model-controlled content, but its report still contains originals. Use
only synthetic/non-sensitive data here. A real system needs a broader threat model,
access controls, retention policy, multiple detection techniques, human review, and
domain-specific validation.

## Provenance, licenses, and layout

* `dslim/bert-base-NER` is MIT licensed; its model card documents CoNLL-2003 training.
* `BramVanroy/conll2003` is fetched only on request at revision
  `4ffbd53d9e0b92b473b9b7dcff12f53e7c17ce0c`; its card identifies it as an exact,
  modern-format mirror and warns that use requires agreement under Reuters source
  restrictions.
* `data/transcripts.jsonl` is fictional, checked-in MIT-licensed project data.

```text
src/nerdact/          schema, lazy model adapter, evaluation, redaction, report, CLI
data/                 validated synthetic JSONL corpus
tests/                network/model-free unit tests using fake pipeline predictions
docs/                 self-contained GitHub Pages report
artifacts/             generated machine-readable results (gitignored)
```

Reproduce everything with `uv sync && uv run ruff check . && uv run ty check && uv run
pytest && uv run nerdact report`. See `justfile` for formatting, benchmark, and cleanup.

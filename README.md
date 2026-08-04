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
uv run --extra benchmark nerdact compare --limit 200  # four pinned checkpoints
uv run --extra benchmark nerdact compare-modern --limit 200
uv run --extra gliner nerdact compare-gliner
uv run --extra pii nerdact compare-pii
uv run pytest
```

## Choose a path

- **Learn and reproduce:** [`learning/`](learning/) organizes the token-classification,
  span, redaction, model-comparison, runtime-label, and practical-PII lessons.
- **Implement:** [`examples/`](examples/) contains two deliberately small, complete paths:
  the best measured fixed-label NER checkpoint and the broader GLiNER2-plus-rules PII hybrid.

The first inference command downloads model weights into Hugging Face's cache; model
files are not committed. `compare` also writes machine-readable results, including
warm latency, throughput, cached snapshot size, peak inference memory, and environment
metadata, to `artifacts/benchmark.json`. `just setup`, `just demo`, `just report`, and
`just check` are equivalent convenience recipes. The checked-in fictional reports are
published from `/docs` at
[`curtisalexander.github.io/nerdact`](https://curtisalexander.github.io/nerdact/).
New artifacts also record a run ID, input and source hashes, effective options, dependency
versions, the Git commit, and whether the generating worktree contained uncommitted changes.

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

[`data/benchmark-manifest.json`](data/benchmark-manifest.json) is the machine-readable source
of truth for benchmark dataset pins, checkpoint revisions, licenses, parameter counts,
training-data statements, label schemas, context limits, decoders, experiment membership, and
report paths. Comparison artifacts record the manifest path and schema version. The loader
rejects incomplete records, unknown references, duplicate identities/report paths, incompatible
fixed-label profiles, and drift between manifest records and adapter defaults.

An optional, bounded comparison (default 200 examples) is available:

```sh
uv run --extra benchmark nerdact benchmark --limit 200
```

This downloads the Parquet-backed `BramVanroy/conll2003` mirror at a pinned revision and
does not redistribute it. That mirror is an exact duplicate of `eriktks/conll2003` made
loadable by modern versions of `datasets`. Its card says access/use requires agreement
and describes the original Reuters corpus restrictions; review and accept those terms
yourself.

To compare four fixed-label checkpoints on both the synthetic transcripts and a clean
CoNLL split, run:

```sh
uv run --extra benchmark nerdact compare --limit 200
```

The comparison includes `dslim` DistilBERT, BERT-base, and BERT-large checkpoints. They
share a publisher, four-label CoNLL task, and approximate sizes of 66M, 110M, and 340M
parameters, making them a useful—though not perfectly controlled—size comparison. It
also includes `Jean-Baptiste/roberta-large-ner-english` as a quality-heavy practical
comparison rather than evidence that RoBERTa's architecture alone caused the result.
That model's author included CoNLL's original **test** split in training, so NERdact
deliberately uses the **validation** split instead of publishing a contaminated test
result. The command regenerates each transcript report and `docs/benchmark.html`.
For comparable cost measurements, every checkpoint runs in a fresh process, receives one
untimed warm-up example, and is then timed over three sequential passes through the twenty
fictional transcripts. Use `--timing-repeats` to increase the sample count. These are
machine-specific single-example device results, not server batching benchmarks.

### Phase 2: modern fixed-label encoders

`compare-modern` evaluates the pinned BERT-base baseline and
`RGarrido03/modernbert-conll2003-ner-large` revision
`62fdc4a112fe832bedda4cee9d467c503bc39355`. Both expose exactly CoNLL's four entity
classes. The ModernBERT snapshot is Apache-2.0 licensed and its card records all nine BIO
labels, CoNLL sample counts, separate validation/test results, three training epochs,
hyperparameters, and a 256-token fine-tuning limit. Its config supports 8192 positions.
The card does not pin the dataset revision or link complete training source, so this is a
practical checkpoint comparison—not an independently reproducible architecture experiment.

The long-context fixtures deterministically place fictional entities across BERT's first
512-token boundary and in later windows. NERdact now asks the Hugging Face pipeline to use
64-token overlaps, preserving original character offsets and resolving duplicate overlap
predictions. ModernBERT receives each fixture in one pass. This tests truncation and boundary
handling, but not realistic long-conversation discourse; repetitive fixtures and a checkpoint
fine-tuned on short newswire sequences cannot establish useful 8192-token NER quality.

Candidate audit (2026-08-03):

* `IsmaelMousa/modernbert-ner-conll2003` at
  `40b4c980c346c99f14a7ff41c027af7f40355f56` has the compatible BIO schema,
  Apache-2.0 metadata, an 8192-position config, training hyperparameters, and validation
  metrics. It was rejected because the card omits the fine-tuning sequence length,
  token-label alignment policy, dataset revision, and exact split/preprocessing recipe.
* `tner/deberta-v3-large-conll2003` at
  `5a5e2661cab3e82d83d2acc0c1edab43964005ce` has the compatible schema and the strongest
  provenance: T-NER training configuration, train split, 128-token limit, seed,
  hyperparameters, and test metrics. It was rejected because the checkpoint repository has
  no license metadata. The MIT license on Microsoft's pretrained base does not establish a
  license for the fine-tuned weights.
* Pretrained `answerdotai/ModernBERT-base` is not an NER checkpoint. Its masked-language-model
  head predicts vocabulary tokens, not BIO entity labels; loading it as token classification
  would create a randomly initialized or incompatible head. A pretrained encoder becomes
  task-specific only after a documented NER fine-tune.

The adapter validates a checkpoint's complete label map before inference. Missing labels,
generic `LABEL_n` heads, and additional/incompatible taxonomies now fail rather than being
silently discarded. The generated Phase 2 lesson and measurements are in
[`docs/modern-encoders.html`](docs/modern-encoders.html).

### Phase 3: runtime-selected labels

`compare-gliner` loads `urchade/gliner_small-v2.1` at revision
`4e091416cf7c3481db542c2a3d26156916f3a47f` through GLiNER's own span decoder—not the
Hugging Face token-classification adapter. The Apache-2.0 checkpoint uses a
DeBERTa-v3-small encoder and identifies `urchade/pile-mistral-v0.1` as training data. Its
configuration limits input to 384 words and candidate spans to 12 words; its card does not
provide a contamination audit for NERdact's fictional examples.

The checked-in [`data/runtime-labels.json`](data/runtime-labels.json) schema supplies concise
and descriptive text for the same four canonical entity types and maps model output back to
typed `PERSON`, `ORGANIZATION`, `LOCATION`, and `MISCELLANEOUS` placeholders. Changing those
descriptions changes model behavior without changing weights. At threshold 0.5 on the twenty
shared-label transcripts, concise prompts reached 85.1% exact F1 with 91.5% recall;
descriptive prompts reached 89.7% F1 with 97.5% precision; fixed BERT reached 80.0% F1. These
are practical checkpoint results on designed examples, not an architecture comparison.

The command sweeps thresholds from 0.1 through 0.7 and writes every prediction to
`artifacts/gliner-benchmark.json`. That sweep is diagnostic, not independent calibration:
choosing the best threshold on these same examples would overstate generalization. GLiNER
thresholds span candidates and then resolves conflicts from highest score downward. Flat NER
suppresses overlaps; nested and multi-label decoding can retain overlapping or competing
spans, which cannot be sent directly to NERdact's intentionally overlap-rejecting redactor.
The generated lesson, curves, changed-prompt examples, and overlap diagnostics are in
[`docs/runtime-labels.html`](docs/runtime-labels.html).

### Phase 4: practical PII redaction

The PII work uses a separate twelve-label transcript taxonomy and calibration/evaluation
split; its results must not be compared directly with the four-label CoNLL experiments.
Candidate audit (2026-08-03):

* `fastino/gliner2-privacy-filter-PII-multi` at revision
  `59894c087cb2923b01f337d4ee72f6ff84d5bdd6` is the selected contextual-model candidate.
  It is an Apache-2.0, roughly 0.3B-parameter GLiNER2 checkpoint trained on 4,910 synthetic
  texts and 129,951 PII mentions in English, French, Spanish, German, Italian, Portuguese,
  and Dutch. Its card and [technical report](https://arxiv.org/abs/2605.09973) publish the
  complete 42-label inventory, generation method, and exact-span SPY results. The current
  snapshot uses a 512-position multilingual DeBERTa-v3-base encoder; ordinary extraction
  does not automatically chunk long inputs, so the adapter must use overlapping long-text
  extraction. NERdact pins GLiNER2 source revision
  `a91fd1d2c72debe43907296c8e375a036c6a4faf` because the needed revision-aware loader and
  long-text API are newer than the current PyPI release. The training corpus and complete
  fine-tuning recipe are not published, the synthetic annotations were not human-validated,
  and evaluation covers only SPY's legal
  and medical English domains. Reported SPY precision is only 0.35–0.37, non-European
  locales and scripts are unmeasured, and the card recommends domain calibration. Its
  trained inventory has no `DEVICE_ID` or `URL` type, so the hybrid system must measure and
  fill those gaps rather than implying model coverage.
* `nvidia/gliner-PII` at revision
  `bd23e8ef4425fd04e34c5204ab49ffaa706eae79` is retained as an audited comparison candidate,
  not the first adapter target. It is a 570M-parameter, English-only GLiNER checkpoint with a
  384-word input limit and 12-word maximum span width. NVIDIA documents approximately 100,000
  synthetic training records from `nvidia/nemotron-pii` (currently revision
  `b70ffaf5ff39e079776134c5bf4381f00a9fd1ed`) and reports strict F1 of 0.70 on Argilla,
  0.64 on AI4Privacy, and 0.87 on Nemotron-PII at threshold 0.3. However, the card claims
  “55+” PII/PHI types without listing them, does not document the benchmark split and label
  mapping needed to reproduce those aggregate scores, and officially supports only English.
  Those omissions prevent a checked mapping to NERdact's taxonomy. The weights use the
  permissive but non-OSI NVIDIA Open Model License—not the repository's MIT license—and
  redistribution requires that license and NVIDIA's attribution notice.

Neither model's published benchmark is evidence of safe transcript redaction. NERdact maps
only documented compatible labels, calibrates the hybrid threshold on eight transcripts, and
then evaluates sixteen separate transcripts. The recall-oriented policy selected `0.70`:
thresholds `0.20` through `0.70` tied on calibration recall and all leakage tie-breakers, so
the declared final tie-break preferred the higher threshold. On held-out examples, the model
alone reached 78.0% exact F1 with 27.0% character leakage, deterministic rules reached 60.9%
F1 with 50.6% leakage, and the resolved hybrid reached 91.8% F1 with 14.6% leakage and no
character over-redaction. Three of sixteen transcripts still leaked at least one annotated
character. Unicode address, free-form credential, date, and username-boundary failures show
why human review remains mandatory. See the per-label results, threshold sweep, provenance,
threat model, retention procedure, and fictional demonstration in
[`docs/practical-pii.html`](docs/practical-pii.html).

The pinned GLiNER2 revision's `extract_entities_long` convenience wrapper references an
undefined variable. The adapter uses the public schema builder and `extract_long` method that
the wrapper itself is intended to call, preserving overlapping chunks and global offsets
without modifying the installed dependency.

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
* `dslim/distilbert-NER` is Apache-2.0 licensed and `dslim/bert-large-NER` is MIT
  licensed; their model cards document the same four-label CoNLL-2003 task.
* `BramVanroy/conll2003` is fetched only on request at revision
  `4ffbd53d9e0b92b473b9b7dcff12f53e7c17ce0c`; its card identifies it as an exact,
  modern-format mirror and warns that use requires agreement under Reuters source
  restrictions.
* `data/transcripts.jsonl` is fictional, checked-in MIT-licensed project data.
* Phase 2's deterministic long-context fixtures are fictional project data generated by
  `_long_context_examples` and covered by the repository's MIT license.
* `urchade/gliner_small-v2.1` is Apache-2.0 licensed; its card identifies
  `urchade/pile-mistral-v0.1` as training data but does not document a contamination audit.
* `fastino/gliner2-privacy-filter-PII-multi` is Apache-2.0 licensed. Its documented training
  corpus is synthetic but is not itself published; the repository pins the evaluated model
  snapshot rather than relying on the mutable model ID.
* `nvidia/gliner-PII` uses the NVIDIA Open Model License. It permits use, modification, and
  commercial distribution subject to its conditions, including redistribution notice and
  attribution requirements; it is not covered by this repository's MIT license.
* Phase 4's separate fictional PII calibration and evaluation files cover twelve labels. Their
  scope and boundary rules are defined in [`data/PII-ANNOTATION.md`](data/PII-ANNOTATION.md).
  Reserved, invalid, fictional, and provider test values are used instead of operational PII.

```text
learning/             guided index over concepts, benchmarks, and generated lessons
examples/             minimal recommended NER and alternative hybrid PII implementations
src/nerdact/          schema, lazy model adapter, evaluation, redaction, report, CLI
data/                 validated synthetic JSONL corpus
tests/                network/model-free unit tests using fake pipeline predictions
docs/                 self-contained GitHub Pages report
artifacts/             generated machine-readable results (gitignored)
```

Reproduce everything with `uv sync && uv run ruff check . && uv run ty check && uv run
pytest && uv run nerdact report`. See `justfile` for formatting, benchmark, and cleanup.

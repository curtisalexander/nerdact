# NERdact learning and evaluation roadmap

NERdact is both executable software and a worked lesson. Each phase must therefore leave
behind reproducible code, pinned inputs, measured results, and documentation that explains
what the experiment can and cannot establish.

## Comparison rules

- Benchmark complete checkpoints, not architecture names. Record the model ID, revision,
  license, parameters, label schema, training data, context length, decoder, and threshold.
- Use exact character-span precision, recall, and F1. Keep leakage and over-redaction beside
  F1 because redaction failures have asymmetric consequences.
- Never compare scores from different datasets as if they were a leaderboard. A four-label
  newswire task and a 42-label PII task answer different questions.
- Separate controlled experiments from practical checkpoint comparisons. Changing the
  encoder, size, tokenizer, fine-tuning data, and labels at once cannot isolate architecture.
- Pin revisions and evaluation data. Document contamination, synthetic data, domain shift,
  licensing restrictions, and unsupported entity types.
- Treat generated reports as sensitive when inputs are sensitive. Only fictional data is
  suitable for the public GitHub Pages site.

## Phase 1 — Classic encoder size and quality

**Learning question:** What do compression, scale, and a stronger classic checkpoint change
when the output taxonomy remains PERSON, ORGANIZATION, LOCATION, and MISCELLANEOUS?

- [x] Establish `dslim/bert-base-NER` as the character-span baseline.
- [x] Compare a pinned RoBERTa-large checkpoint on domain data and uncontaminated CoNLL
  validation examples.
- [x] Add pinned `dslim/distilbert-NER` and `dslim/bert-large-NER` checkpoints.
- [x] Record warm latency, throughput, model download size, and peak memory on the declared
  test machine; quality without cost is not a complete comparison.
- [x] Expand the fictional transcript set with casing, punctuation, uncommon names, hard
  negatives, and boundary cases before interpreting small score differences.
- [x] Publish a lesson explaining distillation, base versus large scale, and why the current
  RoBERTa comparison does not isolate architecture.

**Exit evidence:** one reproducible table for all four checkpoints, per-example reports, and
an explicit quality-versus-cost conclusion.

**Completed:** [`docs/benchmark.html`](docs/benchmark.html) reports the expanded 20-transcript
corpus, 200 uncontaminated CoNLL validation examples, per-checkpoint reports, and isolated
warm-inference measurements from the declared Apple M5 Pro test machine. DistilBERT is the
clear efficiency option, BERT base is the middle-cost compromise, and the practical
RoBERTa-large checkpoint provides the best quality and lowest transcript leakage at roughly
the same cost as BERT large. These designed examples remain too small to establish safety.

## Phase 2 — Modern fixed-label encoders

**Learning question:** Do newer encoder designs improve transcript NER or long-context
handling when evaluated through trustworthy task-specific checkpoints?

- [x] Audit candidate ModernBERT and DeBERTa-v3 NER checkpoints for training data, labels,
  split contamination, license, context length, and model-card completeness.
- [x] Reject candidates that cannot map cleanly to the current four-label evaluation; do not
  silently discard extra labels or merge incompatible annotation policies.
- [x] Add model-specific chunking tests, especially spans crossing a 512-token boundary.
- [x] Compare a 512-token encoder with ModernBERT on synthetic long call transcripts.
- [x] Document why a pretrained base encoder is not a drop-in NER checkpoint.

**Gate:** if no well-documented ModernBERT NER checkpoint exists, record that result instead
of using an opaque community checkpoint. Fine-tuning one ourselves would be a separate phase.

**Completed:** [`docs/modern-encoders.html`](docs/modern-encoders.html) compares a pinned,
four-label ModernBERT-large checkpoint with BERT base on deterministic long fixtures and the
same uncontaminated CoNLL validation subset. BERT uses overlapping 64-token windows; the
fixtures include an entity split by its first 512-token boundary. The accepted ModernBERT
card documents labels, splits, hyperparameters, fine-tuning length, results, and license, but
does not pin its dataset or complete training source. The report therefore treats it as a
practical checkpoint comparison, not isolated architecture evidence. README records rejected
ModernBERT and DeBERTa-v3 candidates and the pretrained-base distinction. On the three
fixtures, chunked BERT recovered every exact span at 20.0 examples/s; ModernBERT recovered
one of three at 10.2 examples/s with 32.4% character leakage, despite reaching 97.7% exact F1
on the first 200 CoNLL validation examples. Long architectural context did not compensate for
this checkpoint's short-sequence fine-tuning and domain shift.

## Phase 3 — Runtime-selected entity labels

**Learning question:** How does label-conditioned span extraction differ from a fixed BIO
token-classification head?

- [x] Add a GLiNER adapter behind the same character-span contract without forcing GLiNER
  through the Hugging Face token-classification adapter.
- [x] Define a checked-in label schema with human-readable descriptions and typed placeholder
  mappings.
- [x] Evaluate label wording, threshold calibration, overlapping spans, and competing labels.
- [x] Compare fixed and runtime-selected models only on a shared subset of compatible labels.
- [x] Add a lesson contrasting BIO token classification with label-conditioned span NER.

**Exit evidence:** reproducible GLiNER results, threshold curves, and examples showing both the
power and instability of changing label descriptions at inference time.

**Completed:** [`docs/runtime-labels.html`](docs/runtime-labels.html) evaluates pinned
`urchade/gliner_small-v2.1` through its native label-conditioned span decoder. The checked-in
schema maps concise and descriptive prompts back to the same four typed placeholders. On the
20 shared-label transcripts at threshold 0.5, concise labels reached 85.1% exact F1 and 91.5%
recall; descriptive labels reached 89.7% F1 and 97.5% precision; the fixed BIO baseline reached
80.0% F1. The 0.1–0.7 curves show that these scores and the precision/recall balance are not
stable under threshold or wording changes. Nested, multi-label diagnostics surfaced overlap
conflicts on two fixtures; flat decoding remains required before redaction. This corpus is the
evaluation set, not an independent calibration set, and the checkpoint's training-data
contamination against the fictional names has not been audited.

## Phase 4 — Practical PII redaction

**Learning question:** Which combination of contextual models and deterministic detectors
minimizes sensitive-data leakage on transcript-like text?

- [x] Create a broader fictional PII corpus and annotation guide covering names, addresses,
  phones, email, governmental IDs, financial IDs, credentials, dates, and digital identity.
- [x] Audit GLiNER2-PII and NVIDIA GLiNER-PII revisions, licenses, training provenance, label
  inventories, language coverage, and benchmark limitations.
- [x] Add a PII-specific model adapter and schema mapping without pretending its metrics are
  comparable to four-label CoNLL results.
- [x] Add deterministic detectors for structurally verifiable data such as email, phone, and
  selected identifiers; preserve detector provenance on every output span.
- [x] Define overlap precedence and conflict handling across model and rule predictions.
- [x] Report per-label exact-span metrics, character leakage, over-redaction, transcript-level
  “any leak” rate, and a recall-oriented operating point.
- [x] Document threat model, local inference, report retention, human review, and why masking
  is not anonymization.

**Exit evidence:** a hybrid demonstration on public fictional data and a documented procedure
for evaluating private call transcripts without publishing them or their reports.

**Foundation:** [`data/PII-ANNOTATION.md`](data/PII-ANNOTATION.md) defines a flat twelve-label
taxonomy and exact-boundary policy. Eight calibration and sixteen held-out evaluation
transcripts cover all labels independently, including hard negatives, spoken and formatted
phones, multiline and Unicode addresses, credentials, and four digital-identity subtypes.
Values are fictional, reserved, deliberately invalid, or provider test values; the split is
designed to prevent choosing thresholds on the same examples used for final reporting.

**Model audit:** [`README.md`](README.md#phase-4-practical-pii-redaction) records immutable
revisions and selects Apache-2.0 `fastino/gliner2-privacy-filter-PII-multi` as the first
contextual-model candidate. Its complete 42-label inventory and seven documented languages
make a checked adapter possible, but its synthetic training corpus is not published, its
reported SPY precision is low, and SPY covers only English legal and medical text. NVIDIA's
English-only 570M checkpoint remains a comparison candidate: it documents its public synthetic
training dataset and aggregate results but not the claimed 55+ label inventory or enough
benchmark mapping detail for a complete taxonomy contract. Neither model covers the entire
twelve-label task as documented; deterministic detectors remain part of the experiment.
The checked-in [`data/pii-labels.json`](data/pii-labels.json) mapping requests only the
selected checkpoint's documented trained labels and maps its fine-grained outputs into the
flat evaluation taxonomy. [`src/nerdact/pii_model.py`](src/nerdact/pii_model.py) preserves
exact character offsets, uses overlapping long-text extraction, and explicitly leaves
`DEVICE_ID` and `URL` without model labels instead of inventing unsupported coverage.
[`src/nerdact/detectors.py`](src/nerdact/detectors.py) adds conservative, locally evaluated
rules for email, URL, formatted phone numbers, validated IP addresses, Luhn-valid payment
cards, checksum-valid routing numbers in routing context, selected government/device formats,
and provider-prefixed API keys. Every rule span carries its exact detector name; unsupported
free-form values remain the contextual model's responsibility.
[`src/nerdact/hybrid.py`](src/nerdact/hybrid.py) merges identical findings and their
provenance, gives deterministic structures precedence over model-only conflicts, prefers an
enclosing URL over structures found inside it, and then uses model confidence and span length.
Rejected conflicts remain in the resolution result for report auditing rather than disappearing.

**Completed:** [`docs/practical-pii.html`](docs/practical-pii.html) selects threshold `0.70`
using only the eight calibration transcripts, then compares model-only, rule-only, and hybrid
results on sixteen held-out fixtures. The hybrid reached 91.8% exact F1 and 87.5% recall with
14.6% character leakage, 0% character over-redaction, and an 18.8% transcript “any leak” rate
(3/16). It improved on the contextual model's 27.0% leakage and the rules' 50.6%, while exact
per-label results expose remaining address, credential, date, and username failures. The
report documents the threat model, local/offline procedure, artifact retention, mandatory
human review, conflict provenance, and why typed masking remains pseudonymization. These
designed fixtures are too small to estimate production safety.

## Cross-cutting work

- [x] Add a machine-readable benchmark manifest so report metadata is not duplicated in code.
- [x] Add environment metadata and deterministic timing methodology.
- [x] Add confidence/threshold sweeps rather than treating `0.5` as universally calibrated.
- [ ] Enable GitHub Pages from `/docs` after all linked generated reports are committed.
- [ ] Keep this roadmap updated with decisions, rejected candidates, and links to result pages.

## Current cautions

- The twenty checked-in transcripts are designed examples, not a representative test set.
- The RoBERTa checkpoint included CoNLL's original test split in training; use validation.
- CoNLL's Reuters-derived data has access and use restrictions described in the README.
- Current fixed-label checkpoints are named-entity models, not general PII detectors.

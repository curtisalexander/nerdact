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

- [ ] Audit candidate ModernBERT and DeBERTa-v3 NER checkpoints for training data, labels,
  split contamination, license, context length, and model-card completeness.
- [ ] Reject candidates that cannot map cleanly to the current four-label evaluation; do not
  silently discard extra labels or merge incompatible annotation policies.
- [ ] Add model-specific chunking tests, especially spans crossing a 512-token boundary.
- [ ] Compare a 512-token encoder with ModernBERT on synthetic long call transcripts.
- [ ] Document why a pretrained base encoder is not a drop-in NER checkpoint.

**Gate:** if no well-documented ModernBERT NER checkpoint exists, record that result instead
of using an opaque community checkpoint. Fine-tuning one ourselves would be a separate phase.

## Phase 3 — Runtime-selected entity labels

**Learning question:** How does label-conditioned span extraction differ from a fixed BIO
token-classification head?

- [ ] Add a GLiNER adapter behind the same character-span contract without forcing GLiNER
  through the Hugging Face token-classification adapter.
- [ ] Define a checked-in label schema with human-readable descriptions and typed placeholder
  mappings.
- [ ] Evaluate label wording, threshold calibration, overlapping spans, and competing labels.
- [ ] Compare fixed and runtime-selected models only on a shared subset of compatible labels.
- [ ] Add a lesson contrasting BIO token classification with label-conditioned span NER.

**Exit evidence:** reproducible GLiNER results, threshold curves, and examples showing both the
power and instability of changing label descriptions at inference time.

## Phase 4 — Practical PII redaction

**Learning question:** Which combination of contextual models and deterministic detectors
minimizes sensitive-data leakage on transcript-like text?

- [ ] Create a broader fictional PII corpus and annotation guide covering names, addresses,
  phones, email, governmental IDs, financial IDs, credentials, dates, and digital identity.
- [ ] Audit GLiNER2-PII and NVIDIA GLiNER-PII revisions, licenses, training provenance, label
  inventories, language coverage, and benchmark limitations.
- [ ] Add a PII-specific model adapter and schema mapping without pretending its metrics are
  comparable to four-label CoNLL results.
- [ ] Add deterministic detectors for structurally verifiable data such as email, phone, and
  selected identifiers; preserve detector provenance on every output span.
- [ ] Define overlap precedence and conflict handling across model and rule predictions.
- [ ] Report per-label exact-span metrics, character leakage, over-redaction, transcript-level
  “any leak” rate, and a recall-oriented operating point.
- [ ] Document threat model, local inference, report retention, human review, and why masking
  is not anonymization.

**Exit evidence:** a hybrid demonstration on public fictional data and a documented procedure
for evaluating private call transcripts without publishing them or their reports.

## Cross-cutting work

- [ ] Add a machine-readable benchmark manifest so report metadata is not duplicated in code.
- [x] Add environment metadata and deterministic timing methodology.
- [ ] Add confidence/threshold sweeps rather than treating `0.5` as universally calibrated.
- [ ] Enable GitHub Pages from `/docs` after all linked generated reports are committed.
- [ ] Keep this roadmap updated with decisions, rejected candidates, and links to result pages.

## Current cautions

- The twenty checked-in transcripts are designed examples, not a representative test set.
- The RoBERTa checkpoint included CoNLL's original test split in training; use validation.
- CoNLL's Reuters-derived data has access and use restrictions described in the README.
- Current fixed-label checkpoints are named-entity models, not general PII detectors.

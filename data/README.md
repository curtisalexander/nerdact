# Synthetic transcript data

`transcripts.jsonl` is fictional, hand-authored data released with this repository
under its MIT license. Names, organizations, and events are invented; resemblance to
real people or organizations is coincidental. Offsets use Unicode Python string indices
and half-open `[start, end)` bounds. Gold labels intentionally stay within CoNLL's four
entity classes. Notes identify examples designed to expose domain mismatch.

The corpus includes conventional positives, repeated values, Unicode, lowercase,
all-caps and erratic casing, punctuation-adjacent boundaries, names with internal
punctuation, spans at transcript edges, and hard negatives. Generic roles, directions,
seasons, and interface values are not entities. A named product may be MISCELLANEOUS;
generic account or support language is not. These are deliberately constructed challenge
cases, not a representative sample of callers, languages, domains, or error frequencies.

`benchmark-manifest.json` is the versioned metadata contract shared by the benchmark commands
and generated artifacts. It pins the public dataset and every evaluated checkpoint, records
audit fields needed to interpret report rows, and defines classic, modern, runtime-label, and
PII experiment membership. Edit and validate this file rather than duplicating checkpoint
metadata in CLI profile constructors.

Phase 4's `pii-calibration.jsonl` and `pii-evaluation.jsonl` are a separate fictional PII
corpus with a broader, non-CoNLL taxonomy. Use calibration examples to choose thresholds and
rule settings before measuring the evaluation split. The label definitions, exact-boundary
policy, exclusions, safety constraints, and private-data review procedure are in
[`PII-ANNOTATION.md`](PII-ANNOTATION.md). Values use reserved domains and IP ranges,
deliberately invalid identifiers, conspicuously fictional values, or provider test card
numbers; none are credentials or accounts that can be used.

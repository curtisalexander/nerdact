# Fictional PII annotation guide

This guide defines the gold character spans in `pii-calibration.jsonl` and
`pii-evaluation.jsonl`. The corpus is a deliberately small challenge set for transcript-like
text, not an estimate of real-world PII frequency or production safety. Every value is
invented, reserved for documentation, deliberately invalid, or a payment-provider test value.
Do not replace these files with private transcripts.

## Experimental split

- **Calibration (8 examples):** inspect predictions and select model thresholds, detector
  settings, and overlap precedence here.
- **Evaluation (16 examples):** run only after those choices are fixed. Do not tune on its
  errors and then report the same run as an unbiased result.

Both files use UTF-8 JSON Lines. Offsets are Python Unicode character indices with half-open
`[start, end)` bounds. Each occurrence is annotated, including repeated values. Gold spans do
not overlap because the redaction contract requires one flat replacement sequence.

## Label inventory

| Label | Include | Exclude |
| --- | --- | --- |
| `PERSON` | A named individual, including multiword and Unicode names | Roles such as “agent” or “billing manager” |
| `ADDRESS` | The complete visible street/mailing address, including locality and postal code | Introducers such as “send to”; a bare non-address place |
| `PHONE_NUMBER` | Printed or spoken phone numbers; include a directly attached extension | “call me” without a number |
| `EMAIL_ADDRESS` | The complete mailbox and domain | Surrounding angle brackets, commas, or sentence punctuation |
| `GOVERNMENT_ID` | Tax, national, passport, or other government-issued identifier values | The document type words |
| `FINANCIAL_ID` | Payment-card, bank-routing, bank-account, or financial-account identifiers | Currency amounts and generic account language |
| `CREDENTIAL` | Passwords, passphrases, API keys, access tokens, and reset secrets | The words “password”, “token”, or “blank” without a secret value |
| `DATE` | Complete visible calendar dates, including month/year expirations | Bare weekdays, relative times, and introducers such as “on” |
| `USERNAME` | Login names and social handles; include a leading `@` when present | The word “user” without a value |
| `IP_ADDRESS` | Complete IPv4 or IPv6 values | Trailing sentence punctuation |
| `DEVICE_ID` | Persistent device identifiers such as IMEI-like or application device IDs | Generic device names such as “handset” |
| `URL` | A complete URL whose path or token identifies a person, profile, or private action | Trailing sentence punctuation and generic public URLs |

`USERNAME`, `IP_ADDRESS`, `DEVICE_ID`, and `URL` form the corpus's **digital identity**
category. The granular labels are retained because detectors have different validation rules
and failures for each structure. `GOVERNMENT_ID` and `FINANCIAL_ID` intentionally group
country- and provider-specific subtypes; future model adapters must document any mapping into
these canonical labels rather than silently dropping or merging model outputs.

## Boundary and ambiguity rules

1. Annotate the smallest complete value that can be replaced without leaving identifying
   fragments. Exclude speaker labels, field names, possessive suffixes, and delimiters.
2. Preserve value-internal spaces, hyphens, punctuation, and line breaks. Include a phone
   extension because leaving it behind can still route to an individual.
3. Treat every explicit calendar date as in scope. This recall-oriented rule avoids requiring
   an annotator to infer whether a date is identifying from incomplete call context.
4. Prefer the more specific structural label over a broad contextual interpretation. An email
   remains `EMAIL_ADDRESS`, not `USERNAME`; a reset link remains `URL`, not `CREDENTIAL`.
5. Do not create nested spans. A profile username embedded inside a URL is represented by the
   enclosing `URL`; annotate `USERNAME` only when it appears separately.
6. Generic category words, roles, interface fields, ordinary counts, relative times, and
   unlabeled organization names are out of scope. Context may still permit re-identification;
   omission from this taxonomy does not mean a value is safe to publish.

## Annotation and review procedure

Annotate from the original text, then verify that every stored `text` equals
`source[start:end]`. A second reviewer should independently check label choice, boundaries,
missed occurrences, and accidental real-looking values. Resolve disagreement against this
guide and record any policy change before evaluating systems. Run the network-free corpus test
with `uv run pytest tests/test_core.py` after every edit; the loader rejects unknown labels,
invalid offsets, mismatched text, duplicate IDs, and overlapping gold spans.

For private transcript evaluation, keep source text and generated reports in an access-
controlled local environment, document retention and deletion, and publish only aggregated
metrics that cannot expose rare values. Human review remains necessary: masking these spans is
pseudonymization, not anonymization.

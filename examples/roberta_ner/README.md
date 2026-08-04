# Recommended fixed-label NER

This example uses only the pinned `Jean-Baptiste/roberta-large-ner-english` model. It was the
strongest fixed-label checkpoint in NERdact's comparisons: 88.4% exact-span F1 and 1.9%
character leakage on 20 fictional call transcripts, plus 97.2% F1 on the bounded CoNLL
validation comparison. Those small evaluations do not prove that it is best for every domain.

From the repository root:

```sh
uv sync
uv run python examples/roberta_ner/redact.py
uv run python examples/roberta_ner/redact.py "Ada works at Acme in Rome"
```

The first run downloads pinned model weights. The implementation keeps the model output as
original `[start, end)` character spans, validates it, and replaces spans right-to-left so
offsets remain correct. A 64-token overlap avoids silently dropping text beyond one model
window.

## Scope

This checkpoint recognizes only `PERSON`, `ORGANIZATION`, `LOCATION`, and `MISCELLANEOUS`.
It does not reliably detect phone numbers, email or street addresses, credentials, account
numbers, or all other PII. The placeholders are pseudonyms, not anonymization. Use fictional
or otherwise non-sensitive input and evaluate representative domain data before adoption.

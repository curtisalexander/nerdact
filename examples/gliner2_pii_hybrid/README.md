# Alternative: hybrid PII detection

This example implements the best broader-PII **system** evaluated in NERdact. It combines one
pinned contextual model, `fastino/gliner2-privacy-filter-PII-multi`, with small deterministic
rules for structurally verifiable values such as email addresses, URLs, IP addresses, phone
numbers, payment cards, and selected identifiers. A deterministic resolver chooses one flat,
auditable set when findings overlap.

From the repository root:

```sh
uv sync --extra pii
uv run --extra pii python examples/gliner2_pii_hybrid/redact.py
uv run --extra pii python examples/gliner2_pii_hybrid/redact.py \
  "Email Ada at ada@example.test or call +1 (202) 555-0147."
```

The first run downloads pinned model weights. The threshold `0.7`, label mapping, long-text
chunking, rules, and overlap policy are the fixed choices evaluated by this repository—there
is no runtime configuration in this minimal example.

## Why hybrid instead of model-only?

On the separate 16-transcript fictional PII evaluation split:

| System | Exact-span F1 | Recall | Character leakage |
| --- | ---: | ---: | ---: |
| GLiNER2 model only | 78.0% | 71.9% | 27.0% |
| Model plus deterministic rules | 91.8% | 87.5% | 14.6% |

The hybrid still leaked annotated characters in 3 of 16 transcripts. Its twelve-label schema
is broader than conventional NER, but incomplete; rules are locale- and format-specific, and
context can re-identify people after replacement. This is pseudonymization—not anonymization
or a production-safe privacy filter. Keep sensitive text local, define retention controls,
evaluate representative languages/domains, and require human review.

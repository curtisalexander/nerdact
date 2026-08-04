# Minimal implementations

These examples are the implementation outcomes of the benchmark and learning work. Each is
complete enough to run, intentionally exposes almost no configuration, and uses validated
source-character spans rather than token strings or global text replacement.

| Example | Use it for | Model/system |
| --- | --- | --- |
| [`roberta_ner`](roberta_ner/) | Conventional person, organization, location, and miscellaneous NER | One pinned RoBERTa-large model |
| [`gliner2_pii_hybrid`](gliner2_pii_hybrid/) | Broader PII detection and typed replacement | One pinned GLiNER2 model plus deterministic structural rules |

Both produce pseudonymized text, not anonymous text. Review their limitations before choosing
one. The benchmark rationale and full lessons are indexed in [`learning/`](../learning/).

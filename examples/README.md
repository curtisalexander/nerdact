# Minimal implementations

These examples are the implementation outcomes of the benchmark and learning work. They use
validated source-character spans so repeated text and tokenizer boundaries cannot cause a
replacement at the wrong location.

| Example | Use it for | Evidence and limitations |
| --- | --- | --- |
| [`roberta_ner`](roberta_ner/) | Conventional person, organization, location, and miscellaneous NER | [Classic checkpoint report](https://curtisalexander.github.io/nerdact/benchmark.html) |
| [`gliner2_pii_hybrid`](gliner2_pii_hybrid/) | Broader PII detection and typed replacement | [Practical PII report](https://curtisalexander.github.io/nerdact/practical-pii.html) |

Both produce pseudonymized text, not anonymous text. Review their limitations before choosing
one. Choose a row, open its README, and run its minimal command. The complete learning path is
indexed in [`learning/`](../learning/).

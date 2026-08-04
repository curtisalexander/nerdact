"""Minimal broad-PII redaction using the evaluated GLiNER2 hybrid system."""

import sys
from pathlib import Path

from nerdact.detectors import detect_structured_pii
from nerdact.hybrid import resolve_pii_overlaps
from nerdact.pii_model import GLiNER2PIIAdapter, load_pii_schema
from nerdact.redact import redact

ROOT = Path(__file__).resolve().parents[2]
schema = load_pii_schema(ROOT / "data" / "pii-labels.json")
model = GLiNER2PIIAdapter(schema, threshold=0.7)


def redact_text(text: str) -> str:
    findings = model.predict(text) + detect_structured_pii(text)
    resolved = resolve_pii_overlaps(findings)
    return redact(text, resolved.entities)[0]


if __name__ == "__main__":
    input_text = (
        " ".join(sys.argv[1:])
        or "Email Mira Sol at mira@example.test or call +1 (202) 555-0147."
    )
    print(redact_text(input_text))

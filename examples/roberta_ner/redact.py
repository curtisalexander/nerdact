"""Minimal fixed-label NER redaction with the recommended checkpoint."""

import sys

from nerdact.model import HuggingFaceNER
from nerdact.redact import redact

MODEL = "Jean-Baptiste/roberta-large-ner-english"
REVISION = "8f3abc1ef81ffbbb0e80568d4fed1dd10d459548"

ner = HuggingFaceNER(model=MODEL, revision=REVISION, threshold=0.5, stride=64)


def redact_text(text: str) -> str:
    return redact(text, ner.predict(text))[0]


if __name__ == "__main__":
    input_text = " ".join(sys.argv[1:]) or "Mira Sol joined Northstar Bicycles in Portland"
    print(redact_text(input_text))

"""Opt-in tests against real pinned model and Transformers windowing behavior."""

import os

import pytest

from nerdact.cli import _long_context_examples
from nerdact.model import HuggingFaceNER

pytestmark = pytest.mark.integration


@pytest.mark.skipif(
    os.environ.get("NERDACT_RUN_INTEGRATION") != "1",
    reason="set NERDACT_RUN_INTEGRATION=1 after caching the pinned model",
)
def test_real_transformers_pipeline_recovers_entities_across_overlapping_windows(monkeypatch):
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    examples = _long_context_examples()
    adapter = HuggingFaceNER(stride=64)

    for example in examples:
        predictions = adapter.predict(example.text)
        keys = [entity.key() for entity in predictions]
        assert example.entities[0].key() in keys
        assert len(keys) == len(set(keys))
        assert all(entity.text == example.text[entity.start : entity.end] for entity in predictions)

set shell := ["bash", "-eu", "-o", "pipefail", "-c"]

setup:
    uv sync

sync:
    uv sync

demo:
    uv run nerdact demo

report:
    uv run nerdact report

benchmark limit="200":
    uv run --extra benchmark nerdact benchmark --limit {{limit}}

compare limit="200":
    uv run --extra benchmark nerdact compare --limit {{limit}}

compare-modern limit="200":
    uv run --extra benchmark nerdact compare-modern --limit {{limit}}

compare-gliner:
    uv run --extra gliner nerdact compare-gliner

compare-pii:
    uv run --extra pii nerdact compare-pii

summarize:
    uv run nerdact summarize

test:
    uv run pytest

integration-test:
    HF_HUB_OFFLINE=1 NERDACT_RUN_INTEGRATION=1 uv run pytest tests/test_integration.py

lint:
    uv run ruff check .

format:
    uv run ruff format .

typecheck:
    uv run ty check

check: lint typecheck test

clean-artifacts:
    rm -rf artifacts

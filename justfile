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

test:
    uv run pytest

lint:
    uv run ruff check .

format:
    uv run ruff format .

typecheck:
    uv run ty check

check: lint typecheck test

clean-artifacts:
    rm -rf artifacts

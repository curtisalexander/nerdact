"""JSON artifact and self-contained, escaped HTML report generation."""

from __future__ import annotations

import html
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .evaluate import evaluate
from .redact import redact
from .schema import Entity, Example


def _pct(value: float) -> str:
    return f"{value:.1%}"


def _marked(text: str, entities: list[Entity] | tuple[Entity, ...]) -> str:
    output, cursor = [], 0
    for entity in sorted(entities):
        if entity.start < cursor:  # Do not emit malformed markup for overlapping model output.
            continue
        output.append(html.escape(text[cursor : entity.start]))
        title = html.escape(entity.label)
        output.append(
            f'<mark class="{title}" title="{title}">{html.escape(text[entity.start : entity.end])}</mark>'
        )
        cursor = entity.end
    output.append(html.escape(text[cursor:]))
    return "".join(output)


def _redacted_marked(text: str, entities: list[Entity]) -> str:
    """Render placeholders with the same label colors as their source spans."""
    _, replacements = redact(text, entities)
    output, cursor = [], 0
    for entity, placeholder in replacements:
        output.append(html.escape(text[cursor : entity.start]))
        label = html.escape(entity.label)
        output.append(
            f'<mark class="{label}" title="Replaced {label}">{html.escape(placeholder)}</mark>'
        )
        cursor = entity.end
    output.append(html.escape(text[cursor:]))
    return "".join(output)


def build_results(
    examples: list[Example], predictions: list[list[Entity]], model: str, revision: str | None
) -> dict[str, Any]:
    metrics = evaluate(examples, predictions)
    records = []
    for example, predicted in zip(examples, predictions, strict=True):
        redacted, _ = redact(example.text, predicted)
        records.append(
            {
                "id": example.id,
                "text": example.text,
                "note": example.note,
                "gold": [asdict(entity) for entity in example.entities],
                "predictions": [asdict(entity) for entity in predicted],
                "redacted": redacted,
            }
        )
    serializable_metrics = {
        **metrics,
        "errors": [
            {"id": e["id"], "fp": [asdict(x) for x in e["fp"]], "fn": [asdict(x) for x in e["fn"]]}
            for e in metrics["errors"]
        ],
    }
    return {
        "model": model,
        "revision": revision,
        "metrics": serializable_metrics,
        "examples": records,
    }


def write_results(results: dict[str, Any], json_path: Path, html_path: Path) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    html_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    html_path.write_text(render_html(results), encoding="utf-8")


def render_html(results: dict[str, Any]) -> str:
    metrics = results["metrics"]
    micro, chars = metrics["micro"], metrics["characters"]
    rows = "".join(
        f"<tr><th>{html.escape(label)}</th><td>{score['tp']}</td><td>{score['fp']}</td><td>{score['fn']}</td><td>{_pct(score['precision'])}</td><td>{_pct(score['recall'])}</td><td>{_pct(score['f1'])}</td></tr>"
        for label, score in metrics["per_label"].items()
    )
    sections = []
    errors_by_id = {item["id"]: item for item in metrics["errors"]}
    for record in results["examples"]:
        predictions = [Entity(**item) for item in record["predictions"]]
        gold = [Entity(**item) for item in record["gold"]]
        error = errors_by_id[record["id"]]
        pred_list = (
            "".join(
                f"<li><code>[{e.start},{e.end})</code> {html.escape(e.label)} — <q>{html.escape(e.text)}</q> ({e.score:.3f})</li>"
                for e in predictions
            )
            or "<li>None</li>"
        )

        def error_list(items: list[dict[str, Any]]) -> str:
            return (
                "".join(
                    f"<li>{html.escape(x['label'])} <code>[{x['start']},{x['end']})</code> <q>{html.escape(x['text'])}</q></li>"
                    for x in items
                )
                or "<li>None</li>"
            )

        sections.append(f"""<article><h3>{html.escape(record["id"])}</h3><p class="note">{html.escape(record["note"])}</p>
<div class="diff"><div><h4>Before <small>model predictions</small></h4><pre>{_marked(record["text"], predictions)}</pre></div>
<div><h4>After <small>typed placeholders</small></h4><pre>{_redacted_marked(record["text"], predictions)}</pre></div></div>
<details><summary>Compare with gold labels and inspect errors</summary><h4>Human-labeled reference</h4><pre>{_marked(record["text"], gold)}</pre><h4>Predictions</h4><ul>{pred_list}</ul><div class="errors"><div><h4>False positives</h4><ul>{error_list(error["fp"])}</ul></div><div><h4>False negatives</h4><ul>{error_list(error["fn"])}</ul></div></div></details></article>""")
    model = html.escape(str(results["model"]))
    revision = html.escape(str(results["revision"] or "un pinned"))
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>NERdact report</title><style>
:root{{--ink:#172033;--muted:#637083;--paper:#f5f7fb;--card:#fff;--accent:#5b4bdb;--line:#dfe4ee}}*{{box-sizing:border-box}}html{{scroll-behavior:smooth}}body{{margin:0;background:var(--paper);color:var(--ink);font:16px/1.55 system-ui,sans-serif}}header,main{{max-width:1050px;margin:auto;padding:2rem}}header{{padding-bottom:1rem}}h1{{font-size:2.8rem;margin:.2rem 0}}h2{{margin-top:3rem}}.eyebrow{{color:var(--accent);font-weight:750;letter-spacing:.12em}}nav a{{margin-right:1rem;color:var(--accent)}}.warning{{background:#fff4d6;border-left:5px solid #e6a700;padding:1rem}}.cards,.process{{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:1rem;margin:2rem 0}}.card,article,.lesson{{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:1.2rem;box-shadow:0 5px 20px #24314d0a}}.card strong{{font-size:1.8rem;display:block}}.step strong{{display:block;color:var(--accent)}}table{{width:100%;border-collapse:collapse;background:white}}th,td{{padding:.7rem;border-bottom:1px solid var(--line);text-align:right}}th:first-child{{text-align:left}}article{{margin:1.5rem 0}}pre{{white-space:pre-wrap;background:#f8f9fc;padding:1rem;border-radius:8px}}mark{{padding:.08rem .18rem;border-radius:3px}}mark.PERSON{{background:#ffe0eb;border-bottom:2px solid #d63b70}}mark.ORGANIZATION{{background:#dff4ff;border-bottom:2px solid #1682b4}}mark.LOCATION{{background:#e1f6df;border-bottom:2px solid #3a9342}}mark.MISCELLANEOUS{{background:#fff0c9;border-bottom:2px solid #b87800}}small,.note{{color:var(--muted)}}.diff,.errors{{display:grid;grid-template-columns:1fr 1fr;gap:1rem}}details summary{{cursor:pointer;color:var(--accent);font-weight:650}}code{{font-size:.9em}}.reproduce{{background:#172033;color:#f5f7fb}}@media(max-width:700px){{header,main{{padding:1rem}}.diff,.errors{{grid-template-columns:1fr}}table{{font-size:.8rem}}}}
</style></head><body><header><div class="eyebrow">NERDACT · CHARACTER-SPAN EVALUATION</div><h1>Domain report</h1><p>Model <code>{model}</code><br>Revision <code>{revision}</code></p><div class="warning"><strong>Teaching demo, not a privacy system.</strong> This news-trained model only recognizes PERSON, ORGANIZATION, LOCATION, and MISCELLANEOUS. It does not detect addresses, phone numbers, email, or general PII.</div></header><main>
<nav aria-label="Report sections"><a href="#workflow">Workflow</a><a href="#metrics">Metrics</a><a href="#examples">Transcripts</a><a href="#next">Choosing a model</a></nav>
<h2 id="workflow">From transcript to typed placeholders</h2><p>Named-entity recognition is <strong>token classification</strong>: the model assigns labels such as <code>B-PER</code>, <code>I-PER</code>, and <code>O</code> to model tokens. Because one visible word can become several subword tokens, NERdact aggregates those predictions and converts them to stable <code>[start, end)</code> character spans in the original transcript.</p>
<section class="process"><div class="lesson step"><strong>1 · Tokenize</strong>Split text into the model's tokens while retaining source offsets.</div><div class="lesson step"><strong>2 · Classify</strong>Predict a BIO label and confidence for every token.</div><div class="lesson step"><strong>3 · Align</strong>Aggregate subwords into labeled character spans.</div><div class="lesson step"><strong>4 · Redact</strong>Assign typed placeholders in reading order and replace spans right-to-left.</div></section>
<section class="cards"><div class="card"><span>Exact micro F1</span><strong>{_pct(micro["f1"])}</strong></div><div class="card"><span>Precision</span><strong>{_pct(micro["precision"])}</strong></div><div class="card"><span>Recall</span><strong>{_pct(micro["recall"])}</strong></div><div class="card"><span>Character leakage</span><strong>{_pct(chars["leakage_rate"])}</strong><small>{chars["leaked_gold_characters"]} / {chars["gold_entity_characters"]} gold entity chars</small></div><div class="card"><span>Over-redaction</span><strong>{_pct(chars["over_redaction_rate"])}</strong><small>{chars["over_redacted_characters"]} / {chars["non_gold_characters"]} non-gold chars</small></div></section>
<h2 id="metrics">How to read the metrics</h2><div class="lesson"><p>An <strong>exact true positive</strong> has the same start, end, and label as the human annotation. Precision asks how many predictions were right; recall asks how many gold entities were found; F1 balances both. Token accuracy is intentionally not featured because abundant <code>O</code> tokens can hide sensitive-entity misses.</p><p><strong>Character leakage</strong> measures gold entity characters left uncovered. <strong>Over-redaction</strong> measures non-entity characters covered by predictions. These coverage measures reveal partial-boundary behavior that exact F1 alone cannot explain.</p></div>
<h3>Exact <code>[start, end)</code> span metrics</h3><table><thead><tr><th>Label</th><th>TP</th><th>FP</th><th>FN</th><th>Precision</th><th>Recall</th><th>F1</th></tr></thead><tbody>{rows}</tbody></table>
<h2 id="examples">Color-coded before and after</h2><p>The <strong>before</strong> side highlights what the model—not the gold data—identified. The <strong>after</strong> side uses the same colors for its typed placeholders. Open each comparison to see the human labels, confidence scores, false positives, and false negatives.</p>{"".join(sections)}
<h2 id="next">A benchmark is not your domain</h2><div class="lesson"><p>The starter model was trained on CoNLL-2003 Reuters newswire, while these examples resemble call transcripts. Casing, punctuation, names, products, and support jargon differ. Good benchmark performance therefore does not establish safe redaction performance.</p><p>When evaluating another Hugging Face model, inspect its model card, license, training domain, language, entity taxonomy, maximum input length, revision, and behavior on representative labeled transcripts. Names and locations are only a subset of PII; practical systems usually combine domain-trained NER with deterministic detectors and human review.</p></div>
<h2>Reproduce this report</h2><pre class="reproduce"><code>uv sync
uv run nerdact report

# Inspect one string
uv run nerdact redact "Mira Sol called from Portland"

# Optional, terms-restricted CoNLL comparison
uv run --extra benchmark nerdact benchmark --limit 200</code></pre><p><small>This self-contained page contains the original fictional transcripts. A real report containing sensitive input would itself be sensitive data.</small></p></main></body></html>"""

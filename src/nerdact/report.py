"""JSON artifact and self-contained, escaped HTML report generation."""

from __future__ import annotations

import html
import json
from collections.abc import Collection
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .evaluate import evaluate
from .redact import redact
from .schema import LABELS, Entity, Example


def _pct(value: float) -> str:
    return f"{value:.1%}"


def _size(value: int) -> str:
    return f"{value / 1024**2:.0f} MiB"


def _confidence(entity: Entity) -> str:
    return f" ({entity.score:.3f})" if entity.score is not None else ""


def _site_nav() -> str:
    return (
        '<a href="index.html">Start: BERT baseline</a> · '
        '<a href="benchmark.html">Compare NER models</a> · '
        '<a href="modern-encoders.html">Long-context study</a> · '
        '<a href="runtime-labels.html">How GLiNER works</a> · '
        '<a href="practical-pii.html">Evaluate hybrid PII</a> · '
        '<a href="conclusion.html">TL;DR and next steps</a>'
    )


def _run_note(container: dict[str, Any]) -> str:
    provenance = container.get("provenance")
    if not isinstance(provenance, dict) or not provenance.get("run_id"):
        return ""
    dirty = " · uncommitted worktree" if provenance.get("git_worktree_dirty") else ""
    return (
        f'<p><small>Run <code>{html.escape(str(provenance["run_id"]))}</code>{dirty}. '
        "Input, source, dependency, option, and Git fingerprints are recorded in the JSON "
        "artifact.</small></p>"
    )


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
    examples: list[Example],
    predictions: list[list[Entity]],
    model: str,
    revision: str | None,
    labels: Collection[str] = LABELS,
) -> dict[str, Any]:
    metrics = evaluate(examples, predictions, labels)
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
    serialized = json.dumps(results, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    page = render_html(results)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    html_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(serialized, encoding="utf-8")
    html_path.write_text(page, encoding="utf-8")


def write_comparison(
    rows: list[dict[str, Any]],
    html_path: Path,
    split: str,
    limit: int,
    focus: str = "classic",
    provenance: dict[str, Any] | None = None,
    reproduce_command: str | None = None,
) -> None:
    """Write a compact domain-versus-public-benchmark model comparison."""
    table_rows = []
    report_links = []
    checkpoint_records = []
    for row in rows:
        domain = row["domain"]["metrics"]
        benchmark = row["benchmark"]["metrics"]
        performance = row["performance"]
        report_links.append(
            f'<a href="{html.escape(row["report"])}">{html.escape(row["name"])} report</a>'
        )
        if "license" in row:
            checkpoint_records.append(
                f"<li><strong>{html.escape(row['name'])}</strong> — license "
                f"{html.escape(row['license'])}; training data "
                f"{html.escape(row['training_data'])}; labels "
                f"{html.escape(', '.join(row['label_schema']))}; context "
                f"{row['context_length']} tokens; decoder {html.escape(row['decoder'])}.</li>"
            )
        table_rows.append(
            f"<tr><th>{html.escape(row['name'])}<small>{html.escape(row['model'])}<br>{html.escape(row['revision'])}</small></th>"
            f"<td>{html.escape(row['introduced'])}</td><td>{html.escape(row['parameters'])}</td>"
            f"<td>{_pct(domain['micro']['precision'])}</td><td>{_pct(domain['micro']['recall'])}</td><td>{_pct(domain['micro']['f1'])}</td>"
            f"<td>{_pct(domain['characters']['leakage_rate'])}</td><td>{_pct(domain['characters']['over_redaction_rate'])}</td>"
            f"<td>{_pct(benchmark['micro']['precision'])}</td><td>{_pct(benchmark['micro']['recall'])}</td><td>{_pct(benchmark['micro']['f1'])}</td>"
            f"<td>{performance['warm_latency_median_ms']:.1f} ms</td><td>{performance['examples_per_second']:.1f}/s</td>"
            f"<td>{_size(performance['cached_snapshot_bytes'])}</td><td>{_size(performance['peak_memory_bytes'])}<small>{html.escape(performance['peak_memory_kind'])}</small></td></tr>"
        )
    environment = rows[0]["environment"]
    domain_count = len(rows[0]["domain"]["examples"])
    transcript_label = "transcript" if domain_count == 1 else "transcripts"
    threshold = (
        provenance.get("options", {}).get("threshold", 0.5)
        if isinstance(provenance, dict)
        else 0.5
    )
    if focus == "modern":
        baseline_metrics = rows[0]["domain"]["metrics"]
        modern_metrics = rows[1]["domain"]["metrics"]
        baseline_speed = rows[0]["performance"]["examples_per_second"]
        modern_speed = rows[1]["performance"]["examples_per_second"]
        heading = "Long context is a checkpoint property, not a model-name promise"
        comparison_lesson = (
            "BERT base has a 512-token context and therefore uses overlapping 64-token "
            "windows here. ModernBERT supports 8192 tokens and receives each synthetic call "
            "in one pass. The ModernBERT checkpoint was only fine-tuned with sequences up to "
            "256 tokens, however, so architectural capacity does not establish learned "
            "long-context NER quality. The checkpoints also differ in scale, tokenizer, "
            "publisher, and fine-tuning procedure; this is not a controlled architecture test."
        )
        conclusion = (
            f"Chunked BERT recovered {_pct(baseline_metrics['micro']['recall'])} of exact "
            f"long-fixture entities with {_pct(baseline_metrics['characters']['leakage_rate'])} "
            f"character leakage; ModernBERT recovered {_pct(modern_metrics['micro']['recall'])} "
            f"with {_pct(modern_metrics['characters']['leakage_rate'])} leakage. BERT also ran "
            f"at {baseline_speed:.1f} examples/s versus ModernBERT's {modern_speed:.1f}/s. "
            "Here, overlap avoids BERT truncation and is both more accurate and cheaper. This "
            "does not show that separated windows preserve all document context, nor does it "
            "generalize beyond three deliberately simple fixtures."
        )
        warning_heading = "Candidate gate and contamination limits"
        warning = (
            "The accepted ModernBERT checkpoint documents the nine BIO labels, CoNLL-2003 "
            "sample counts, train/validation/test results, max fine-tuning length, "
            "hyperparameters, and Apache-2.0 license. Its dataset revision and source code are "
            "not pinned, so this project pins the model snapshot and independently evaluates "
            f"the CoNLL {html.escape(split)} subset. This is a practical checkpoint result, "
            "not an independently reproducible architecture experiment."
        )
        default_command = "uv run --extra benchmark nerdact compare-modern --limit " + str(limit)
    else:
        heading = "Size, architecture, and fine-tuning all matter"
        comparison_lesson = (
            "The three <code>dslim</code> checkpoints form the cleaner size experiment: one "
            "publisher, one four-label CoNLL task, and DistilBERT/BERT encoders at roughly "
            "66M, 110M, and 340M parameters. Even this does not prove that scale alone caused "
            "every difference because each checkpoint is a separately fine-tuned artifact."
            "</p><p>RoBERTa-large is a practical checkpoint comparison, not a controlled "
            "architecture experiment. It changes model family, tokenizer, scale, and "
            "fine-tuning procedure at once. A strong score on designed examples is "
            "encouraging, not proof of safety."
        )
        conclusion = (
            "DistilBERT is the clear download-size, latency, and throughput choice here, but "
            "it also has the weakest exact-span quality and substantial leakage. BERT base is "
            "the middle-cost compromise. BERT large improves quality at materially higher "
            "cost. At the large-model tier, this RoBERTa checkpoint provides the strongest "
            "practical quality result. None is suitable as a general PII redactor."
        )
        warning_heading = "Data leakage avoided"
        warning = (
            "The RoBERTa model card says its author included the original CoNLL <strong>test"
            f"</strong> split in training. This run therefore uses the held-out <strong>{html.escape(split)}"
            "</strong> split. Dataset access remains subject to the Reuters terms in the README."
        )
        default_command = "uv run --extra benchmark nerdact compare --limit " + str(limit)
    command = reproduce_command or default_command
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>NERdact model benchmark</title><style>
:root{{--ink:#172033;--muted:#637083;--paper:#f5f7fb;--card:#fff;--accent:#5b4bdb;--line:#dfe4ee}}*{{box-sizing:border-box}}body{{margin:0;background:var(--paper);color:var(--ink);font:16px/1.6 system-ui,sans-serif}}main{{max-width:1100px;margin:auto;padding:3rem 2rem}}h1{{font-size:2.7rem;margin:.2rem 0}}.eyebrow,a{{color:var(--accent)}}.eyebrow{{font-weight:750;letter-spacing:.12em}}.card{{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:1.3rem;margin:1.5rem 0}}details>summary{{cursor:pointer;color:var(--accent);font-weight:750;font-size:1.15rem}}details h3{{margin-top:1.6rem}}table{{width:100%;border-collapse:collapse;background:white;font-size:.9rem}}th,td{{padding:.8rem;border-bottom:1px solid var(--line);text-align:right}}th:first-child{{text-align:left}}th small{{display:block;color:var(--muted);font-weight:400}}code{{background:#eceefa;padding:.1rem .35rem;border-radius:4px}}.warning{{border-left:5px solid #e6a700}}@media(max-width:760px){{main{{padding:1.5rem 1rem}}table{{display:block;overflow-x:auto}}}}
</style></head><body><main><div class="eyebrow">NERDACT · MODEL COMPARISON</div><h1>{heading}</h1><nav>{_site_nav()}</nav><p>{" · ".join(report_links)}</p>
<div class="card"><strong>Read the domain columns first.</strong> They show exact-span quality on the same {domain_count} fictional {transcript_label}. Use leakage to see how much annotated text remained exposed. The CoNLL columns are a bounded secondary check, not the standard published token-level benchmark.</div>
<table><thead><tr><th rowspan="2">Checkpoint</th><th rowspan="2">Architecture year</th><th rowspan="2">Approx. parameters</th><th colspan="5">Domain transcripts</th><th colspan="3">Space-joined CoNLL {html.escape(split)} ({limit}) · exact span</th><th colspan="4">Warm inference cost</th></tr><tr><th>Precision</th><th>Recall</th><th>F1</th><th>Leakage</th><th>Over-redaction</th><th>Precision</th><th>Recall</th><th>F1</th><th>Median latency</th><th>Throughput</th><th>Cache snapshot</th><th>Memory measurement</th></tr></thead><tbody>{"".join(table_rows)}</tbody></table>
<details class="card"><summary>Methods, checkpoint records, and test machine</summary><h3>Checkpoint records</h3><ul>{"".join(checkpoint_records)}</ul><p>All rows use a confidence threshold of <code>{threshold}</code>. Revisions appear in the table and the machine-readable artifact.</p><h3>Timing</h3><p>Each checkpoint ran in a fresh process. After one untimed warm-up example, NERdact timed sequential, single-example inference over the {domain_count} domain {transcript_label} for {rows[0]["performance"]["timed_repeats"]} repeats using a monotonic clock and device synchronization. Throughput is examples per second. Memory is peak reserved accelerator memory on CUDA, complete-process peak RSS on CPU, and the maximum post-inference sample of current driver allocation on MPS; transient MPS allocations may be missed. Cache snapshot size sums files present in the pinned model snapshot.</p><p><strong>{html.escape(environment["processor"])}</strong> ({html.escape(environment["machine"])}) · {html.escape(environment["os"])} · device <code>{html.escape(environment["device"])}</code> · Python {html.escape(environment["python"])} · PyTorch {html.escape(environment["torch"])} · Transformers {html.escape(environment["transformers"])}</p><h3>CoNLL method</h3><p>The dataset supplies tokens rather than original whitespace, so NERdact joins each example with spaces and scores exact character spans. This compares checkpoints consistently inside NERdact; it does not reproduce standard token-level CoNLL/seqeval scores.</p></details>
<div class="card"><h2>What this comparison can establish</h2><p>{comparison_lesson}</p></div>
<div class="card"><h2>Quality versus cost conclusion</h2><p>{conclusion}</p></div>
<div class="card warning"><h2>{warning_heading}</h2><p>{warning}</p></div>
<h2>Reproduce</h2><pre><code>{command}</code></pre>{_run_note({"provenance": provenance} if provenance else {})}<p><small>Models are revision-pinned. Generated reports contain the original input text and must be protected when using non-synthetic data.</small></p></main></body></html>"""
    html_path.parent.mkdir(parents=True, exist_ok=True)
    html_path.write_text(page, encoding="utf-8")


def render_html(results: dict[str, Any]) -> str:
    metrics = results["metrics"]
    micro, chars = metrics["micro"], metrics["characters"]
    rows = "".join(
        f"<tr><th>{html.escape(label)}</th><td>{score['tp']}</td><td>{score['fp']}</td><td>{score['fn']}</td><td>{_pct(score['precision'])}</td><td>{_pct(score['recall'])}</td><td>{_pct(score['f1'])}</td></tr>"
        for label, score in metrics["per_label"].items()
    )
    errors_by_id = {item["id"]: item for item in metrics["errors"]}
    records_by_id = {record["id"]: record for record in results["examples"]}

    def render_case(record: dict[str, Any]) -> str:
        predictions = [Entity(**item) for item in record["predictions"]]
        gold = [Entity(**item) for item in record["gold"]]
        error = errors_by_id[record["id"]]
        pred_list = (
            "".join(
                f"<li><code>[{e.start},{e.end})</code> {html.escape(e.label)} — <q>{html.escape(e.text)}</q>{_confidence(e)}</li>"
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

        return f"""<details class="case" id="case-{html.escape(record['id'])}"><summary><strong>{html.escape(record["id"])}</strong><span>{html.escape(record["note"])}</span></summary><div class="case-body">
<div class="diff"><div><h4>Prediction overlay</h4><pre>{_marked(record["text"], predictions)}</pre></div>
<div><h4>Redacted output</h4><pre>{_redacted_marked(record["text"], predictions)}</pre></div></div>
<h4>Human-labeled reference</h4><pre>{_marked(record["text"], gold)}</pre><h4>Model predictions</h4><ul>{pred_list}</ul><div class="errors"><div><h4>False positives</h4><ul>{error_list(error["fp"])}</ul></div><div><h4>False negatives</h4><ul>{error_list(error["fn"])}</ul></div></div></div></details>"""

    sections = "".join(render_case(record) for record in results["examples"])
    guided_lessons = {
        "welcome": "Exact success: each predicted boundary and label matches the reference.",
        "product": "Wrong type: the characters are covered, but Aurora Plus receives the wrong label.",
        "hyphen-apostrophe": "Boundary failure: one hyphenated person becomes two predictions and two placeholders.",
        "lowercase": "Complete miss: no prediction means no replacement and full character leakage.",
    }
    guided = []
    for record_id, lesson in guided_lessons.items():
        record = records_by_id.get(record_id)
        if record is None:
            continue
        predictions = [Entity(**item) for item in record["predictions"]]
        guided.append(
            f"""<article class="guided"><div class="lesson-tag">{html.escape(record_id)}</div><h3>{html.escape(lesson)}</h3><div class="diff"><div><h4>Prediction overlay</h4><pre>{_marked(record["text"], predictions)}</pre></div><div><h4>Redacted output</h4><pre>{_redacted_marked(record["text"], predictions)}</pre></div></div><a href="#case-{html.escape(record_id)}">Inspect predictions and exact errors ↓</a></article>"""
        )
    guided_html = "".join(guided)

    first_prediction = next(
        (
            Entity(**item)
            for record in results["examples"]
            for item in record["predictions"]
        ),
        None,
    )
    span_demo = (
        f'<code>[{first_prediction.start},{first_prediction.end})</code> '
        f'<strong>{html.escape(first_prediction.label)}</strong> '
        f'<q>{html.escape(first_prediction.text)}</q>{_confidence(first_prediction)}'
        if first_prediction is not None
        else "No prediction is available in this run."
    )
    product = records_by_id.get("product")
    if product and product["predictions"] and product["gold"]:
        predicted_product = Entity(**product["predictions"][0])
        gold_product = Entity(**product["gold"][0])
        exact_demo = (
            f'<div><span>Prediction</span><code>[{predicted_product.start},{predicted_product.end}) '
            f'{html.escape(predicted_product.label)} · {html.escape(predicted_product.text)}</code></div>'
            f'<div><span>Reference</span><code>[{gold_product.start},{gold_product.end}) '
            f'{html.escape(gold_product.label)} · {html.escape(gold_product.text)}</code></div>'
        )
        exact_explanation = (
            "Here the characters match but the label does not. Exact scoring counts a "
            "false-positive organization and a false-negative miscellaneous entity, while "
            "character coverage still considers the text covered."
        )
    else:
        exact_demo = "<p>Boundary and label must both match the human reference.</p>"
        exact_explanation = (
            "Exact scoring and character coverage answer different questions: a wrong-label "
            "span can fail exact evaluation while still covering the text."
        )

    model = html.escape(str(results["model"]))
    revision = html.escape(str(results["revision"] or "un pinned"))
    exact_command = html.escape(
        str(results.get("reproduce_command", "uv run nerdact report"))
    )
    if "welcome" in records_by_id:
        reproduce_blocks = f"""<h3>Generate the baseline lesson</h3><pre class="reproduce"><code>uv sync
uv run nerdact report</code></pre><h3>Inspect one fictional string</h3><pre class="reproduce"><code>uv run nerdact redact "Mira Sol called from Portland"</code></pre><details><summary>Recreate this exact generating run</summary><pre class="reproduce"><code>{exact_command}</code></pre></details>"""
    else:
        reproduce_blocks = f"""<h3>Recreate this report</h3><pre class="reproduce"><code>{exact_command}</code></pre>"""
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>NERdact report</title><style>
:root{{--ink:#172033;--muted:#637083;--paper:#f5f7fb;--card:#fff;--accent:#5b4bdb;--line:#dfe4ee;--warning:#fff4d6}}*{{box-sizing:border-box}}html{{scroll-behavior:smooth}}body{{margin:0;background:var(--paper);color:var(--ink);font:16px/1.65 system-ui,sans-serif}}header,main,footer{{max-width:1050px;margin:auto;padding:2rem}}header{{padding-bottom:1rem}}h1{{max-width:850px;font-size:clamp(2.4rem,6vw,4rem);line-height:1.05;margin:.35rem 0 1rem}}h2{{font-size:2rem;line-height:1.2;margin:4rem 0 1rem;scroll-margin-top:1rem}}h3{{line-height:1.3}}a{{color:var(--accent)}}a:focus-visible,summary:focus-visible{{outline:3px solid #9d91ff;outline-offset:3px}}.eyebrow{{color:var(--accent);font-weight:750;letter-spacing:.12em}}.deck{{max-width:780px;font-size:1.2rem;color:#46536a}}nav{{display:flex;flex-wrap:wrap;gap:.45rem 1rem;margin:1rem 0}}.project-nav{{padding-bottom:.75rem;border-bottom:1px solid var(--line)}}.report-nav{{font-size:.95rem}}.section-nav{{position:sticky;top:0;z-index:2;background:#f5f7fbef;padding:.65rem 0;border-bottom:1px solid var(--line);backdrop-filter:blur(8px)}}.warning{{background:var(--warning);border-left:5px solid #e6a700;padding:1rem 1.2rem;margin:1.5rem 0}}.model-focus{{background:#eeecff;border-left:5px solid var(--accent);padding:.9rem 1.1rem;margin:1.25rem 0}}.model-focus code{{overflow-wrap:anywhere}}.run-details{{color:var(--muted);font-size:.92rem}}.run-details code{{overflow-wrap:anywhere}}.process,.cards,.legend,.next-links{{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:1rem;margin:1.5rem 0}}.lesson,.card,.guided,.exact-demo{{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:1.25rem}}.step strong,.lesson-tag{{display:block;color:var(--accent);font-weight:750}}.evaluation-metrics .lesson>strong{{display:block;margin-bottom:.35rem}}.card strong{{display:block;font-size:1.8rem}}.card small{{display:block;color:var(--muted)}}.metric-group{{margin:2rem 0}}.metric-group>h3{{margin-bottom:.3rem}}.legend{{display:flex;flex-wrap:wrap}}.key{{display:inline-flex;align-items:center;gap:.45rem;font-size:.9rem;font-weight:650}}.swatch{{width:.9rem;height:.9rem;border-radius:3px}}.swatch.PERSON{{background:#ffe0eb;border:1px solid #d63b70}}.swatch.ORGANIZATION{{background:#dff4ff;border:1px solid #1682b4}}.swatch.LOCATION{{background:#e1f6df;border:1px solid #3a9342}}.swatch.MISCELLANEOUS{{background:#fff0c9;border:1px solid #b87800}}.span-demo{{font-size:1.1rem;padding:1rem;border-left:4px solid var(--accent);background:#eeecff}}.exact-demo div{{display:grid;grid-template-columns:90px 1fr;gap:1rem;margin:.5rem 0}}.exact-demo span{{font-weight:700}}.exact-demo code{{overflow-wrap:anywhere}}.table-wrap{{overflow-x:auto;border:1px solid var(--line);border-radius:12px}}table{{width:100%;border-collapse:collapse;background:white;min-width:650px}}th,td{{padding:.7rem;border-bottom:1px solid var(--line);text-align:right}}th:first-child{{text-align:left}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;background:#f8f9fc;padding:1rem;border-radius:8px}}mark{{padding:.08rem .18rem;border-radius:3px}}mark.PERSON{{background:#ffe0eb;border-bottom:2px solid #d63b70}}mark.ORGANIZATION{{background:#dff4ff;border-bottom:2px solid #1682b4}}mark.LOCATION{{background:#e1f6df;border-bottom:2px solid #3a9342}}mark.MISCELLANEOUS{{background:#fff0c9;border-bottom:2px solid #b87800}}.diff,.errors{{display:grid;grid-template-columns:1fr 1fr;gap:1rem}}.guided{{margin:1.25rem 0}}.guided h3{{margin:.25rem 0 1rem}}.case{{background:var(--card);border:1px solid var(--line);border-radius:12px;margin:.75rem 0}}.case>summary{{display:flex;flex-direction:column;cursor:pointer;padding:1rem 1.2rem;color:var(--accent)}}.case>summary span{{color:var(--muted);font-weight:400}}.case-body{{padding:0 1.2rem 1.2rem}}.reproduce{{background:#172033;color:#f5f7fb}}.next-links a{{display:block;background:white;border:1px solid var(--line);border-radius:10px;padding:1rem;text-decoration:none;font-weight:700}}footer{{color:var(--muted);border-top:1px solid var(--line);margin-top:3rem}}@media(max-width:700px){{header,main,footer{{padding:1rem}}.section-nav{{position:static}}.diff,.errors{{grid-template-columns:1fr}}h2{{margin-top:3rem}}}}
</style></head><body><header><nav class="project-nav" aria-label="Project"><a href="https://github.com/curtisalexander/nerdact">Repository</a><a href="https://github.com/curtisalexander/nerdact/tree/main/learning">Guided learning</a><a href="https://github.com/curtisalexander/nerdact/tree/main/examples">Examples</a></nav><div class="eyebrow">NERDACT · SPAN-FIRST NER LESSON</div><h1>How token predictions become evaluated redactions</h1><p class="deck">Follow a four-label NER model from named text, through token labels and source-character spans, to measured errors and typed placeholders on fictional call transcripts.</p><div class="model-focus"><strong>Results on this entire page:</strong> <code>{model}</code>. Every score, prediction, and redacted example below comes from this checkpoint.</div><details class="run-details"><summary>Revision and run details</summary><p>Model <code>{model}</code><br>Revision <code>{revision}</code></p>{_run_note(results)}</details><nav class="report-nav" aria-label="Experiment reports">{_site_nav()}</nav><div class="warning"><strong>Teaching demo, not a privacy system.</strong> This report evaluates only PERSON, ORGANIZATION, LOCATION, and MISCELLANEOUS. It does not detect addresses, phone numbers, email, account numbers, or general PII. Typed replacement is pseudonymization, not anonymization.</div></header><main>
<nav class="section-nav" aria-label="Lesson sections"><a href="#ner">1 · NER</a><a href="#spans">2 · Spans</a><a href="#evaluation">3 · Evaluation</a><a href="#results">4 · Results</a><a href="#redaction">5 · Redaction</a><a href="#limits">6 · Limits</a><a href="#reproduce">7 · Reproduce</a></nav>
<h2 id="ner">1. NER assigns labels to pieces of text</h2><p><strong>Named-entity recognition (NER)</strong> finds named people, organizations, locations, and other names. A token-classification model does this indirectly: it labels model tokens. In BIO notation, <code>B-PER</code> begins a person, <code>I-PER</code> continues that person, and <code>O</code> means “outside an entity.”</p><p>One visible word can split into several model tokens, so token labels are not yet safe replacement instructions.</p><section class="process"><div class="lesson step"><strong>Text</strong>The original transcript.</div><div class="lesson step"><strong>Token labels</strong>Conceptual BIO predictions such as <code>B-PER</code> and <code>I-PER</code>.</div><div class="lesson step"><strong>Entity spans</strong>Aggregated labels mapped back to exact source offsets.</div></section>
<h2 id="spans">2. Character spans are the shared contract</h2><p>NERdact converts model output to <code>(start, end, label, text, score)</code>. The half-open range <code>[start, end)</code> includes <code>start</code> and excludes <code>end</code> in the original Python string.</p><div class="span-demo">{span_demo}</div><p>Evaluation and redaction now use the same representation. Boundaries are inspectable, and replacements target the exact source characters rather than every matching word.</p><div class="legend" aria-label="Entity label colors"><span class="key"><span class="swatch PERSON"></span>PERSON</span><span class="key"><span class="swatch ORGANIZATION"></span>ORGANIZATION</span><span class="key"><span class="swatch LOCATION"></span>LOCATION</span><span class="key"><span class="swatch MISCELLANEOUS"></span>MISCELLANEOUS</span></div>
<h2 id="evaluation">3. Exact evaluation checks boundary and label</h2><p>A prediction is correct only when its <strong>start, end, and label</strong> match the human annotation. A partial name is an error. The right characters with the wrong type are also an error.</p><div class="exact-demo">{exact_demo}</div><p>{exact_explanation}</p><div class="process evaluation-metrics"><div class="lesson"><strong>Precision</strong>Of predicted entities, how many were exact?</div><div class="lesson"><strong>Recall</strong>Of human-labeled entities, how many were recovered exactly?</div><div class="lesson"><strong>F1</strong>The balance of exact precision and recall.</div><div class="lesson"><strong>Character leakage</strong>Annotated entity characters left uncovered by every prediction.</div><div class="lesson"><strong>Over-redaction</strong>Non-entity characters covered by predictions.</div></div><p><small>Token accuracy is intentionally omitted because abundant <code>O</code> tokens can hide entity misses.</small></p>
<h2 id="results">4. Results on the fictional transcripts</h2><p>Exact scores judge whole labeled spans; coverage scores judge characters. Read both: a wrong-label span can fail exact evaluation while covering the text, and a partial span can leak only part of a name.</p><section class="metric-group"><h3>Exact entity matching</h3><div class="cards"><div class="card"><span>Recall</span><strong>{_pct(micro["recall"])}</strong></div><div class="card"><span>Precision</span><strong>{_pct(micro["precision"])}</strong></div><div class="card"><span>Micro F1</span><strong>{_pct(micro["f1"])}</strong></div></div></section><section class="metric-group"><h3>Character coverage</h3><div class="cards"><div class="card"><span>Leakage</span><strong>{_pct(chars["leakage_rate"])}</strong><small>{chars["leaked_gold_characters"]} / {chars["gold_entity_characters"]} annotated entity characters</small></div><div class="card"><span>Over-redaction</span><strong>{_pct(chars["over_redaction_rate"])}</strong><small>{chars["over_redacted_characters"]} / {chars["non_gold_characters"]} non-entity characters</small></div></div></section><h3>Exact results by label</h3><div class="table-wrap"><table><thead><tr><th>Label</th><th>True positives</th><th>False positives</th><th>False negatives</th><th>Precision</th><th>Recall</th><th>F1</th></tr></thead><tbody>{rows}</tbody></table></div>
<h2 id="redaction">5. Redaction follows predictions—not the human labels</h2><p>NERdact assigns typed placeholders in reading order and replaces the predicted spans. It does not consult the reference annotation during redaction. Every miss, wrong type, or bad boundary can therefore flow into the output.</p>{guided_html}<h3>Inspect all {len(results["examples"])} transcript cases</h3><p>The complete audit includes successes, hard negatives, casing changes, punctuation boundaries, invented names, and support jargon. Open any case to compare predictions with the human-labeled reference.</p>{sections}
<h2 id="limits">6. What this report cannot guarantee</h2><div class="lesson"><h3>NER is narrower than PII detection</h3><p>The four labels in this report do not cover addresses, phone numbers, email, account numbers, or all sensitive information.</p><h3>Training domain matters</h3><p>The default checkpoint was trained on CoNLL-2003 Reuters newswire. A replacement model must be checked against its own training domain; call language, casing, punctuation, products, and support jargon may differ.</p><h3>Redaction is not anonymization</h3><p>Missed text remains visible, false positives remove useful text, and surrounding context can still identify someone. Reused placeholders can reveal repetition.</p><h3>Scores are evidence, not guarantees</h3><p>These designed examples expose failure modes; they do not estimate production safety. Sensitive work requires representative evaluation, broader detection, access and retention controls, and human review.</p></div><div class="next-links"><a href="benchmark.html">Compare fixed-label checkpoints →</a><a href="runtime-labels.html">Explore runtime-selected labels →</a><a href="practical-pii.html">Study the broader PII hybrid →</a></div>
<h2 id="reproduce">7. Reproduce and continue</h2>{reproduce_blocks}<p>Continue with the <a href="https://github.com/curtisalexander/nerdact/tree/main/learning">guided curriculum and full reproduction order</a>.</p></main><footer><strong>Artifact reminder:</strong> this self-contained page includes the original fictional transcripts. A report generated from sensitive input would itself be sensitive.</footer></body></html>"""


def write_runtime_comparison(artifact: dict[str, Any], html_path: Path) -> None:
    """Render the runtime-label wording and threshold experiment."""
    runs = artifact["runs"]
    fixed = artifact["fixed"]["metrics"]
    threshold_rows = "".join(
        f"<tr><th>{html.escape(run['wording'])}</th><td>{run['threshold']:.2f}</td>"
        f"<td>{_pct(run['results']['metrics']['micro']['precision'])}</td>"
        f"<td>{_pct(run['results']['metrics']['micro']['recall'])}</td>"
        f"<td>{_pct(run['results']['metrics']['micro']['f1'])}</td>"
        f"<td>{_pct(run['results']['metrics']['characters']['leakage_rate'])}</td>"
        f"<td>{_pct(run['results']['metrics']['characters']['over_redaction_rate'])}</td></tr>"
        for run in runs
    )
    operating_runs = [run for run in runs if run["threshold"] == artifact["operating_threshold"]]
    comparison_rows = (
        f"<tr><th>Fixed BIO baseline</th><td>{_pct(fixed['micro']['precision'])}</td>"
        f"<td>{_pct(fixed['micro']['recall'])}</td><td>{_pct(fixed['micro']['f1'])}</td>"
        f"<td>{_pct(fixed['characters']['leakage_rate'])}</td></tr>"
        + "".join(
            f"<tr><th>GLiNER · {html.escape(run['wording'])}</th>"
            f"<td>{_pct(run['results']['metrics']['micro']['precision'])}</td>"
            f"<td>{_pct(run['results']['metrics']['micro']['recall'])}</td>"
            f"<td>{_pct(run['results']['metrics']['micro']['f1'])}</td>"
            f"<td>{_pct(run['results']['metrics']['characters']['leakage_rate'])}</td></tr>"
            for run in operating_runs
        )
    )
    concise = next(run for run in operating_runs if run["wording"] == "concise")["results"]
    descriptive = next(run for run in operating_runs if run["wording"] == "descriptive")["results"]
    wording_examples = []
    for short, long in zip(concise["examples"], descriptive["examples"], strict=True):
        short_keys = {(item["start"], item["end"], item["label"]) for item in short["predictions"]}
        long_keys = {(item["start"], item["end"], item["label"]) for item in long["predictions"]}
        if short_keys != long_keys:
            wording_examples.append(
                f"<li><strong>{html.escape(short['id'])}</strong> — concise: "
                f"{html.escape(', '.join(item['text'] + ' → ' + item['label'] for item in short['predictions']) or 'none')}; "
                f"descriptive: {html.escape(', '.join(item['text'] + ' → ' + item['label'] for item in long['predictions']) or 'none')}.</li>"
            )
    diagnostic_items = "".join(
        f"<li><strong>{html.escape(item['id'])}</strong> — flat decoder: {item['flat_count']} spans; "
        f"nested/multi-label decoder: {item['nested_count']} spans; overlap conflicts: "
        f"{item['overlap_count']}.</li>"
        for item in artifact["overlap_diagnostics"]
    )
    schema_items = "".join(
        f"<li><code>{html.escape(item['type'])}</code> → placeholder "
        f"<code>[{html.escape(item['placeholder'])}_n]</code>; concise "
        f"<q>{html.escape(item['concise'])}</q>; descriptive "
        f"<q>{html.escape(item['descriptive'])}</q>.</li>"
        for item in artifact["schema"]["labels"]
    )
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>NERdact runtime-selected labels</title><style>
:root{{--ink:#172033;--muted:#637083;--paper:#f5f7fb;--card:#fff;--accent:#5b4bdb;--line:#dfe4ee}}*{{box-sizing:border-box}}body{{margin:0;background:var(--paper);color:var(--ink);font:16px/1.6 system-ui,sans-serif}}main{{max-width:1050px;margin:auto;padding:3rem 2rem}}h1{{font-size:2.7rem;line-height:1.1}}.eyebrow,a{{color:var(--accent)}}.eyebrow{{font-weight:750;letter-spacing:.12em}}.card{{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:1.3rem;margin:1.5rem 0}}table{{width:100%;border-collapse:collapse;background:white}}th,td{{padding:.7rem;border-bottom:1px solid var(--line);text-align:right}}th:first-child{{text-align:left}}code{{background:#eceefa;padding:.1rem .35rem;border-radius:4px}}.warning{{border-left:5px solid #e6a700}}@media(max-width:700px){{main{{padding:1.5rem 1rem}}table{{display:block;overflow-x:auto}}}}
</style></head><body><main><div class="eyebrow">NERDACT · PHASE 3</div><h1>Labels became inference inputs</h1><nav>{_site_nav()}</nav><p>GLiNER <code>{html.escape(artifact["model"])}</code> at revision <code>{html.escape(artifact["revision"])}</code></p>{_run_note(artifact)}
<div class="card"><h2>BIO classification versus label-conditioned spans</h2><p>The fixed baseline assigns one of its trained BIO tags to every token, then aggregates tokens. GLiNER instead encodes the requested label text with the transcript, scores candidate start/end spans against those labels, applies a threshold, and greedily resolves conflicts. The output is normalized to the same exact <code>[start,end)</code> contract, but changing a label description can change both boundaries and classes without changing model weights.</p></div>
<h2>Shared-label comparison at {artifact["operating_threshold"]:.2f}</h2><table><thead><tr><th>Decoder</th><th>Precision</th><th>Recall</th><th>F1</th><th>Leakage</th></tr></thead><tbody>{comparison_rows}</tbody></table><p>All rows use the same {artifact["example_count"]} fictional transcripts and only PERSON, ORGANIZATION, LOCATION, and MISCELLANEOUS. This is a compatible-label comparison, not a claim that the models share training data or architecture.</p>
<h2>Threshold and wording sweep</h2><table><thead><tr><th>Label wording</th><th>Threshold</th><th>Precision</th><th>Recall</th><th>F1</th><th>Leakage</th><th>Over-redaction</th></tr></thead><tbody>{threshold_rows}</tbody></table><p>Leakage is uncovered gold-character coverage regardless of predicted label; wrong-label coverage still fails exact-span scoring.</p><div class="card warning"><strong>Not an independent calibration set.</strong> These curves describe the same small designed corpus used for evaluation. Selecting its best threshold and reporting that score as generalization would be optimistic; a deployment threshold needs separate, representative calibration data.</div>
<h2>Wording instability at {artifact["operating_threshold"]:.2f}</h2><ul>{"".join(wording_examples) or "<li>No predictions changed at this operating point; inspect other thresholds in the JSON artifact.</li>"}</ul>
<h2>Overlapping and competing candidates</h2><p>GLiNER thresholds candidates before score-ordered conflict resolution. <code>flat_ner=true</code> suppresses every overlap; nested mode permits containment, and <code>multi_label=true</code> permits competing labels on identical boundaries. Overlapping output cannot be passed directly to deterministic redaction, which intentionally rejects overlaps.</p><ul>{diagnostic_items}</ul>
<div class="card"><h2>Checked-in schema</h2><ul>{schema_items}</ul></div>
<div class="card warning"><h2>What this cannot establish</h2><p>The checkpoint card identifies Apache-2.0 weights and the Pile-Mistral dataset, but does not provide a contamination audit for these fictional names. Its 384-word limit and 12-word maximum span width remain model constraints. Runtime labels add flexibility, not guaranteed coverage or calibrated confidence.</p></div>
<h2>Reproduce</h2><pre><code>uv sync --extra gliner
{html.escape(str(artifact.get("reproduce_command", "uv run --extra gliner nerdact compare-gliner")))}</code></pre><p><small>The JSON artifact contains every prediction and metric. Reports are sensitive when their inputs are sensitive.</small></p></main></body></html>"""
    html_path.parent.mkdir(parents=True, exist_ok=True)
    html_path.write_text(page, encoding="utf-8")


def write_pii_comparison(artifact: dict[str, Any], html_path: Path) -> None:
    """Render practical-PII calibration, held-out metrics, and operational limits."""
    calibration_count = len(artifact["calibration_runs"][0]["systems"]["hybrid"]["examples"])
    evaluation_count = len(artifact["evaluation"]["hybrid"]["examples"])
    calibration_rows = []
    for run in artifact["calibration_runs"]:
        metrics = run["systems"]["hybrid"]["metrics"]
        selected = (
            " <strong>selected</strong>"
            if run["threshold"] == artifact["operating_threshold"]
            else ""
        )
        calibration_rows.append(
            f"<tr><th>{run['threshold']:.2f}{selected}</th>"
            f"<td>{_pct(metrics['micro']['precision'])}</td>"
            f"<td>{_pct(metrics['micro']['recall'])}</td>"
            f"<td>{_pct(metrics['micro']['f1'])}</td>"
            f"<td>{_pct(metrics['characters']['leakage_rate'])}</td>"
            f"<td>{_pct(metrics['characters']['over_redaction_rate'])}</td>"
            f"<td>{_pct(metrics['transcripts']['any_leak_rate'])}</td></tr>"
        )
    systems = artifact["evaluation"]
    system_rows = []
    for key, name in (
        ("model", "Contextual model only"),
        ("rules", "Deterministic rules only"),
        ("hybrid", "Resolved hybrid"),
    ):
        metrics = systems[key]["metrics"]
        system_rows.append(
            f"<tr><th>{name}</th><td>{_pct(metrics['micro']['precision'])}</td>"
            f"<td>{_pct(metrics['micro']['recall'])}</td>"
            f"<td>{_pct(metrics['micro']['f1'])}</td>"
            f"<td>{_pct(metrics['characters']['leakage_rate'])}</td>"
            f"<td>{_pct(metrics['characters']['over_redaction_rate'])}</td>"
            f"<td>{_pct(metrics['transcripts']['any_leak_rate'])}</td></tr>"
        )
    hybrid = systems["hybrid"]
    metrics = hybrid["metrics"]
    label_rows = "".join(
        f"<tr><th>{html.escape(label)}</th><td>{score['tp']}</td><td>{score['fp']}</td>"
        f"<td>{score['fn']}</td><td>{_pct(score['precision'])}</td>"
        f"<td>{_pct(score['recall'])}</td><td>{_pct(score['f1'])}</td></tr>"
        for label, score in metrics["per_label"].items()
    )
    errors = {item["id"]: item for item in metrics["errors"]}
    rejected = {item["id"]: item["entities"] for item in systems["rejected_conflicts"]}
    examples = []
    for record in hybrid["examples"]:
        predictions = [Entity(**item) for item in record["predictions"]]
        provenance = (
            "".join(
                f"<li><code>{html.escape(item['label'])}</code> "
                f"<q>{html.escape(item['text'])}</q> — "
                f"{html.escape(', '.join(item['provenance']))}</li>"
                for item in record["predictions"]
            )
            or "<li>No accepted findings.</li>"
        )
        missed = (
            "".join(
                f"<li><code>{html.escape(item['label'])}</code> "
                f"<q>{html.escape(item['text'])}</q></li>"
                for item in errors[record["id"]]["fn"]
            )
            or "<li>None.</li>"
        )
        false_positives = (
            "".join(
                f"<li><code>{html.escape(item['label'])}</code> "
                f"<q>{html.escape(item['text'])}</q></li>"
                for item in errors[record["id"]]["fp"]
            )
            or "<li>None.</li>"
        )
        examples.append(
            f"<article><h3>{html.escape(record['id'])}</h3>"
            f"<p>{html.escape(record['note'])}</p><h4>Original with accepted findings</h4>"
            f"<pre>{_marked(record['text'], predictions)}</pre><h4>Redacted demonstration</h4>"
            f"<pre>{_redacted_marked(record['text'], predictions)}</pre>"
            f"<details><summary>Audit provenance, misses, and conflicts</summary>"
            f"<h4>Accepted provenance</h4><ul>{provenance}</ul>"
            f"<h4>Exact false positives</h4><ul>{false_positives}</ul>"
            f"<h4>Exact false negatives</h4><ul>{missed}</ul>"
            f"<p>Rejected overlapping candidates: {len(rejected[record['id']])}.</p>"
            f"</details></article>"
        )
    chars = metrics["characters"]
    transcripts = metrics["transcripts"]
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>NERdact practical PII redaction</title><style>
:root{{--ink:#172033;--muted:#637083;--paper:#f5f7fb;--card:#fff;--accent:#5b4bdb;--line:#dfe4ee;--warn:#a65a00}}*{{box-sizing:border-box}}body{{margin:0;background:var(--paper);color:var(--ink);font:16px/1.6 system-ui,sans-serif}}main{{max-width:1050px;margin:auto;padding:3rem 2rem}}h1{{font-size:2.7rem;line-height:1.1}}h2{{margin-top:2.8rem}}.eyebrow,a{{color:var(--accent)}}.eyebrow{{font-weight:750;letter-spacing:.12em}}.card,article{{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:1.3rem;margin:1.5rem 0}}.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:1rem}}.cards strong{{display:block;font-size:1.7rem}}.warning{{border-left:5px solid #e6a700}}table{{width:100%;border-collapse:collapse;background:white}}th,td{{padding:.7rem;border-bottom:1px solid var(--line);text-align:right}}th:first-child{{text-align:left}}pre{{white-space:pre-wrap;background:#f8f9fc;padding:1rem;border-radius:8px}}mark{{background:#ffe0eb;border-bottom:2px solid #d63b70;padding:.08rem .18rem;border-radius:3px}}code{{background:#eceefa;padding:.1rem .35rem;border-radius:4px}}summary{{cursor:pointer;color:var(--accent);font-weight:650}}small{{color:var(--muted)}}@media(max-width:700px){{main{{padding:1.5rem 1rem}}table{{display:block;overflow-x:auto}}}}
</style></head><body><main><div class="eyebrow">NERDACT · PHASE 4</div><h1>Hybrid detection reduces—but does not eliminate—PII leakage</h1><nav>{_site_nav()}</nav><p>Contextual checkpoint <code>{html.escape(artifact["model"])}</code><br>Revision <code>{html.escape(artifact["revision"])}</code> · Apache-2.0</p>{_run_note(artifact)}
<div class="card warning"><strong>Fictional demonstration, not a privacy guarantee.</strong> The {evaluation_count} evaluation transcripts are designed challenge cases. Their measured error rate is not an estimate for private calls, unseen languages, or production traffic.</div>
<h2>What makes this system “hybrid”?</h2><div class="card"><p><strong>It combines two different ways to find PII.</strong> The GLiNER2 contextual model finds values whose meaning depends on language, such as a person, address, username, or date. Deterministic detectors validate recognizable structures, such as an email address, URL, IP address, phone number, payment card, or API-key prefix. An overlap resolver then produces one flat set of spans for redaction.</p><p><code>contextual model findings + structural rule findings → overlap resolver → typed redaction</code></p><p>Rules improve precision on formats they understand; the model covers values that cannot be recognized by shape alone. Neither half is sufficient by itself, and the combined system still misses unsupported, unusual, or ambiguous values.</p></div>
<h2>Calibrate before evaluating</h2><p>The model threshold was selected only on {calibration_count} calibration transcripts. The fixed recall-oriented policy is: {html.escape(artifact["selection_policy"])}. The held-out evaluation split was then run once at <code>{artifact["operating_threshold"]:.2f}</code>.</p>
<table><thead><tr><th>Threshold</th><th>Precision</th><th>Recall</th><th>F1</th><th>Leakage</th><th>Over-redaction</th><th>Any-leak transcripts</th></tr></thead><tbody>{"".join(calibration_rows)}</tbody></table>
<h2>Held-out system comparison</h2><table><thead><tr><th>System</th><th>Precision</th><th>Recall</th><th>F1</th><th>Leakage</th><th>Over-redaction</th><th>Any-leak transcripts</th></tr></thead><tbody>{"".join(system_rows)}</tbody></table><p>The model and rules rows are ablations on the same evaluation examples. Leakage is uncovered gold-character coverage regardless of predicted label; wrong-label coverage still fails exact-span scoring. The hybrid merges exact duplicates and provenance, gives verifiable structures precedence in conflicts, encloses URL internals, and otherwise uses model confidence and span length. Rejected conflicts remain in the JSON artifact.</p>
<section class="cards"><div class="card"><span>Hybrid exact F1</span><strong>{_pct(metrics["micro"]["f1"])}</strong></div><div class="card"><span>Character leakage</span><strong>{_pct(chars["leakage_rate"])}</strong><small>{chars["leaked_gold_characters"]} / {chars["gold_entity_characters"]} gold characters</small></div><div class="card"><span>Over-redaction</span><strong>{_pct(chars["over_redaction_rate"])}</strong></div><div class="card"><span>Transcripts with any leak</span><strong>{transcripts["with_any_leak"]} / {transcripts["count"]}</strong></div></section>
<h2>Hybrid exact-span results by label</h2><table><thead><tr><th>Label</th><th>TP</th><th>FP</th><th>FN</th><th>Precision</th><th>Recall</th><th>F1</th></tr></thead><tbody>{label_rows}</tbody></table>
<h2>Threat model and operating procedure</h2><div class="card"><p><strong>Goal:</strong> replace the twelve annotated direct-identifier types before a transcript is shared with a less-trusted reviewer. <strong>Out of scope:</strong> inference from unmasked context, linkage attacks, voice/audio identity, unsupported labels, and maliciously crafted input. Typed masking is pseudonymization, not anonymization; remaining context can identify a caller.</p><p>Inference runs locally, but first use may download model code and weights. For private evaluation, pre-cache and verify the pinned snapshot, disable network access, prevent transcript logging and shell-history capture, use encrypted access-controlled storage, and delete source text, redacted derivatives, JSON, and HTML on a documented schedule. Publish only aggregates that cannot expose rare values.</p><p>Human review is mandatory at the recall-oriented threshold. Review every transcript against the annotation guide, prioritize misses over false positives, and route uncertain spans for correction. Do not silently tune on the held-out report; create a new representative test split after changing policy.</p></div>
<h2>Fictional hybrid demonstration</h2>{"".join(examples)}
<h2>Reproduce</h2><pre><code>uv sync --extra pii
{html.escape(str(artifact.get("reproduce_command", "uv run --extra pii nerdact compare-pii")))}</code></pre><p><small>The generated JSON and HTML retain original inputs and are sensitive when inputs are sensitive. Only the checked-in fictional report is suitable for GitHub Pages.</small></p></main></body></html>"""
    html_path.parent.mkdir(parents=True, exist_ok=True)
    html_path.write_text(page, encoding="utf-8")


def write_learning_summary(
    benchmark: dict[str, Any], pii: dict[str, Any], html_path: Path
) -> None:
    """Render the concise conclusion and implementation handoff for the learning path."""
    roberta = next(row for row in benchmark["models"] if row["report"] == "roberta-large.html")
    roberta_metrics = roberta["domain"]["metrics"]
    roberta_examples = {item["id"]: item for item in roberta["domain"]["examples"]}
    pii_systems = pii["evaluation"]
    model_metrics = pii_systems["model"]["metrics"]
    hybrid_metrics = pii_systems["hybrid"]["metrics"]
    hybrid_examples = {item["id"]: item for item in pii_systems["hybrid"]["examples"]}

    lowercase = html.escape(roberta_examples["lowercase"]["text"])
    slash_boundary = html.escape(roberta_examples["boundary-slashes"]["text"])
    contact = html.escape(hybrid_examples["pii-eval-contact"]["text"])
    unicode_address = html.escape(hybrid_examples["pii-eval-unicode-address"]["text"])
    passphrase = html.escape(hybrid_examples["pii-eval-password-spaces"]["text"])
    roberta_chars = roberta_metrics["characters"]
    model_chars = model_metrics["characters"]
    hybrid_chars = hybrid_metrics["characters"]

    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>NERdact conclusions and next steps</title><style>
:root{{--ink:#172033;--muted:#637083;--paper:#f5f7fb;--card:#fff;--accent:#5b4bdb;--line:#dfe4ee;--good:#e8f7ec;--warn:#fff4d6}}*{{box-sizing:border-box}}html{{scroll-behavior:smooth}}body{{margin:0;background:var(--paper);color:var(--ink);font:17px/1.65 system-ui,sans-serif}}main,footer{{max-width:1000px;margin:auto;padding:2.5rem 2rem}}h1{{font-size:clamp(2.5rem,6vw,4rem);line-height:1.05;margin:.35rem 0 1rem;max-width:900px}}h2{{font-size:2rem;line-height:1.2;margin:4rem 0 1rem}}h3{{line-height:1.3;margin-top:0}}a,.eyebrow{{color:var(--accent)}}.eyebrow{{font-weight:750;letter-spacing:.12em}}.deck{{font-size:1.22rem;max-width:820px;color:#46536a}}nav{{display:flex;flex-wrap:wrap;gap:.45rem 1rem;margin:1.2rem 0 2rem}}.grid,.metrics,.path{{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:1rem;margin:1.25rem 0}}.card,.example,.callout,.definition{{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:1.3rem}}.choice{{border-top:5px solid var(--accent)}}.metrics .card strong{{display:block;font-size:1.8rem}}.metrics small,.muted{{color:var(--muted)}}.callout{{border-left:5px solid #e6a700;background:var(--warn)}}.definition{{border-left:5px solid var(--accent);background:#eeecff;margin:1.25rem 0}}.good{{background:var(--good)}}ul{{padding-left:1.2rem}}li+li{{margin-top:.45rem}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;background:#f8f9fc;border-radius:8px;padding:1rem}}code{{background:#eceefa;padding:.1rem .35rem;border-radius:4px}}.flow{{font-weight:700;text-align:center;padding:.8rem;background:#eeecff;border-radius:8px}}.actions a{{display:block;text-decoration:none;font-weight:700;margin:.55rem 0}}footer{{border-top:1px solid var(--line);color:var(--muted)}}@media(max-width:700px){{main,footer{{padding:1.5rem 1rem}}h2{{margin-top:3rem}}}}
</style></head><body><main><div class="eyebrow">NERDACT · CONCLUSION</div><h1>Choose the narrow model or the broader system</h1><p class="deck">The experiments lead to two useful implementations. Use RoBERTa when four-label named-entity recognition is the task. Use the GLiNER2 hybrid when broader PII coverage matters—and understand that “hybrid” describes the whole pipeline, not the model alone.</p><nav>{_site_nav()}</nav>
<section class="grid"><article class="card choice"><h2>Fixed-label NER</h2><h3>Choose RoBERTa</h3><p>Best measured checkpoint here for <strong>person, organization, location, and miscellaneous names</strong>. It is narrow, predictable, and does not pretend to cover general PII.</p><p><a href="#roberta">Understand RoBERTa ↓</a></p></article><article class="card choice"><h2>Broader PII</h2><h3>Choose the GLiNER2 hybrid</h3><p>Combines contextual span detection with validated structural rules for twelve PII types. Broader coverage brings more policy, calibration, and review work.</p><p><a href="#hybrid">Understand the hybrid ↓</a></p></article></section>
<div class="callout"><strong>Do not compare the two headline F1 scores as a race.</strong> RoBERTa was tested on 20 four-label NER transcripts. The hybrid was tested on a separate 16-transcript, twelve-label PII split. Different taxonomies and fixtures answer different questions.</div>
<h2>The learning path, reorganized</h2><section class="path"><div class="card"><strong>1 · Learn the contract</strong><p><a href="index.html">BERT-base baseline</a></p><p class="muted">The whole page uses <code>dslim/bert-base-NER</code>—not DistilBERT. Learn spans, exact scoring, leakage, and safe replacement here.</p></div><div class="card"><strong>2 · Choose fixed-label NER</strong><p><a href="benchmark.html">NER checkpoint comparison</a></p><p class="muted">The table supports one practical outcome: RoBERTa was strongest on this small comparison.</p></div><div class="card"><strong>Optional deep dives</strong><p><a href="modern-encoders.html">Long context</a> · <a href="runtime-labels.html">How GLiNER works</a></p><p class="muted">Read these for architecture and inference concepts; skip them if you only need an implementation.</p></div><div class="card"><strong>3 · Choose broader PII</strong><p><a href="practical-pii.html">Hybrid PII evaluation</a></p><p class="muted">See calibration, ablations, per-label results, and every held-out example.</p></div></section>
<section id="roberta"><h2>RoBERTa NER: strong, narrow, and still fallible</h2><div class="definition"><p><strong>What it is.</strong> <code>{html.escape(roberta["model"])}</code> is a RoBERTa-large encoder fine-tuned as a fixed token classifier. Its trained BIO tags become four entity types: PERSON, ORGANIZATION, LOCATION, and MISCELLANEOUS. You cannot ask it for email or phone labels at runtime.</p></div><section class="metrics"><div class="card"><span>Exact-span F1</span><strong>{_pct(roberta_metrics["micro"]["f1"])}</strong><small>42 of 47 entities exactly recovered</small></div><div class="card"><span>Character leakage</span><strong>{_pct(roberta_chars["leakage_rate"])}</strong><small>{roberta_chars["leaked_gold_characters"]} of {roberta_chars["gold_entity_characters"]} entity characters uncovered</small></div><div class="card"><span>Exact precision</span><strong>{_pct(roberta_metrics["micro"]["precision"])}</strong><small>On 20 fictional call transcripts</small></div></section><div class="grid"><article class="card good"><h3>Where it was strong</h3><ul><li>Recovered lowercase names, organizations, and places that the BERT-base baseline missed.</li><li>Handled the designed Unicode, uncommon-name, hyphen/apostrophe, and quoted-punctuation cases exactly.</li><li>Produced far less character leakage than every other fixed-label checkpoint tested.</li></ul><div class="example"><strong>Lowercase success</strong><pre>{lowercase}</pre><span>All three entities were exact.</span></div></article><article class="card"><h3>Where it fell down</h3><ul><li>Attached punctuation to <q>Imani Ko/</q> and labeled <q>Red Fern</q> as a location in the slash/em-dash case.</li><li>Misclassified erratically cased organization and location text.</li><li>Over-redacted <q>West</q> as a location in a hard negative.</li><li>Still cannot detect broader PII such as email, phone, credentials, or account numbers.</li></ul><div class="example"><strong>Boundary and label failure</strong><pre>{slash_boundary}</pre><span>The slash changed the person boundary; Red Fern received the wrong label.</span></div></article></div><p class="muted">The model card says the original CoNLL test split was included in training, so NERdact used validation for its bounded public-data comparison. These designed results do not establish production safety.</p></section>
<section id="hybrid"><h2>GLiNER2 hybrid PII: broader because two detectors cooperate</h2><div class="definition"><p><strong>The GLiNER2 model is only one half.</strong> <code>{html.escape(pii["model"])}</code> accepts label descriptions at inference and finds contextual spans. NERdact adds deterministic structural detectors, then resolves overlaps into one auditable redaction set. That complete contextual-plus-rule system is the <strong>hybrid</strong>.</p><div class="flow">GLiNER2 contextual findings + validated structural rules → overlap resolver → typed redaction</div></div><div class="grid"><div class="card"><h3>Contextual model</h3><p>Useful when shape is insufficient: people, postal addresses, usernames, dates, and natural-language credentials.</p></div><div class="card"><h3>Structural rules</h3><p>Useful when format is evidence: email, URL, IP, phone, Luhn-valid cards, routing checksums, IDs, and API-key prefixes.</p></div><div class="card"><h3>Resolver</h3><p>Merges duplicates, gives validated structures precedence in conflicts, and retains rejected candidates for audit.</p></div></div><section class="metrics"><div class="card"><span>Model-only F1</span><strong>{_pct(model_metrics["micro"]["f1"])}</strong><small>{_pct(model_chars["leakage_rate"])} character leakage</small></div><div class="card"><span>Hybrid F1</span><strong>{_pct(hybrid_metrics["micro"]["f1"])}</strong><small>{_pct(hybrid_chars["leakage_rate"])} character leakage</small></div><div class="card"><span>Hybrid recall</span><strong>{_pct(hybrid_metrics["micro"]["recall"])}</strong><small>3 of 16 transcripts still had some leak</small></div></section><div class="grid"><article class="card good"><h3>Where it was strong</h3><ul><li>The model recognized the person while model and rules agreed on the email and phone.</li><li>Structural validation recovered URLs, IP addresses, phone extensions, payment cards, device IDs, and API-key-shaped credentials that model-only detection missed or bounded incorrectly.</li><li>The resolver kept one non-overlapping set with provenance for audit.</li></ul><div class="example"><strong>Model and rules cooperate</strong><pre>{contact}</pre><span>Person, email, and phone were all exact.</span></div></article><article class="card"><h3>Where it fell down</h3><ul><li>Missed a Unicode postal address while finding the person beside it.</li><li>Missed a space-separated passphrase because it had no recognizable structure and the model did not select it.</li><li>Found a date of birth but missed the similarly formatted appointment date.</li><li>Misread <q>@ari_joon</q> as an email address rather than a username.</li></ul><div class="example"><strong>Contextual misses</strong><pre>{unicode_address}\n{passphrase}</pre><span>The rules cannot rescue values without a supported structure.</span></div></article></div><p class="muted">The hybrid's rules are format- and locale-specific. Its twelve-label schema is still incomplete, and masked context can still identify a person. This is pseudonymization, not anonymization.</p></section>
<h2>Take the next step in <code>examples/</code></h2><section class="grid actions"><article class="card"><h3>Run RoBERTa NER</h3><a href="https://github.com/curtisalexander/nerdact/tree/main/examples/roberta_ner">Open <code>examples/roberta_ner/</code> →</a><pre>uv sync
uv run python examples/roberta_ner/redact.py \
  "Ada works at Acme in Rome"</pre><p>Start here for conventional named entities.</p></article><article class="card"><h3>Run the GLiNER2 hybrid</h3><a href="https://github.com/curtisalexander/nerdact/tree/main/examples/gliner2_pii_hybrid">Open <code>examples/gliner2_pii_hybrid/</code> →</a><pre>uv sync --extra pii
uv run --extra pii python \
  examples/gliner2_pii_hybrid/redact.py \
  "Email Ada at ada@example.test"</pre><p>Start here when broader PII is the actual requirement.</p></article></section><div class="callout"><strong>Before real use:</strong> define your taxonomy, build representative calibration and held-out sets, inspect character leakage and false positives, protect source and generated reports, and require human review.</div></main><footer>All metrics come from small checked-in fictional evaluations. Follow the linked detailed reports for methods, revisions, and limitations.</footer></body></html>"""
    html_path.parent.mkdir(parents=True, exist_ok=True)
    html_path.write_text(page, encoding="utf-8")

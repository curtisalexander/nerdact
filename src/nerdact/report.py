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


def _site_nav() -> str:
    return (
        '<a href="index.html">Baseline</a> · '
        '<a href="benchmark.html">Classic checkpoints</a> · '
        '<a href="modern-encoders.html">Modern encoders</a> · '
        '<a href="runtime-labels.html">Runtime labels</a> · '
        '<a href="practical-pii.html">Practical PII</a>'
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
            f"the CoNLL {html.escape(split)} subset. Other audited candidates were rejected; "
            "see the README."
        )
        command = "uv run --extra benchmark nerdact compare-modern --limit " + str(limit)
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
        command = "uv run --extra benchmark nerdact compare --limit " + str(limit)
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>NERdact model benchmark</title><style>
:root{{--ink:#172033;--muted:#637083;--paper:#f5f7fb;--card:#fff;--accent:#5b4bdb;--line:#dfe4ee}}*{{box-sizing:border-box}}body{{margin:0;background:var(--paper);color:var(--ink);font:16px/1.6 system-ui,sans-serif}}main{{max-width:1100px;margin:auto;padding:3rem 2rem}}h1{{font-size:2.7rem;margin:.2rem 0}}.eyebrow,a{{color:var(--accent)}}.eyebrow{{font-weight:750;letter-spacing:.12em}}.card{{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:1.3rem;margin:1.5rem 0}}table{{width:100%;border-collapse:collapse;background:white;font-size:.9rem}}th,td{{padding:.8rem;border-bottom:1px solid var(--line);text-align:right}}th:first-child{{text-align:left}}th small{{display:block;color:var(--muted);font-weight:400}}code{{background:#eceefa;padding:.1rem .35rem;border-radius:4px}}.warning{{border-left:5px solid #e6a700}}@media(max-width:760px){{main{{padding:1.5rem 1rem}}table{{display:block;overflow-x:auto}}}}
</style></head><body><main><div class="eyebrow">NERDACT · MODEL COMPARISON</div><h1>{heading}</h1><nav>{_site_nav()}</nav><p>{" · ".join(report_links)}</p>
<div class="card"><p>Every checkpoint saw the same {domain_count} fictional call {transcript_label} and the same first {limit} examples from the CoNLL-2003 <strong>{html.escape(split)}</strong> split. Domain metrics require exact character boundaries and labels. Leakage and over-redaction measure character coverage regardless of the predicted label; wrong-label coverage still fails exact-span scoring.</p></div>
<table><thead><tr><th rowspan="2">Checkpoint</th><th rowspan="2">Architecture year</th><th rowspan="2">Approx. parameters</th><th colspan="5">Synthetic call transcripts</th><th colspan="3">CoNLL {html.escape(split)} ({limit})</th><th colspan="4">Warm inference cost</th></tr><tr><th>Precision</th><th>Recall</th><th>F1</th><th>Leakage</th><th>Over-redaction</th><th>Precision</th><th>Recall</th><th>F1</th><th>Median latency</th><th>Throughput</th><th>Cache snapshot</th><th>Peak memory</th></tr></thead><tbody>{"".join(table_rows)}</tbody></table>
<div class="card"><h2>Checkpoint records</h2><ul>{"".join(checkpoint_records)}</ul><p>All rows use a confidence threshold of <code>0.5</code>. Revisions appear in the table and the machine-readable artifact.</p></div>
<div class="card"><h2>Timing method and test machine</h2><p>Each checkpoint ran in a fresh process. After one untimed warm-up example, NERdact timed sequential, single-example inference over the {domain_count} checked-in {transcript_label} for {rows[0]["performance"]["timed_repeats"]} repeats using a monotonic clock and device synchronization. Throughput is examples per second. Peak memory is accelerator memory when inference uses CUDA or MPS and complete-process peak RSS on CPU; the JSON artifact also retains peak RSS. Cache snapshot size sums the unique files actually present in the pinned model snapshot, so it describes this run's downloaded files rather than every alternate format hosted in the repository.</p><p><strong>{html.escape(environment["processor"])}</strong> ({html.escape(environment["machine"])}) · {html.escape(environment["os"])} · device <code>{html.escape(environment["device"])}</code> · Python {html.escape(environment["python"])} · PyTorch {html.escape(environment["torch"])} · Transformers {html.escape(environment["transformers"])}</p></div>
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
</style></head><body><header><div class="eyebrow">NERDACT · CHARACTER-SPAN EVALUATION</div><h1>Domain report</h1><p>Model <code>{model}</code><br>Revision <code>{revision}</code></p>{_run_note(results)}<nav>{_site_nav()}</nav><div class="warning"><strong>Teaching demo, not a privacy system.</strong> This news-trained model only recognizes PERSON, ORGANIZATION, LOCATION, and MISCELLANEOUS. It does not detect addresses, phone numbers, email, or general PII.</div></header><main>
<nav aria-label="Report sections"><a href="#workflow">Workflow</a><a href="#metrics">Metrics</a><a href="#examples">Transcripts</a><a href="#next">Choosing a model</a></nav>
<h2 id="workflow">From transcript to typed placeholders</h2><p>Named-entity recognition is <strong>token classification</strong>: the model assigns labels such as <code>B-PER</code>, <code>I-PER</code>, and <code>O</code> to model tokens. Because one visible word can become several subword tokens, NERdact aggregates those predictions and converts them to stable <code>[start, end)</code> character spans in the original transcript.</p>
<section class="process"><div class="lesson step"><strong>1 · Tokenize</strong>Split text into the model's tokens while retaining source offsets.</div><div class="lesson step"><strong>2 · Classify</strong>Predict a BIO label and confidence for every token.</div><div class="lesson step"><strong>3 · Align</strong>Aggregate subwords into labeled character spans.</div><div class="lesson step"><strong>4 · Redact</strong>Assign typed placeholders in reading order and replace spans right-to-left.</div></section>
<section class="cards"><div class="card"><span>Exact micro F1</span><strong>{_pct(micro["f1"])}</strong></div><div class="card"><span>Precision</span><strong>{_pct(micro["precision"])}</strong></div><div class="card"><span>Recall</span><strong>{_pct(micro["recall"])}</strong></div><div class="card"><span>Character leakage</span><strong>{_pct(chars["leakage_rate"])}</strong><small>{chars["leaked_gold_characters"]} / {chars["gold_entity_characters"]} gold entity chars</small></div><div class="card"><span>Over-redaction</span><strong>{_pct(chars["over_redaction_rate"])}</strong><small>{chars["over_redacted_characters"]} / {chars["non_gold_characters"]} non-gold chars</small></div></section>
<h2 id="metrics">How to read the metrics</h2><div class="lesson"><p>An <strong>exact true positive</strong> has the same start, end, and label as the human annotation. Precision asks how many predictions were right; recall asks how many gold entities were found; F1 balances both. Token accuracy is intentionally not featured because abundant <code>O</code> tokens can hide sensitive-entity misses.</p><p><strong>Character leakage</strong> measures gold entity characters left uncovered by any prediction, regardless of its label. A wrong-label prediction can therefore cover a character while still failing exact-span scoring. <strong>Over-redaction</strong> measures non-entity characters covered by predictions. These coverage measures reveal partial-boundary behavior that exact F1 alone cannot explain.</p></div>
<h3>Exact <code>[start, end)</code> span metrics</h3><table><thead><tr><th>Label</th><th>TP</th><th>FP</th><th>FN</th><th>Precision</th><th>Recall</th><th>F1</th></tr></thead><tbody>{rows}</tbody></table>
<h2 id="examples">Color-coded before and after</h2><p>The <strong>before</strong> side highlights what the model—not the gold data—identified. The <strong>after</strong> side uses the same colors for its typed placeholders. Open each comparison to see the human labels, confidence scores, false positives, and false negatives.</p>{"".join(sections)}
<h2 id="next">A benchmark is not your domain</h2><div class="lesson"><p>The starter model was trained on CoNLL-2003 Reuters newswire, while these examples resemble call transcripts. Casing, punctuation, names, products, and support jargon differ. Good benchmark performance therefore does not establish safe redaction performance.</p><p>When evaluating another Hugging Face model, inspect its model card, license, training domain, language, entity taxonomy, maximum input length, revision, and behavior on representative labeled transcripts. Names and locations are only a subset of PII; practical systems usually combine domain-trained NER with deterministic detectors and human review.</p></div>
<h2>Reproduce this report</h2><pre class="reproduce"><code>uv sync
uv run nerdact report

# Inspect one string
uv run nerdact redact "Mira Sol called from Portland"

# Optional, terms-restricted CoNLL comparison
uv run --extra benchmark nerdact benchmark --limit 200</code></pre><p><small>This self-contained page contains the original fictional transcripts. A real report containing sensitive input would itself be sensitive data.</small></p></main></body></html>"""


def write_runtime_comparison(artifact: dict[str, Any], html_path: Path) -> None:
    """Render the Phase 3 label-wording and threshold experiment."""
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
uv run --extra gliner nerdact compare-gliner</code></pre><p><small>The JSON artifact contains every prediction and metric. Reports are sensitive when their inputs are sensitive.</small></p></main></body></html>"""
    html_path.parent.mkdir(parents=True, exist_ok=True)
    html_path.write_text(page, encoding="utf-8")


def write_pii_comparison(artifact: dict[str, Any], html_path: Path) -> None:
    """Render calibration, held-out metrics, and operational limits for Phase 4."""
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
<div class="card warning"><strong>Fictional demonstration, not a privacy guarantee.</strong> The sixteen evaluation transcripts are designed challenge cases. Their measured error rate is not an estimate for private calls, unseen languages, or production traffic.</div>
<h2>Calibrate before evaluating</h2><p>The model threshold was selected only on eight calibration transcripts. The fixed recall-oriented policy is: {html.escape(artifact["selection_policy"])}. The held-out evaluation split was then run once at <code>{artifact["operating_threshold"]:.2f}</code>.</p>
<table><thead><tr><th>Threshold</th><th>Precision</th><th>Recall</th><th>F1</th><th>Leakage</th><th>Over-redaction</th><th>Any-leak transcripts</th></tr></thead><tbody>{"".join(calibration_rows)}</tbody></table>
<h2>Held-out system comparison</h2><table><thead><tr><th>System</th><th>Precision</th><th>Recall</th><th>F1</th><th>Leakage</th><th>Over-redaction</th><th>Any-leak transcripts</th></tr></thead><tbody>{"".join(system_rows)}</tbody></table><p>The model and rules rows are ablations on the same evaluation examples. Leakage is uncovered gold-character coverage regardless of predicted label; wrong-label coverage still fails exact-span scoring. The hybrid merges exact duplicates and provenance, gives verifiable structures precedence in conflicts, encloses URL internals, and otherwise uses model confidence and span length. Rejected conflicts remain in the JSON artifact.</p>
<section class="cards"><div class="card"><span>Hybrid exact F1</span><strong>{_pct(metrics["micro"]["f1"])}</strong></div><div class="card"><span>Character leakage</span><strong>{_pct(chars["leakage_rate"])}</strong><small>{chars["leaked_gold_characters"]} / {chars["gold_entity_characters"]} gold characters</small></div><div class="card"><span>Over-redaction</span><strong>{_pct(chars["over_redaction_rate"])}</strong></div><div class="card"><span>Transcripts with any leak</span><strong>{transcripts["with_any_leak"]} / {transcripts["count"]}</strong></div></section>
<h2>Hybrid exact-span results by label</h2><table><thead><tr><th>Label</th><th>TP</th><th>FP</th><th>FN</th><th>Precision</th><th>Recall</th><th>F1</th></tr></thead><tbody>{label_rows}</tbody></table>
<h2>Threat model and operating procedure</h2><div class="card"><p><strong>Goal:</strong> replace the twelve annotated direct-identifier types before a transcript is shared with a less-trusted reviewer. <strong>Out of scope:</strong> inference from unmasked context, linkage attacks, voice/audio identity, unsupported labels, and maliciously crafted input. Typed masking is pseudonymization, not anonymization; remaining context can identify a caller.</p><p>Inference runs locally, but first use may download model code and weights. For private evaluation, pre-cache and verify the pinned snapshot, disable network access, prevent transcript logging and shell-history capture, use encrypted access-controlled storage, and delete source text, redacted derivatives, JSON, and HTML on a documented schedule. Publish only aggregates that cannot expose rare values.</p><p>Human review is mandatory at the recall-oriented threshold. Review every transcript against the annotation guide, prioritize misses over false positives, and route uncertain spans for correction. Do not silently tune on the held-out report; create a new representative test split after changing policy.</p></div>
<h2>Fictional hybrid demonstration</h2>{"".join(examples)}
<h2>Reproduce</h2><pre><code>uv sync --extra pii
uv run --extra pii nerdact compare-pii</code></pre><p><small>The generated JSON and HTML retain original inputs and are sensitive when inputs are sensitive. Only the checked-in fictional report is suitable for GitHub Pages.</small></p></main></body></html>"""
    html_path.parent.mkdir(parents=True, exist_ok=True)
    html_path.write_text(page, encoding="utf-8")

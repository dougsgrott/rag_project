"""Markdown report exporter for RAG offline evaluation."""

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Sequence
from rag.pipeline.evaluate import EvaluationReport, EvalRecord, aggregate

__all__ = ["generate_markdown_report", "save_markdown_report"]


def _fmt_num(val: Optional[float]) -> str:
    return f"{val:.3f}" if val is not None else "—"


def _fmt_retrieval(r: Optional[Any]) -> str:
    if r is None:
        return "— (no qrels)"
    return f"**R@{r.k}**: {r.recall_at_k:.3f} | **P@{r.k}**: {r.precision_at_k:.3f} | **MRR**: {r.mrr:.3f} | **nDCG**: {r.ndcg:.3f} | **Hit**: {int(r.hit_rate)}"


def generate_markdown_report(
    report: EvaluationReport, 
    meta_config: Dict[str, Any],
    question_types: Optional[Dict[str, str]] = None
) -> str:
    """Generates a detailed, auditable Markdown report from an EvaluationReport."""
    lines = [
        "# RAG Evaluation Experiment Report",
        f"Generated on: `{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}`",
        "",
        "## Experiment Configuration",
        f"- **Domain Context**: `{meta_config.get('domain', 'default')}`",
        f"- **Chat Model (LLM)**: `{meta_config.get('chat_model', 'N/A')}`",
        f"- **Embedding Model**: `{meta_config.get('embedding_model', 'N/A')}`",
        f"- **Vector Search Cutoff (Wide)**: `top_k={meta_config.get('search_top_k', 150)}`",
        f"- **Reranker Cutoff (Tight)**: `top_k={meta_config.get('final_top_k', 5)}`",
        f"- **Total Test Cases Processed**: **{len(report.records)}**",
        "",
        "## Summary Metrics",
        "| Metric | Global Score |",
        "| :--- | :--- |",
        f"| Faithfulness | {_fmt_num(report.mean_faithfulness)} |",
        f"| Answer Relevancy | {_fmt_num(report.mean_answer_relevancy)} |",
        f"| Context Precision | {_fmt_num(report.mean_context_precision)} |",
        f"| Context Recall | {_fmt_num(report.mean_context_recall)} |",
        f"| Answer Correctness | {_fmt_num(report.mean_answer_correctness)} |",
        "",
        "### Global Retrieval System Performance",
        f"- **Retrieval Stage (Wide Search)**: {_fmt_retrieval(report.mean_retrieval)}",
        f"- **Reranking Stage (Tight Context)**: {_fmt_retrieval(report.mean_rerank)}",
        ""
    ]

    # Category Breakdowns (Slicing by question_type sidecar)
    if question_types and report.records:
        lines.append("## Metrics by Question Type")
        by_type: Dict[str, list[EvalRecord]] = {}
        for rec in report.records:
            q_type = question_types.get(rec.case.query, "unknown")
            by_type.setdefault(q_type, []).append(rec)
            
        for q_type, recs in sorted(by_type.items()):
            sub_report = aggregate(recs)
            lines.extend([
                f"### Category: `{q_type}` ({len(recs)} cases)",
                "| Metric | Score |",
                "| :--- | :--- |",
                f"| Faithfulness | {_fmt_num(sub_report.mean_faithfulness)} |",
                f"| Answer Relevancy | {_fmt_num(sub_report.mean_answer_relevancy)} |",
                f"| Context Precision | {_fmt_num(sub_report.mean_context_precision)} |",
                f"| Context Recall | {_fmt_num(sub_report.mean_context_recall)} |",
                f"| Answer Correctness | {_fmt_num(sub_report.mean_answer_correctness)} |",
                f"| **Retrieval (Search)** | {_fmt_retrieval(sub_report.mean_retrieval)} |",
                f"| **Retrieval (Rerank)** | {_fmt_retrieval(sub_report.mean_rerank)} |",
                ""
            ])

    # Granular Per-Case Inspection Blocks
    lines.append("## Per-Case Breakdown")
    for i, rec in enumerate(report.records, 1):
        q_type = question_types.get(rec.case.query, "N/A") if question_types else "N/A"
        lines.extend([
            "---",
            f"### Case {i}: {rec.case.query}",
            f"- **Question Type Label**: `{q_type}`",
            f"- **Gold Ground Truth Reference**: *{rec.case.reference or 'None provided'}*",
            f"- **System Generated Answer**: {rec.answer.content}",
            "",
            "#### Accuracy Scores",
            "| Faithfulness | Answer Relevancy | Context Precision | Context Recall | Answer Correctness |",
            "| :---: | :---: | :---: | :---: | :---: |",
            f"| {_fmt_num(rec.result.faithfulness)} | {_fmt_num(rec.result.answer_relevancy)} | {_fmt_num(rec.result.context_precision)} | {_fmt_num(rec.result.context_recall)} | {_fmt_num(rec.result.answer_correctness)} |",
            "",
            f"- **Retriever Metrics**: {_fmt_retrieval(rec.retrieval)}",
            f"- **Reranker Metrics**: {_fmt_retrieval(rec.rerank)}",
            ""
        ])

        # Collapsible Section for Final Reranked Chunks sent to Generator
        lines.extend([
            "<details>",
            "<summary><b>View Final Reranked Context (Top Chunks sent to LLM)</b></summary>",
            ""
        ])
        if not rec.context:
            lines.append("*No chunks selected by reranker.*")
        for idx, item in enumerate(rec.context, 1):
            lines.extend([
                f"##### [{idx}] Source: `{item.chunk.document_source}` (Position: {item.chunk.position}) | Rerank Score: `{item.score:.4f}`",
                "```text",
                item.chunk.content.strip(),
                "```",
                ""
            ])
        lines.append("</details>")
        lines.append("")

        # Collapsible Section for Wide Candidate Set
        lines.extend([
            "<details>",
            "<summary><b>View Pre-Rerank Wide Candidate Set (First 10 Truncated)</b></summary>",
            ""
        ])
        if not rec.candidates:
            lines.append("*No candidate items found in initial vector store search.*")
        for idx, item in enumerate(rec.candidates[:10], 1):
            lines.extend([
                f"##### [{idx}] Source: `{item.chunk.document_source}` (Position: {item.chunk.position}) | Vector Score: `{item.score:.4f}`",
                "```text",
                item.chunk.content.strip()[:200] + "...",
                "```",
                ""
            ])
        if len(rec.candidates) > 10:
            lines.append(f"*... and {len(rec.candidates) - 10} additional wide candidate chunks skipped in preview.*")
        lines.append("</details>")
        lines.append("")

    return "\n".join(lines)


def save_markdown_report(
    report: EvaluationReport,
    file_path: str | Path,
    meta_config: Dict[str, Any],
    question_types: Optional[Dict[str, str]] = None
) -> None:
    """Compiles the report strings and commits them safely to disk."""
    md_content = generate_markdown_report(report, meta_config, question_types)
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(md_content, encoding="utf-8")
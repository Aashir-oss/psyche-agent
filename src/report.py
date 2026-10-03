"""DOCX report builder. python-docx is imported lazily so `import src.report` is light.

The document is generated from STRUCTURED data (the same dicts the UI shows),
not re-typed by an LLM — so numbers, card IDs, and references match the UI exactly.

LLM text is sanitized before insertion: the editor model sometimes emits
backslash-escaped markdown (\\*, \\[, \\-, trailing \\) which would otherwise
leak into the document as visible garbage.
"""
import re
from datetime import datetime
from pathlib import Path

from .config import REPORTS_DIR, DISCLAIMER


def _docx():
    try:
        from docx import Document
        from docx.shared import Pt, Inches
        from docx.enum.text import WD_ALIGN_PARAGRAPH
    except ImportError as e:
        raise RuntimeError(
            f"DOCX export needs python-docx: pip install -r requirements.txt ({e.name})."
        )
    return Document, Pt, Inches, WD_ALIGN_PARAGRAPH


def _clean_text(s) -> str:
    """Strip markdown artifacts from LLM output so the DOCX reads cleanly."""
    if not s:
        return ""
    t = str(s)
    t = t.replace("\\\n", "\n")          # escaped newline
    t = re.sub(r"\\([\\*\[\]()_#\-+.!])", r"\1", t)  # \* -> *, \[ -> [, \- -> - ...
    t = re.sub(r"\\\s*$", "", t)          # trailing backslash (md hard-break)
    t = re.sub(r"\*\*(.+?)\*\*", r"\1", t)  # **bold** -> bold
    t = re.sub(r"__(.+?)__", r"\1", t)      # __bold__ -> __
    t = re.sub(r"\*([^*]+)\*", r"\1", t)    # *italic* -> italic
    t = t.replace("`", "")
    t = re.sub(r"[ \t]+", " ", t)
    return t.strip()


def _clean_data(data: dict) -> dict:
    """Recursively sanitize every string in the report payload."""
    if isinstance(data, dict):
        return {k: _clean_data(v) for k, v in data.items()}
    if isinstance(data, list):
        return [_clean_data(v) for v in data]
    if isinstance(data, str):
        return _clean_text(data)
    return data


def build_docx(data: dict, graph_paths: list[str], out_name: str | None = None) -> str:
    """Build the formatted .docx report. Returns the file path."""
    Document, Pt, Inches, WD_ALIGN_PARAGRAPH = _docx()
    data = _clean_data(data or {})
    doc = Document()

    style = doc.styles["Normal"]
    style.font.size = Pt(11)

    # ---- title page ----
    title = doc.add_heading("Your Personal Psych — Psychology Assessment Report", level=0)
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    sub = doc.add_paragraph()
    sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
    sub.add_run(
        f"{data.get('username', '')}  •  "
        f"{data.get('date', datetime.now().strftime('%Y-%m-%d'))}"
    ).font.size = Pt(11)

    disc = doc.add_paragraph()
    disc.add_run("Important — please read: ").bold = True
    disc.add_run(DISCLAIMER)
    doc.add_page_break()

    # ---- summary ----
    doc.add_heading("Summary", level=1)
    doc.add_paragraph(data.get("summary") or "")

    # ---- perspectives ----
    doc.add_heading("Perspectives", level=1)
    for p in data.get("perspectives") or []:
        doc.add_heading(f"{p.get('name') or ''} perspective", level=2)
        doc.add_paragraph(f"Hypothesis: {p.get('hypothesis') or ''}")
        doc.add_paragraph(f"Confidence: {p.get('confidence') or ''}")
        if p.get("key_factors"):
            doc.add_paragraph(f"Key factors: {p['key_factors']}")

    # ---- what the perspectives agree on ----
    if data.get("consensus"):
        doc.add_heading("What the perspectives agree on", level=1)
        doc.add_paragraph(data["consensus"])

    # ---- questions explored (with the user's answers) ----
    qa_pairs = data.get("qa_pairs") or []
    if qa_pairs:
        doc.add_heading("Questions explored", level=1)
        skipped = 0
        for qa in qa_pairs:
            ans = (qa.get("answer", "") or "").strip()
            if not ans or ans == "(skipped)":
                skipped += 1
                continue
            doc.add_paragraph(
                f"[{qa.get('perspective') or ''}] {qa.get('question') or ''}",
                style="List Bullet",
            )
            doc.add_paragraph(f"→ {ans}")
        if skipped:
            p = doc.add_paragraph()
            p.add_run(f"({skipped} question(s) skipped)").italic = True
    elif data.get("questions_asked"):
        doc.add_heading("Questions explored", level=1)
        for q in data["questions_asked"]:
            doc.add_paragraph(q, style="List Bullet")

    # ---- advice paths (heading only when the planner produced paths) ----
    advice_paths = data.get("advice_paths") or []
    if advice_paths:
        doc.add_heading("Advice paths", level=1)
    for i, path in enumerate(advice_paths, 1):
        # last-gate sanitization: never let a raw "PATH 1:" prefix or a
        # TECHNIQUE_IDS line leak into the user-facing document
        _title = re.sub(r"^(PATH\s*\d+\s*[:\-–—]\s*)+", "",
                        path.get("title") or "", flags=re.IGNORECASE)
        doc.add_heading(f"Path {i}: {_title}", level=2)
        for step in path.get("steps") or []:
            # strip raw "(card xyz)" markers — technique names carry it already
            step = re.sub(r"\s*\(card [a-z0-9_]+\)", "", step or "")
            if re.match(r"(?i)^\s*TECHNIQUE_IDS\s*:", step):
                continue
            doc.add_paragraph(step, style="List Bullet")
        tnames = path.get("technique_names") or [
            t.replace("_", " ").title() for t in path.get("technique_ids", [])
        ]
        if tnames:
            doc.add_paragraph(
                "Techniques used: " + ", ".join(tnames)
            ).italic = True

    # ---- screening scores (GAD-7 / PHQ-9) ----
    screening = data.get("screening") or []
    if screening:
        doc.add_heading("Screening scores", level=1)
        for s in screening:
            doc.add_paragraph(
                f"{s.get('scale') or ''}: {s.get('score') or ''} "
                f"({s.get('band') or ''}) — checked {s.get('date') or ''}",
                style="List Bullet",
            )
        doc.add_paragraph(
            "GAD-7 and PHQ-9 are standard screening questionnaires, not diagnostic "
            "tools. These scores are screening signals, not diagnoses — discuss "
            "them with a qualified professional."
        )

    # ---- graphs (only when there is at least one) ----
    real_graphs = [gp for gp in (graph_paths or []) if gp and Path(gp).exists()]
    if real_graphs:
        doc.add_heading("Progress chart", level=1)
        for gp in real_graphs:
            doc.add_picture(gp, width=Inches(5.5))

    # ---- when to seek help (only when the planner actually said something) ----
    if (data.get("when_to_seek_help") or "").strip():
        doc.add_heading("When to see a professional", level=1)
        doc.add_paragraph(data["when_to_seek_help"])

    # ---- references (only when there is at least one) ----
    if data.get("references"):
        doc.add_heading("References", level=1)
        doc.add_paragraph(
            "Every technique below comes from the project's vetted knowledge library. "
            "Summaries are original; methods are cited to their sources."
        )
        for r in data["references"]:
            p = doc.add_paragraph(style="List Bullet")
            line = (f"{r.get('id') or ''} — {r.get('title') or ''} "
                    f"— {r.get('source') or ''}")
            if r.get("license"):
                line += f" [{r['license']}]"
            if r.get("url"):
                line += f" ({r['url']})"
            p.add_run(line)

    doc.add_paragraph("")
    end = doc.add_paragraph()
    end.add_run(DISCLAIMER).italic = True

    out_name = out_name or f"psyche_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.docx"
    out = REPORTS_DIR / out_name
    doc.save(out)
    return str(out)

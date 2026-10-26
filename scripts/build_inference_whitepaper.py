#!/usr/bin/env python3
"""Build the editable inference specification as a standalone A4 PDF.

Install docs/whitepaper/requirements.txt into the project's .venv. Content lives
in docs/whitepaper/inference.json; no network or sibling source is needed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    FrameBreak,
    NextPageTemplate,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("docs/whitepaper/inference.json"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("Use a new output filename; existing artifacts are never overwritten")
    data = json.loads(args.source.read_text())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    width, height = A4
    margin, gutter, footer = 43, 23, 44
    column_width = (width - 2 * margin - gutter) / 2
    ink, muted, line = (
        colors.HexColor("#252a32"),
        colors.HexColor("#63718a"),
        colors.HexColor("#d1d7df"),
    )
    body = ParagraphStyle(
        "Body",
        fontName="Helvetica",
        fontSize=9.4,
        leading=13.2,
        textColor=ink,
        spaceAfter=9,
        alignment=TA_LEFT,
        splitLongWords=True,
        allowWidows=0,
        allowOrphans=0,
    )
    heading = ParagraphStyle(
        "Heading",
        parent=body,
        fontName="Helvetica-Bold",
        fontSize=11.2,
        leading=14,
        spaceBefore=10,
        spaceAfter=7,
        keepWithNext=True,
    )
    small = ParagraphStyle("Reference", parent=body, fontSize=8, leading=10.5, spaceAfter=7)
    seen_pages = []

    def decorate(canvas, document):
        page = document.page
        seen_pages.append(page)
        canvas.saveState()
        canvas.setTitle(data["title"])
        canvas.setAuthor("Attelier")
        canvas.setSubject(data["subtitle"])
        canvas.setCreator("Attelier whitepaper builder")
        canvas.setFillColor(muted)
        canvas.setFont("Helvetica", 8)
        canvas.drawString(margin, height - 35, "ATTELIER  /  WHITEPAPER")
        canvas.drawRightString(width - margin, height - 35, data["date"])
        canvas.setStrokeColor(line)
        canvas.setLineWidth(0.45)
        canvas.line(margin, height - 44, width - margin, height - 44)
        canvas.setFillColor(ink)
        if page == 1:
            canvas.setFont("Helvetica-Bold", 27)
            canvas.drawString(margin, height - 89, data["title"])
            canvas.setFont("Helvetica", 10)
            canvas.drawString(margin, height - 121, data["subtitle"])
            canvas.setFillColor(muted)
            canvas.setFont("Helvetica", 8)
            canvas.drawString(margin, height - 138, data["status"])
        elif page <= len(data["pages"]):
            title = Paragraph(
                escape(data["pages"][page - 1]["title"]),
                ParagraphStyle(
                    "PageTitle", fontName="Helvetica-Bold", fontSize=19, leading=23, textColor=ink
                ),
            )
            _, title_height = title.wrap(width - 2 * margin, 65)
            title.drawOn(canvas, margin, height - 65 - title_height)
        canvas.setFillColor(muted)
        canvas.setFont("Helvetica", 7.3)
        canvas.drawString(
            margin, 25, "Permissioned bootstrap · Accountable service · Time to video"
        )
        canvas.drawRightString(width - margin, 25, f"{page:02d} / {len(data['pages']):02d}")
        canvas.restoreState()

    templates = []
    for index, _page in enumerate(data["pages"]):
        top = height - (184 if index == 0 else 112)
        frames = [
            Frame(
                margin + side * (column_width + gutter),
                footer,
                column_width,
                top - footer,
                id=f"p{index}-c{side}",
                leftPadding=0,
                rightPadding=0,
                topPadding=0,
                bottomPadding=0,
            )
            for side in (0, 1)
        ]
        templates.append(PageTemplate(id=f"page-{index}", frames=frames, onPage=decorate))
    document = BaseDocTemplate(
        str(args.out),
        pagesize=A4,
        pageTemplates=templates,
        leftMargin=margin,
        rightMargin=margin,
        topMargin=44,
        bottomMargin=footer,
    )
    story = []
    for page_index, page in enumerate(data["pages"]):
        if page_index:
            story.extend([NextPageTemplate(f"page-{page_index}"), PageBreak()])
        for column_index, sections in enumerate(page["columns"]):
            if column_index:
                story.append(FrameBreak())
            for section in sections:
                story.append(Paragraph(escape(section["heading"]), heading))
                for paragraph in section["paragraphs"]:
                    story.append(Paragraph(escape(paragraph), body))
            if page_index == len(data["pages"]) - 1 and column_index == 1:
                story.extend([Spacer(1, 6), Paragraph("Sources", heading)])
                for index, reference in enumerate(data["references"], 1):
                    url = reference["url"]
                    if not url.startswith("https://"):
                        raise ValueError("Reference must be HTTPS")
                    story.append(
                        Paragraph(
                            f'[{index}] <link href="{escape(url)}" color="#445c7d">'
                            f"{escape(reference['label'])}</link>",
                            small,
                        )
                    )
    document.build(story)
    if len(seen_pages) != len(data["pages"]):
        raise RuntimeError(
            f"Content overflow: expected {len(data['pages'])} pages, got {len(seen_pages)}"
        )
    receipt = {
        "sourceSha256": hashlib.sha256(args.source.read_bytes()).hexdigest(),
        "pdfSha256": hashlib.sha256(args.out.read_bytes()).hexdigest(),
        "pages": len(seen_pages),
        "author": "Attelier",
        "status": data["status"],
    }
    args.out.with_suffix(".receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()

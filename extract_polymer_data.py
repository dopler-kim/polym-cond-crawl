#!/usr/bin/env python3
"""
Polymer Solid Electrolyte Data Extractor
=========================================
Crawls experimental data from PDF research articles on polymer-based solid
electrolytes using Claude API (text + vision). Outputs a formatted Excel file.

Required environment variable:
    ANTHROPIC_API_KEY — your Anthropic API key

Usage:
    python extract_polymer_data.py [--desktop-path PATH] [--output PATH]
"""

import argparse
import base64
import json
import os
import re
import time
from pathlib import Path
from typing import Optional

import fitz  # PyMuPDF
import anthropic
import pandas as pd
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_DESKTOP = Path.home() / "Desktop"
DEFAULT_OUTPUT  = DEFAULT_DESKTOP / "polymer_electrolyte_data.xlsx"
MODEL           = "claude-sonnet-4-6"

# Minimum image dimensions to bother analysing (skip icons / logos)
MIN_IMAGE_WIDTH  = 200
MIN_IMAGE_HEIGHT = 150

# Maximum characters sent to Claude for text extraction
MAX_TEXT_CHARS = 180_000

# Seconds to wait between successive image API calls (rate-limit safety)
IMAGE_CALL_DELAY = 0.5

COLUMN_ORDER = [
    "doi",
    "paper_title",
    "polymer_structure",
    "polymer_length",
    "filler",
    "ionic_salt",
    "ratio_polymer_filler_salt",
    "temperature_C",
    "humidity_percent",
    "ionic_conductivity_S_cm",
    "notes",
    "source_file",
]

COLUMN_WIDTHS = {
    "doi":                       32,
    "paper_title":               42,
    "polymer_structure":         40,
    "polymer_length":            22,
    "filler":                    22,
    "ionic_salt":                18,
    "ratio_polymer_filler_salt": 28,
    "temperature_C":             16,
    "humidity_percent":          16,
    "ionic_conductivity_S_cm":   26,
    "notes":                     44,
    "source_file":               32,
}

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

TEXT_PROMPT = """\
You are an expert in polymer chemistry and solid electrolyte research.

Extract ALL experimental data points from the research paper text below.
Each unique sample / experimental condition should become one JSON object.

Fields to extract (use null when not reported):
  doi                       — DOI string, e.g. "10.1039/c9ta00001a"
  paper_title               — full title of the paper
  polymer_structure         — SMILES string of the repeat unit (monomer) structure,
                              e.g. "COCCO" for PEO, "CC(F)(F)CC(F)(F)" for PVDF.
                              If the SMILES cannot be determined, fall back to the
                              common abbreviation (e.g. "PEO").
  polymer_length            — molecular weight or degree of polymerisation,
                              e.g. "600000 g/mol", "DP=200"
  filler                    — inorganic/organic filler, e.g. "LLZO", "SiO2",
                              or "none" if explicitly stated as unfilled
  ionic_salt                — lithium or other salt, e.g. "LiTFSI", "LiClO4"
  ratio_polymer_filler_salt — weight or molar ratio, e.g. "70:10:20 wt%"
  temperature_C             — measurement temperature in °C as a number string
  humidity_percent          — relative humidity if reported, e.g. "50" or "dry"
  ionic_conductivity_S_cm   — conductivity in S/cm, e.g. "1.2e-4"
  notes                     — any other important details (preparation method,
                              special conditions, etc.)

Return ONLY a valid JSON array — no markdown, no explanation.
If no relevant data is found, return [].

Paper text:
{text}
"""

IMAGE_PROMPT = """\
You are an expert in polymer chemistry and solid electrolyte research.

Examine the figure below from a polymer-based solid electrolyte paper.

If the figure contains quantitative experimental data (conductivity plot,
Arrhenius plot, bar chart, table, etc.) extract every readable data point.

Use these fields (null when not determinable from the image):
  doi, paper_title, polymer_structure (SMILES of monomer repeat unit, or
  abbreviation if SMILES cannot be determined), polymer_length, filler,
  ionic_salt, ratio_polymer_filler_salt, temperature_C, humidity_percent,
  ionic_conductivity_S_cm, notes

For "notes" describe what the figure shows, e.g.:
  "Arrhenius plot, read from curve labelled PEO/LiTFSI"

Return ONLY a valid JSON array.
If the figure is a schematic, molecular diagram, or contains no numerical
conductivity data, return [].

Figure caption / nearby text:
{caption}
"""

# ---------------------------------------------------------------------------
# PDF helpers
# ---------------------------------------------------------------------------

def extract_text(pdf_path: Path) -> str:
    """Return concatenated plain text from all pages of a PDF."""
    doc = fitz.open(str(pdf_path))
    pages = [page.get_text() for page in doc]
    doc.close()
    return "\n".join(pages)


def extract_images(pdf_path: Path) -> list[dict]:
    """
    Return a list of dicts:
        data    : raw image bytes
        ext     : file extension string ("png", "jpeg", …)
        page    : 1-based page number
        context : first 500 chars of that page's text (used as caption hint)
    Only images larger than MIN_IMAGE_WIDTH × MIN_IMAGE_HEIGHT are included.
    """
    doc = fitz.open(str(pdf_path))
    images = []
    for page_num, page in enumerate(doc):
        page_text = page.get_text()
        for img_ref in page.get_images(full=True):
            xref = img_ref[0]
            try:
                img = doc.extract_image(xref)
                if (img
                        and img["width"]  >= MIN_IMAGE_WIDTH
                        and img["height"] >= MIN_IMAGE_HEIGHT):
                    images.append({
                        "data":    img["image"],
                        "ext":     img["ext"],
                        "page":    page_num + 1,
                        "context": page_text[:500],
                    })
            except Exception:
                continue
    doc.close()
    return images


def find_doi(text: str) -> Optional[str]:
    """Quick regex DOI extraction (fallback / sanity check)."""
    pattern = r'\b(10\.\d{4,}/[^\s\]\[,;:"\'<>{}\|\\^`\n]+)'
    matches = re.findall(pattern, text)
    return matches[0].rstrip(".") if matches else None

# ---------------------------------------------------------------------------
# Claude API helpers
# ---------------------------------------------------------------------------

_MEDIA_TYPES = {
    "png":  "image/png",
    "jpg":  "image/jpeg",
    "jpeg": "image/jpeg",
    "gif":  "image/gif",
    "webp": "image/webp",
    "bmp":  "image/bmp",
}


def _parse_json_array(text: str) -> list[dict]:
    """Extract the first JSON array from an arbitrary Claude response string."""
    match = re.search(r'\[[\s\S]*\]', text)
    if not match:
        return []
    try:
        result = json.loads(match.group())
        return result if isinstance(result, list) else []
    except json.JSONDecodeError:
        return []


def claude_extract_text(client: anthropic.Anthropic, text: str) -> list[dict]:
    """Send paper text to Claude and return extracted data-point dicts."""
    if len(text) > MAX_TEXT_CHARS:
        text = text[:MAX_TEXT_CHARS] + "\n\n[Text truncated]"

    response = client.messages.create(
        model=MODEL,
        max_tokens=4096,
        messages=[{"role": "user", "content": TEXT_PROMPT.format(text=text)}],
    )
    return _parse_json_array(response.content[0].text)


def claude_extract_image(
    client: anthropic.Anthropic,
    image_bytes: bytes,
    ext: str,
    caption: str,
) -> list[dict]:
    """Send a figure to Claude vision and return extracted data-point dicts."""
    media_type = _MEDIA_TYPES.get(ext.lower(), "image/png")
    b64 = base64.standard_b64encode(image_bytes).decode()

    response = client.messages.create(
        model=MODEL,
        max_tokens=2048,
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type":       "base64",
                        "media_type": media_type,
                        "data":       b64,
                    },
                },
                {
                    "type": "text",
                    "text": IMAGE_PROMPT.format(caption=caption[:400] or "N/A"),
                },
            ],
        }],
    )
    return _parse_json_array(response.content[0].text)

# ---------------------------------------------------------------------------
# Per-PDF processing
# ---------------------------------------------------------------------------

def process_pdf(pdf_path: Path, client: anthropic.Anthropic) -> list[dict]:
    """
    Extract all data points from a single PDF.
    1. Text  → Claude text extraction
    2. Images → Claude vision extraction (figures / graphs)
    Returns a list of dicts, each representing one data point.
    """
    print(f"\n[PDF] {pdf_path.name}")
    all_points: list[dict] = []

    # ---- Text ---------------------------------------------------------
    print("  Extracting text …")
    text = extract_text(pdf_path)

    if text.strip():
        doi_hint = find_doi(text)
        if doi_hint:
            print(f"  DOI (regex): {doi_hint}")

        print("  Querying Claude (text) …")
        try:
            points = claude_extract_text(client, text)
        except Exception as exc:
            print(f"  Warning – text query failed: {exc}")
            points = []

        print(f"  → {len(points)} data point(s) from text")
        for p in points:
            p["source_file"] = pdf_path.name
        all_points.extend(points)
    else:
        print("  No extractable text (scanned PDF?). Skipping text step.")

    # Carry forward DOI / title for image entries if Claude found them
    carried_doi   = next((p.get("doi")         for p in all_points if p.get("doi")),         None)
    carried_title = next((p.get("paper_title") for p in all_points if p.get("paper_title")), None)

    # ---- Images -------------------------------------------------------
    print("  Extracting images …")
    images = extract_images(pdf_path)
    print(f"  {len(images)} image(s) found")

    for idx, img in enumerate(images, 1):
        print(f"  Querying Claude (image {idx}/{len(images)}, p.{img['page']}) …")
        try:
            img_points = claude_extract_image(
                client, img["data"], img["ext"], img["context"]
            )
        except Exception as exc:
            print(f"  Warning – image query failed: {exc}")
            img_points = []

        for p in img_points:
            if not p.get("doi")         and carried_doi:   p["doi"]         = carried_doi
            if not p.get("paper_title") and carried_title: p["paper_title"] = carried_title
            p["source_file"] = pdf_path.name
        all_points.extend(img_points)

        if idx < len(images):
            time.sleep(IMAGE_CALL_DELAY)

    print(f"  Total data points: {len(all_points)}")
    return all_points

# ---------------------------------------------------------------------------
# Excel output
# ---------------------------------------------------------------------------

HEADER_COLOR = "1F4E79"   # dark blue
ALT_ROW_COLOR = "D6E4F0"  # light blue


def save_excel(data: list[dict], output_path: Path) -> None:
    """Write collected data to a formatted Excel workbook."""
    if not data:
        print("\nNo data extracted — Excel file not created.")
        return

    df = pd.DataFrame(data)
    for col in COLUMN_ORDER:
        if col not in df.columns:
            df[col] = None
    df = df[COLUMN_ORDER].fillna("")

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="Extracted Data", index=False)
        ws = writer.sheets["Extracted Data"]

        # Header styling
        hdr_fill = PatternFill(start_color=HEADER_COLOR,
                               end_color=HEADER_COLOR, fill_type="solid")
        hdr_font = Font(color="FFFFFF", bold=True, size=11)
        hdr_align = Alignment(horizontal="center", vertical="center",
                               wrap_text=True)

        for col_idx, col_name in enumerate(COLUMN_ORDER, 1):
            cell = ws.cell(row=1, column=col_idx)
            cell.fill  = hdr_fill
            cell.font  = hdr_font
            cell.alignment = hdr_align

        # Column widths
        for col_idx, col_name in enumerate(COLUMN_ORDER, 1):
            ws.column_dimensions[get_column_letter(col_idx)].width = \
                COLUMN_WIDTHS.get(col_name, 20)

        # Freeze header
        ws.freeze_panes = "A2"

        # Alternating row background
        alt_fill = PatternFill(start_color=ALT_ROW_COLOR,
                               end_color=ALT_ROW_COLOR, fill_type="solid")
        body_align = Alignment(vertical="top", wrap_text=True)

        for row_idx in range(2, len(df) + 2):
            for col_idx in range(1, len(COLUMN_ORDER) + 1):
                cell = ws.cell(row=row_idx, column=col_idx)
                cell.alignment = body_align
                if row_idx % 2 == 0:
                    cell.fill = alt_fill

        # Row height hint
        ws.row_dimensions[1].height = 30

    print(f"\nSaved → {output_path}  ({len(df)} row(s))")

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract polymer electrolyte data from PDF articles."
    )
    parser.add_argument(
        "--desktop-path",
        type=Path,
        default=DEFAULT_DESKTOP,
        help=f"Folder containing PDFs (default: {DEFAULT_DESKTOP})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output Excel file path (default: {DEFAULT_OUTPUT})",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "ANTHROPIC_API_KEY is not set.\n"
            "Run:  export ANTHROPIC_API_KEY='sk-ant-...'"
        )

    client = anthropic.Anthropic(api_key=api_key)

    pdf_files = sorted(args.desktop_path.glob("*.pdf"))
    if not pdf_files:
        print(f"No PDF files found in: {args.desktop_path}")
        return

    print(f"Found {len(pdf_files)} PDF(s) in {args.desktop_path}")
    print(f"Output  → {args.output}\n")

    all_data: list[dict] = []
    for pdf_path in pdf_files:
        try:
            all_data.extend(process_pdf(pdf_path, client))
        except Exception as exc:
            print(f"  ERROR processing {pdf_path.name}: {exc}")

    save_excel(all_data, args.output)
    print("Done.")


if __name__ == "__main__":
    main()

# scripts/extract_text.py
import fitz  # PyMuPDF
from pathlib import Path
import re

PDF_FILE = r"C:\Users\Acer\Desktop\projects\week_5\day1\data\constitution.pdf"
OUT_FILE = Path("constitution.txt")

def extract_text_with_layout(pdf_file, skip_pages=7):
    doc = fitz.open(pdf_file)
    text_pages = []

    for i, page in enumerate(doc):
        if i < skip_pages:
            continue  # skip first `skip_pages` pages

        blocks = page.get_text("blocks")  # get blocks with position info
        # sort blocks top to bottom, left to right
        blocks.sort(key=lambda b: (b[1], b[0]))  # sort by y0, x0
        page_text = ""
        for b in blocks:
            block_text = b[4].strip()
            if block_text:
                page_text += block_text + "\n"

        # Remove page numbers at end of page (assumes page number is a line with only digits)
        page_text = re.sub(r"\n\d+\s*$", "", page_text.strip())

        text_pages.append(page_text)

    return "\n".join(text_pages)

if __name__ == "__main__":
    raw_text = extract_text_with_layout(PDF_FILE)
    OUT_FILE.write_text(raw_text, encoding="utf-8")
    print(f"Extracted text with layout saved to {OUT_FILE}")

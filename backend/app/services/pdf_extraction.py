import io

import pdfplumber


class PDFExtractionError(Exception):
    """Raised when a PDF cannot be parsed or contains no extractable text."""


def extract_text_from_pdf(data: bytes) -> tuple[str, int]:
    """Extract text from PDF bytes. Returns (text, page_count)."""
    try:
        with pdfplumber.open(io.BytesIO(data)) as pdf:
            pages = [page.extract_text() or "" for page in pdf.pages]
    except Exception as exc:
        raise PDFExtractionError(f"Could not parse PDF file: {exc}") from exc

    text = "\n\n".join(pages).strip()
    if not text:
        raise PDFExtractionError(
            "No extractable text found. The PDF may be a scanned image or empty."
        )
    return text, len(pages)

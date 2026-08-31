import pytest

from app.services.pdf_extraction import PDFExtractionError, extract_text_from_pdf
from tests.utils import make_pdf


def test_extracts_text_and_page_count():
    pdf = make_pdf(["Jane Doe", "Skills: Python, React"])
    text, page_count = extract_text_from_pdf(pdf)
    assert page_count == 1
    assert "Jane Doe" in text
    assert "Python" in text


def test_rejects_invalid_pdf():
    with pytest.raises(PDFExtractionError):
        extract_text_from_pdf(b"this is not a pdf")


def test_rejects_pdf_without_text():
    with pytest.raises(PDFExtractionError):
        extract_text_from_pdf(make_pdf([]))

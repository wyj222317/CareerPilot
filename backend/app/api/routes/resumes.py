from fastapi import APIRouter, Depends, HTTPException, UploadFile

from app.core.config import Settings, get_settings
from app.models.schemas import ResumeUploadResponse
from app.services.pdf_extraction import PDFExtractionError, extract_text_from_pdf

router = APIRouter(prefix="/api/resumes", tags=["resumes"])


@router.post("/upload", response_model=ResumeUploadResponse)
async def upload_resume(
    file: UploadFile,
    settings: Settings = Depends(get_settings),
) -> ResumeUploadResponse:
    filename = file.filename or "resume.pdf"
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=415, detail="Only PDF resumes are supported.")

    data = await file.read()
    if not data:
        raise HTTPException(status_code=422, detail="Uploaded file is empty.")
    if len(data) > settings.max_upload_size_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"File exceeds the {settings.max_upload_size_mb} MB limit.",
        )
    if not data[:5].startswith(b"%PDF-"):
        raise HTTPException(
            status_code=415,
            detail="File does not appear to be a valid PDF.",
        )

    try:
        text, page_count = extract_text_from_pdf(data)
    except PDFExtractionError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return ResumeUploadResponse(
        filename=filename,
        page_count=page_count,
        character_count=len(text),
        extracted_text=text,
    )

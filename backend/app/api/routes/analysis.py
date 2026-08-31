from fastapi import APIRouter, Depends, HTTPException

from app.models.schemas import MatchAnalysisResponse, MatchRequest
from app.services.analysis import MatchAnalysisService
from app.services.llm import get_llm_service
from app.services.llm.base import LLMService

router = APIRouter(prefix="/api/analysis", tags=["analysis"])


@router.post("/match", response_model=MatchAnalysisResponse)
def analyze_match(
    request: MatchRequest,
    llm_service: LLMService = Depends(get_llm_service),
) -> MatchAnalysisResponse:
    try:
        return MatchAnalysisService(llm_service).analyze(
            resume_text=request.resume_text,
            job_description=request.job_description,
        )
    except ValueError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

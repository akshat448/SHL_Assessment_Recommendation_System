from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Union
from pydantic import BaseModel, Field

class RecommendationRequest(BaseModel):
    query: str = Field(..., description="Natural language query or job description URL")
    top_k: int = Field(10, description="Maximum number of candidates to retrieve")
    top_n: int = Field(5, description="Maximum number of final recommendations to return")
    use_reranker: bool = Field(True, description="Whether to use the reranker to improve results")
    use_llm_explanations: bool = Field(True, description="Whether to generate explanations using LLM")

class AssessmentRecommendation(BaseModel):
    assessment_name: str
    assessment_url: Optional[str] = None
    remote_testing: Optional[str] = None
    adaptive_support: Optional[str] = None
    duration_minutes: Optional[Union[int, str]] = None
    test_type_categories: List[str] = []
    job_levels: List[str] = []
    languages: List[str] = []
    explanation: Optional[str] = None
    score: Optional[float] = None

class RecommendationResponse(BaseModel):
    query: str
    recommendations: List[AssessmentRecommendation]
    metadata: Dict[str, Any] = {}
    processing_time: float
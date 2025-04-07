from fastapi import APIRouter, HTTPException, Query, Depends
from typing import List, Dict, Any, Optional
import time
import os
import sys

# Import models
from api.models.schema import RecommendationRequest, RecommendationResponse

# Import agent manager
from agents.agent_manager import AgentManager

router = APIRouter(
    prefix="/api/recommendations",
    tags=["recommendations"],
)

# Initialize the agent manager (lazy loading)
_agent_manager = None
def get_agent_manager(use_reranker: bool = True, use_llm_explanations: bool = True):
    global _agent_manager
    if _agent_manager is None:
        _agent_manager = AgentManager(
            use_reranker=use_reranker,
            use_llm_explanations=use_llm_explanations
        )
    return _agent_manager

@router.post("/", response_model=RecommendationResponse)
async def get_recommendations(request: RecommendationRequest):
    """
    Get assessment recommendations based on a natural language query.
    """
    start_time = time.time()
    
    try:
        # Get agent manager with appropriate settings
        agent_manager = get_agent_manager(
            use_reranker=request.use_reranker, 
            use_llm_explanations=request.use_llm_explanations
        )
        
        # Process the query
        results = agent_manager.process_query(
            query=request.query,
            top_k=request.top_k,
            top_n_from_k=request.top_n,
            format_as_text=False
        )
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Return response
        return RecommendationResponse(
            query=request.query,
            recommendations=results.get("recommendations", []),
            metadata=results.get("metadata", {}),
            processing_time=processing_time
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Error processing request: {str(e)}"
        )

@router.get("/query", response_model=RecommendationResponse)
async def query_recommendations(
    query: str = Query(..., description="Natural language query describing the assessment needs"),
    top_k: int = Query(10, description="Maximum number of candidates to retrieve"),
    top_n: int = Query(5, description="Maximum number of final recommendations to return"),
    use_reranker: bool = Query(True, description="Whether to use the reranker"),
    use_llm_explanations: bool = Query(True, description="Whether to generate explanations using LLM")
):
    """
    Get assessment recommendations using GET parameters.
    This endpoint is useful for quick testing or direct browser access.
    """
    request = RecommendationRequest(
        query=query,
        top_k=top_k,
        top_n=top_n,
        use_reranker=use_reranker,
        use_llm_explanations=use_llm_explanations
    )
    return await get_recommendations(request)
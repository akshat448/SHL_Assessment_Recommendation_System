from fastapi import APIRouter, HTTPException, Query, Depends
from typing import List, Dict, Any, Optional
import time
import os
import sys
import re
import logging
import concurrent.futures
import asyncio
import json

# Import models
from api.models.schema import RecommendationRequest, RecommendationResponse

# Import agent manager
from agents.agent_manager import AgentManager

# Import helpers
from api.utils.helpers import scrape_job_description, summarize_job_description

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/recommendations",
    tags=["recommendations"],
)

# Initialize the agent manager (lazy loading)
_agent_manager = None
_query_enhancer = None

def get_agent_manager(use_reranker: bool = True, use_llm_explanations: bool = True):
    global _agent_manager
    if _agent_manager is None:
        _agent_manager = AgentManager(
            use_reranker=use_reranker,
            use_llm_explanations=use_llm_explanations
        )
    return _agent_manager

# Asynchronous version of scrape_job_description to not block the event loop
async def scrape_job_description_async(url: str) -> str:
    """
    Asynchronously scrape job description from a URL.
    Runs in a thread pool to avoid blocking the event loop.
    """
    loop = asyncio.get_event_loop()
    try:
        with concurrent.futures.ThreadPoolExecutor() as pool:
            return await asyncio.wait_for(
                loop.run_in_executor(pool, scrape_job_description, url),
                timeout=15.0  # 15 second timeout
            )
    except asyncio.TimeoutError:
        logger.error(f"Timeout when scraping URL: {url}")
        raise ValueError(f"Timeout when scraping URL: {url}")
    except Exception as e:
        logger.error(f"Error when scraping URL: {url} - {str(e)}")
        raise ValueError(f"Error when scraping URL: {url} - {str(e)}")

# Asynchronous version of summarize_job_description
async def summarize_job_description_async(job_description: str) -> str:
    """
    Asynchronously summarize a job description.
    Runs in a thread pool to avoid blocking the event loop.
    """
    loop = asyncio.get_event_loop()
    try:
        with concurrent.futures.ThreadPoolExecutor() as pool:
            return await asyncio.wait_for(
                loop.run_in_executor(pool, summarize_job_description, job_description),
                timeout=20.0  # 20 second timeout
            )
    except asyncio.TimeoutError:
        logger.error("Timeout when summarizing job description")
        # Return truncated job description if summarization times out
        return job_description[:1000] + "..." if len(job_description) > 1000 else job_description
    except Exception as e:
        logger.error(f"Error when summarizing job description: {str(e)}")
        # Return truncated job description if summarization fails
        return job_description[:1000] + "..." if len(job_description) > 1000 else job_description

@router.post("/", response_model=RecommendationResponse)
async def get_recommendations(request: RecommendationRequest):
    """
    Get assessment recommendations based on a natural language query or job description URL.
    """
    start_time = time.time()
    query = request.query
    metadata = None  # Will hold the structured metadata if a JD is processed
    
    try:
        # Check if query contains a URL
        url_pattern = r"https?://(?:www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b(?:[-a-zA-Z0-9()@:%_\+.~#?&//=]*)"
        url_match = re.search(url_pattern, query)
        
        # If URL is found, try to scrape the job description
        if url_match:
            url = url_match.group(0)
            logger.info(f"Found URL in query: {url}")
            
            try:
                # Step 1: Scrape the full job description
                job_description = await scrape_job_description_async(url)
                
                if job_description:
                    logger.info(f"Successfully scraped job description ({len(job_description)} chars)")
                    
                    # Step 2: Create a summary of the job description
                    jd_summary = await summarize_job_description_async(job_description)
                    logger.info(f"Created summary of job description ({len(jd_summary)} chars)")
                    
                    # Get agent manager
                    agent_manager = get_agent_manager()
                    
                    # Step 3: Extract structured metadata from the summary
                    logger.info("Extracting structured metadata from job description summary")
                    metadata = agent_manager.query_enhancer.enhance_query(f"Job Description: {jd_summary}")
                    logger.info(f"Extracted metadata from job description: {json.dumps(metadata, indent=2)}")
                    
                    # Keep the original query (with URL) for reference
                    # But the metadata will drive the recommendations
                else:
                    logger.warning(f"No job description found at URL: {url}")
            except ValueError as e:
                logger.warning(f"Failed to process job description: {str(e)}")
                logger.info("Proceeding with original query")
        
        # Get agent manager with appropriate settings
        agent_manager = get_agent_manager(
            use_reranker=request.use_reranker, 
            use_llm_explanations=request.use_llm_explanations
        )
        
        # Process the query - if metadata was extracted, use it directly
        if metadata:
            logger.info("Using extracted metadata for recommendations")
            results = agent_manager.process_query(
                query=query,  # Original query with URL 
                metadata=metadata,  # Pre-extracted metadata
                top_k=request.top_k,
                top_n_from_k=request.top_n,
                format_as_text=False
            )
        else:
            # Normal query processing
            logger.info(f"Processing standard query (length: {len(query)} chars)")
            results = agent_manager.process_query(
                query=query,
                top_k=request.top_k,
                top_n_from_k=request.top_n,
                format_as_text=False
            )
        
        # Calculate processing time
        processing_time = time.time() - start_time
        logger.info(f"Query processed in {processing_time:.2f} seconds")
        
        # Return response
        return RecommendationResponse(
            query=request.query,  # Return original query for reference
            recommendations=results.get("recommendations", []),
            metadata=results.get("metadata", {}),
            processing_time=processing_time
        )
    
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}", exc_info=True)
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
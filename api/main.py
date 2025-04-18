from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict
import time
import os
import sys

# Add agents directory to path so we can import from it
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import agent manager
from agents.agent_manager import AgentManager

# Create FastAPI app
app = FastAPI(
    title="SHL Assessment Recommendation API",
    description="API for recommending SHL assessments based on natural language queries"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add request timing middleware
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response

# Request Models
class RecommendRequest(BaseModel):
    query: str

class DashboardRequest(BaseModel):
    query: str
    top_k: int = 25
    top_n: int = 10
    use_reranker: bool = True
    use_llm_explanations: bool = False

# Response Models
class RecommendAssessment(BaseModel):
    """Basic assessment model without explanations for /recommend endpoints"""
    url: str
    adaptive_support: str
    description: str
    duration: int
    remote_support: str
    test_type: List[str]

class DashboardAssessment(BaseModel):
    """Enhanced assessment model with explanations for /dashboard endpoint"""
    url: str
    adaptive_support: str
    description: str
    duration: int
    remote_support: str
    test_type: List[str]
    explanation: Optional[str] = None

class RecommendResponse(BaseModel):
    """Response model for /recommend endpoints"""
    recommended_assessments: List[RecommendAssessment]

class DashboardResponse(BaseModel):
    """Response model for /dashboard endpoint"""
    recommended_assessments: List[DashboardAssessment]

# Initialize agent manager (lazy loading)
_agent_manager = None
def get_agent_manager(use_reranker: bool = True, use_llm_explanations: bool = False):
    global _agent_manager
    if _agent_manager is None:
        _agent_manager = AgentManager(
            use_reranker=use_reranker,
            use_llm_explanations=use_llm_explanations
        )
    return _agent_manager

# Health Check Endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy"}

# POST /recommend endpoint
@app.post("/recommend", response_model=RecommendResponse)
async def recommend_assessments_post(request: RecommendRequest):
    if not request.query:
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    try:
        # Process the query using the agent manager
        agent_manager = get_agent_manager()
        results = agent_manager.process_query(
            query=request.query,
            top_k=25,  # Default values
            top_n_from_k=10,
            format_as_text=False
        )
        
        # Format the response
        recommended_assessments = []
        if "recommendations" in results:
            for rec in results["recommendations"]:
                # Get test types from test_type_categories or test_types
                test_types = rec.get("test_type_categories", []) or rec.get("test_types", [])
                
                # Ensure duration is an integer
                duration = 0
                if "duration_minutes" in rec and rec["duration_minutes"] is not None:
                    try:
                        duration = int(rec["duration_minutes"])
                    except (ValueError, TypeError):
                        duration = 0
                
                # Create assessment object with required fields
                assessment = {
                    "url": rec.get("assessment_url", ""),
                    "adaptive_support": rec.get("adaptive_support", "No"),
                    "description": rec.get("description", "").replace("\n", " "),
                    "duration": duration,
                    "remote_support": rec.get("remote_testing", "No"),
                    "test_type": test_types
                }
                recommended_assessments.append(assessment)
        
        return {
            "recommended_assessments": recommended_assessments
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

# GET /recommend endpoint (same functionality as POST but using query parameters)
@app.get("/recommend", response_model=RecommendResponse)
async def recommend_assessments_get(
    query: str,
    top_k: Optional[int] = 25,
    top_n: Optional[int] = 10
):
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    try:
        # Process the query using the agent manager
        agent_manager = get_agent_manager()
        results = agent_manager.process_query(
            query=query,
            top_k=top_k,
            top_n_from_k=top_n,
            format_as_text=False
        )
        
        # Format the response (same as POST endpoint)
        recommended_assessments = []
        if "recommendations" in results:
            for rec in results["recommendations"]:
                # Get test types from test_type_categories or test_types
                test_types = rec.get("test_type_categories", []) or rec.get("test_types", [])
                
                # Ensure duration is an integer
                duration = 0
                if "duration_minutes" in rec and rec["duration_minutes"] is not None:
                    try:
                        duration = int(rec["duration_minutes"])
                    except (ValueError, TypeError):
                        duration = 0
                
                # Create assessment object with required fields
                assessment = {
                    "url": rec.get("assessment_url", ""),
                    "adaptive_support": rec.get("adaptive_support", "No"),
                    "description": rec.get("description", "").replace("\n", " "),
                    "duration": duration,
                    "remote_support": rec.get("remote_testing", "No"),
                    "test_type": test_types
                }

                recommended_assessments.append(assessment)
        
        return {
            "recommended_assessments": recommended_assessments
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")


# Update the dashboard endpoint with the new response model
@app.post("/dashboard", response_model=DashboardResponse)
async def dashboard_recommendations(request: DashboardRequest):
    if not request.query:
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    try:
        # Get agent manager with specified parameters
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
        
        # Format the response
        recommended_assessments = []
        
        if "recommendations" in results:
            for rec in results["recommendations"]:
                # Get test types from test_type_categories or test_types
                test_types = rec.get("test_type_categories", []) or rec.get("test_types", [])
                
                # Ensure duration is an integer
                duration = 0
                if "duration_minutes" in rec and rec["duration_minutes"] is not None:
                    try:
                        duration = int(rec["duration_minutes"])
                    except (ValueError, TypeError):
                        duration = 0
                
                # Create assessment object with required fields
                assessment = {
                    "url": rec.get("assessment_url", ""),
                    "adaptive_support": rec.get("adaptive_support", "No"),
                    "description": rec.get("description", "").replace("\n", " "),
                    "duration": duration,
                    "remote_support": rec.get("remote_testing", "No"),
                    "test_type": test_types
                }
                
                # Only include explanation if LLM explanations are enabled
                if request.use_llm_explanations and "explanation" in rec and rec["explanation"]:
                    assessment["explanation"] = rec["explanation"].replace("**Explanation:** ", "").replace("\n", " ")
                
                recommended_assessments.append(assessment)
        
        # Return the assessments with embedded explanations
        return {"recommended_assessments": recommended_assessments}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8080, reload=True)
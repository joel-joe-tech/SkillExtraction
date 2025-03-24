from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
from job_rag_system import JobRAGSystem
import uvicorn
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Job Market RAG API",
    description="API for querying job market data using RAG (Retrieval-Augmented Generation)",
    version="1.0.0"
)

# Initialize RAG system
rag_system = JobRAGSystem()

class QuestionRequest(BaseModel):
    question: str

class CareerPathRequest(BaseModel):
    current_skills: List[str]
    target_role: str

class SkillNetworkRequest(BaseModel):
    skill_name: str

@app.get("/")
async def root():
    """Root endpoint returning API information."""
    return {
        "name": "Job Market RAG API",
        "version": "1.0.0",
        "endpoints": [
            "/ask",
            "/career-path",
            "/skill-network"
        ]
    }

@app.post("/ask")
async def ask_question(request: QuestionRequest):
    """Answer a question about jobs and careers using RAG."""
    try:
        answer = rag_system.answer_question(request.question)
        return {"answer": answer}
    except Exception as e:
        logger.error(f"Error answering question: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/career-path")
async def get_career_path(request: CareerPathRequest):
    """Get a career path recommendation based on current skills and target role."""
    try:
        path = rag_system.get_career_path(request.current_skills, request.target_role)
        return path
    except Exception as e:
        logger.error(f"Error getting career path: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/skill-network")
async def get_skill_network(request: SkillNetworkRequest):
    """Get the network of related skills and jobs for a specific skill."""
    try:
        network = rag_system.get_skill_network(request.skill_name)
        return network
    except Exception as e:
        logger.error(f"Error getting skill network: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.on_event("shutdown")
async def shutdown_event():
    """Clean up resources when the API is shut down."""
    rag_system.close()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000) 
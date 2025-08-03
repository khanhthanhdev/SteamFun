"""
Main API Router

Combines all endpoint routers for API version 1.
"""

from fastapi import APIRouter

# Create main API router
api_router = APIRouter(prefix="/api/v1")

# Placeholder for future endpoint routers
# These will be added in subsequent tasks:
# - Video generation endpoints
# - RAG system endpoints  
# - AWS integration endpoints
# - LangGraph agents endpoints

@api_router.get("/status")
async def api_status():
    """API status endpoint"""
    return {
        "status": "active",
        "version": "1.0.0",
        "endpoints": {
            "video": "Not implemented yet",
            "rag": "Not implemented yet", 
            "aws": "Not implemented yet",
            "agents": "Not implemented yet"
        }
    }
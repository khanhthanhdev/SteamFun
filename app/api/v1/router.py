"""
Main API Router

Combines all endpoint routers for API version 1.
"""

from fastapi import APIRouter
from .endpoints import video, rag, agents, aws

# Create main API router
api_router = APIRouter(prefix="/api/v1")

# Include endpoint routers
api_router.include_router(video.router)
api_router.include_router(rag.router)
api_router.include_router(agents.router)
api_router.include_router(aws.router)

@api_router.get("/status")
async def api_status():
    """API status endpoint"""
    return {
        "status": "active",
        "version": "1.0.0",
        "endpoints": {
            "video": "✅ Implemented",
            "rag": "✅ Implemented", 
            "aws": "✅ Implemented",
            "agents": "✅ Implemented"
        }
    }
import logging
from fastapi import FastAPI, HTTPException
import uvicorn

from src.api.models import QueryRequest, QueryResponse
from src.rag.engine import rag_engine
from src.core.config import config

# Configure logging
logging.basicConfig(
    level=logging.DEBUG if config.debug else logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="AI Policy Advisor",
    description="A RAG-based policy engine using llama-stack",
    version="0.1.0",
)


@app.post("/query", response_model=QueryResponse)
async def query_policy(request: QueryRequest) -> QueryResponse:
    """Query the policy engine with a natural language question."""
    try:
        logger.info(f"Received query: {request.query}")
        result = rag_engine.query(request.query)
        logger.debug(f"Query result: {result}")
        return result
    except Exception as e:
        logger.error(f"Error processing query: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Error processing query: {str(e)}"
        )


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}


def start():
    """Start the API server."""
    uvicorn.run(
        "src.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=config.debug,
    )


if __name__ == "__main__":
    start() 
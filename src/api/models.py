from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    """Request model for policy queries."""
    query: str = Field(..., description="The query about policy to check")


class SourceInfo(BaseModel):
    """Information about a source document."""
    source: str = Field(..., description="The source file path")
    text: str = Field(..., description="The relevant text from the source")


class QueryResponse(BaseModel):
    """Response model for policy queries."""
    answer: str = Field(..., description="The policy decision or answer")
    sources: List[SourceInfo] = Field(
        default_factory=list, description="Source documents used for the answer"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Metadata about the query and response"
    ) 
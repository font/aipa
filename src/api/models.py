from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    """Request model for policy queries."""
    query: str = Field(..., description="The query about policy to check")


class ManifestValidationRequest(BaseModel):
    """Request model for Kubernetes manifest validation."""
    manifest: str = Field(..., description="The Kubernetes manifest YAML content")


class PolicyViolationResponse(BaseModel):
    """Response model for policy violations."""
    rule: str = Field(..., description="The policy rule that was violated")
    manifest_path: str = Field(..., description="Path in the manifest where violation occurred")
    violation: str = Field(..., description="Description of the violation")
    severity: str = Field(default="error", description="Severity of the violation")


class ManifestValidationResponse(BaseModel):
    """Response model for manifest validation."""
    violations: List[PolicyViolationResponse] = Field(
        default_factory=list, description="List of policy violations found"
    )
    compliant: bool = Field(..., description="Whether the manifest is compliant")
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Metadata about the validation"
    )


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
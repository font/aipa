from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field

class ContentEvaluationRequest(BaseModel):
    """Request model for content evaluation."""
    content: str = Field(..., description="The content to evaluate against policies")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Additional metadata for the request")

class PolicyViolation(BaseModel):
    """Model for policy violation."""
    source: str = Field(..., description="Source of the policy violation (RAG Policy Engine or Llama Stack Safety API)")
    policy: str = Field(..., description="The policy that was violated")
    explanation: str = Field(..., description="Explanation of why the content violates the policy")

class ContentEvaluationResponse(BaseModel):
    """Response model for content evaluation."""
    is_compliant: bool = Field(..., description="Whether the content complies with all policies")
    violations: List[PolicyViolation] = Field(default_factory=list, description="List of policy violations")
    rag_evaluation: Dict[str, Any] = Field(..., description="Evaluation result from the RAG-based policy engine")
    llama_stack_evaluation: Dict[str, Any] = Field(..., description="Evaluation result from the llama-stack Safety API")

class PolicySyncRequest(BaseModel):
    """Request model for policy synchronization."""
    policy_file_path: Optional[str] = Field(default=None, description="Path to the policy document to sync")

class PolicySyncResponse(BaseModel):
    """Response model for policy synchronization."""
    success: bool = Field(..., description="Whether the policy sync was successful")
    message: str = Field(..., description="Message describing the result of the operation")
    details: Optional[Dict[str, Any]] = Field(default=None, description="Additional details about the operation")
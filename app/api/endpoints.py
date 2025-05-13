from fastapi import APIRouter, Depends, HTTPException
import logging
from typing import Dict, Any, Optional

from app.api.models import (
    ContentEvaluationRequest,
    ContentEvaluationResponse,
    PolicySyncRequest,
    PolicySyncResponse,
    PolicyViolation
)
from app.core.integrated_policy_engine import IntegratedPolicyEngine
from app.config.settings import DEFAULT_POLICY_FILE

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/api/v1", tags=["policy-engine"])

# Create a dependency for the policy engine
async def get_policy_engine() -> IntegratedPolicyEngine:
    """Dependency to get the policy engine instance."""
    try:
        return IntegratedPolicyEngine(DEFAULT_POLICY_FILE)
    except Exception as e:
        logger.error(f"Failed to initialize policy engine: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to initialize policy engine: {str(e)}")

@router.post("/evaluate", response_model=ContentEvaluationResponse)
async def evaluate_content(
    request: ContentEvaluationRequest,
    policy_engine: IntegratedPolicyEngine = Depends(get_policy_engine)
) -> ContentEvaluationResponse:
    """
    Evaluate if content complies with policies.
    """
    try:
        result = await policy_engine.evaluate_content(request.content, request.metadata)
        
        # Convert raw violations to PolicyViolation models
        violations = []
        for violation in result.get("violations", []):
            violations.append(PolicyViolation(
                source=violation.get("source", "Unknown"),
                policy=violation.get("policy", "Unknown"),
                explanation=violation.get("explanation", "")
            ))
        
        return ContentEvaluationResponse(
            is_compliant=result.get("is_compliant", False),
            violations=violations,
            rag_evaluation=result.get("rag_evaluation", {}),
            llama_stack_evaluation=result.get("llama_stack_evaluation", {})
        )
    except Exception as e:
        logger.error(f"Error evaluating content: {e}")
        raise HTTPException(status_code=500, detail=f"Error evaluating content: {str(e)}")

@router.post("/sync-policy", response_model=PolicySyncResponse)
async def sync_policy(
    request: PolicySyncRequest,
    policy_engine: IntegratedPolicyEngine = Depends(get_policy_engine)
) -> PolicySyncResponse:
    """
    Sync the policy document to llama-stack Shield resource.
    """
    try:
        # If a custom policy file path is provided, create a new policy engine instance
        if request.policy_file_path:
            policy_engine = IntegratedPolicyEngine(request.policy_file_path)
        
        result = await policy_engine.sync_policies_to_llama_stack()
        
        if result.get("error"):
            return PolicySyncResponse(
                success=False,
                message=f"Failed to sync policies: {result.get('error')}",
                details=result
            )
        
        return PolicySyncResponse(
            success=True,
            message="Policies successfully synced to llama-stack Shield resource",
            details=result
        )
    except Exception as e:
        logger.error(f"Error syncing policies: {e}")
        raise HTTPException(status_code=500, detail=f"Error syncing policies: {str(e)}") 
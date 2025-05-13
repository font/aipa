import logging
import asyncio
from typing import Dict, Any, Optional, List

from app.core.policy_engine import PolicyEngine
from app.integrations.llama_stack.safety_api import LlamaStackSafetyAPI
from app.config.settings import DEFAULT_POLICY_FILE

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class IntegratedPolicyEngine:
    """
    Integrated policy engine that combines RAG-based policy enforcement with llama-stack Safety API.
    """
    
    def __init__(self, policy_file_path: str = DEFAULT_POLICY_FILE):
        """
        Initialize the integrated policy engine.
        
        Args:
            policy_file_path: Path to the policy document
        """
        self.policy_file_path = policy_file_path
        
        # Initialize the RAG-based policy engine
        self.policy_engine = PolicyEngine(policy_file_path)
        
        # Initialize the llama-stack Safety API client
        self.safety_api = LlamaStackSafetyAPI()
        
        logger.info(f"Initialized integrated policy engine with policy file: {policy_file_path}")
    
    async def evaluate_content(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Evaluate if content complies with policies using both the RAG-based policy engine
        and the llama-stack Safety API.
        
        Args:
            content: The content to evaluate
            metadata: Additional metadata for the request
            
        Returns:
            Comprehensive evaluation result
        """
        # Run both evaluations concurrently
        rag_evaluation_task = asyncio.create_task(self._run_rag_evaluation(content))
        llama_stack_evaluation_task = asyncio.create_task(self._run_llama_stack_evaluation(content, metadata))
        
        # Wait for both evaluations to complete
        rag_evaluation = await rag_evaluation_task
        llama_stack_evaluation = await llama_stack_evaluation_task
        
        # Combine the results
        combined_result = self._combine_evaluation_results(rag_evaluation, llama_stack_evaluation)
        
        return combined_result
    
    async def _run_rag_evaluation(self, content: str) -> Dict[str, Any]:
        """Run the RAG-based policy evaluation."""
        # The evaluate_content method is synchronous, so run it in a thread pool
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.policy_engine.evaluate_content, content)
    
    async def _run_llama_stack_evaluation(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Run the llama-stack Safety API evaluation."""
        return await self.safety_api.check_content_safety(content, metadata)
    
    def _combine_evaluation_results(self, rag_result: Dict[str, Any], llama_stack_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Combine the evaluation results from both systems.
        
        Args:
            rag_result: Result from the RAG-based policy engine
            llama_stack_result: Result from the llama-stack Safety API
            
        Returns:
            Combined evaluation result
        """
        # Determine overall compliance
        rag_compliant = rag_result.get("compliance_status", "").upper() == "COMPLIANT"
        llama_stack_safe = llama_stack_result.get("is_safe", False)
        
        overall_compliant = rag_compliant and llama_stack_safe
        
        # Collect violations
        violations = []
        
        if not rag_compliant and rag_result.get("violated_policy") != "None":
            violations.append({
                "source": "RAG Policy Engine",
                "policy": rag_result.get("violated_policy", "Unknown"),
                "explanation": rag_result.get("explanation", "")
            })
        
        # Extract violations from llama-stack result if available
        if not llama_stack_safe and "violations" in llama_stack_result:
            for violation in llama_stack_result["violations"]:
                violations.append({
                    "source": "Llama Stack Safety API",
                    "policy": violation.get("policy", "Unknown"),
                    "explanation": violation.get("explanation", "")
                })
        
        return {
            "is_compliant": overall_compliant,
            "violations": violations,
            "rag_evaluation": rag_result,
            "llama_stack_evaluation": llama_stack_result
        }
    
    async def sync_policies_to_llama_stack(self) -> Dict[str, Any]:
        """
        Sync the local policy document to llama-stack Shield resource.
        
        Returns:
            Result of the policy update operation
        """
        # Format the policy data for llama-stack
        policy_data = self._format_policies_for_llama_stack()
        
        # Update the Shield resource policy
        result = await self.safety_api.update_shield_policy(policy_data)
        
        return result
    
    def _format_policies_for_llama_stack(self) -> Dict[str, Any]:
        """
        Format the local policy document for llama-stack Shield resource.
        
        Returns:
            Formatted policy data
        """
        policy_sections = self.policy_engine.policy_sections
        formatted_policies = []
        
        for section, rules in policy_sections.items():
            for rule in rules:
                formatted_policies.append({
                    "category": section,
                    "rule": rule,
                    "severity": "high"  # Default severity
                })
        
        return {
            "policies": formatted_policies,
            "metadata": {
                "source": "RAG Policy Engine",
                "version": "1.0.0"
            }
        }
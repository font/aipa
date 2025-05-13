import httpx
import json
import logging
from typing import Dict, Any, Optional

from app.config.settings import LLAMA_STACK_API_KEY, LLAMA_STACK_BASE_URL, SHIELD_RESOURCE_ID

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LlamaStackSafetyAPI:
    """
    Integration with llama-stack's Safety API for Shield resources.
    """
    
    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None, shield_id: Optional[str] = None):
        """
        Initialize the Safety API client.
        
        Args:
            api_key: The API key for llama-stack
            base_url: The base URL for the llama-stack API
            shield_id: The Shield resource ID
        """
        self.api_key = api_key or LLAMA_STACK_API_KEY
        if not self.api_key:
            raise ValueError("llama-stack API key is required")
        
        self.base_url = base_url or LLAMA_STACK_BASE_URL
        self.shield_id = shield_id or SHIELD_RESOURCE_ID
        if not self.shield_id:
            raise ValueError("Shield resource ID is required")
        
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
    
    async def check_content_safety(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Check if content is safe according to Shield policies.
        
        Args:
            content: The content to check
            metadata: Additional metadata for the request
            
        Returns:
            Safety check result
        """
        url = f"{self.base_url}/v1/shield/{self.shield_id}/check"
        
        payload = {
            "content": content,
            "metadata": metadata or {}
        }
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    url,
                    headers=self.headers,
                    json=payload,
                    timeout=30.0
                )
                
                response.raise_for_status()
                return response.json()
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error occurred: {e}")
            return {
                "error": str(e),
                "status_code": e.response.status_code,
                "is_safe": False
            }
        except Exception as e:
            logger.error(f"Error occurred: {e}")
            return {
                "error": str(e),
                "is_safe": False
            }
    
    async def update_shield_policy(self, policy_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update the Shield resource policy.
        
        Args:
            policy_data: The policy data to update
            
        Returns:
            Update result
        """
        url = f"{self.base_url}/v1/shield/{self.shield_id}/policy"
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.put(
                    url,
                    headers=self.headers,
                    json=policy_data,
                    timeout=30.0
                )
                
                response.raise_for_status()
                return response.json()
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error occurred: {e}")
            return {
                "error": str(e),
                "status_code": e.response.status_code,
                "success": False
            }
        except Exception as e:
            logger.error(f"Error occurred: {e}")
            return {
                "error": str(e),
                "success": False
            }
    
    async def get_shield_info(self) -> Dict[str, Any]:
        """
        Get information about the Shield resource.
        
        Returns:
            Shield resource information
        """
        url = f"{self.base_url}/v1/shield/{self.shield_id}"
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    url,
                    headers=self.headers,
                    timeout=30.0
                )
                
                response.raise_for_status()
                return response.json()
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error occurred: {e}")
            return {
                "error": str(e),
                "status_code": e.response.status_code
            }
        except Exception as e:
            logger.error(f"Error occurred: {e}")
            return {
                "error": str(e)
            } 
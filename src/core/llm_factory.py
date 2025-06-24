"""LLM Factory for creating different LLM providers."""

import logging
from typing import Optional

from llama_index.core.llms import LLM
from llama_index.core.settings import Settings

from src.core.config import config, LLMProvider

logger = logging.getLogger(__name__)


class LLMFactory:
    """Factory for creating LLM instances based on provider configuration."""
    
    @staticmethod
    def create_llm(provider: Optional[LLMProvider] = None) -> LLM:
        """Create an LLM instance based on the provider.
        
        Args:
            provider: LLM provider to use. If None, uses config.llm.provider
            
        Returns:
            LLM instance
            
        Raises:
            ValueError: If provider is not supported or configuration is invalid
        """
        provider = provider or config.llm.provider
        
        logger.info(f"Creating LLM with provider: {provider}")
        
        if provider == "anthropic":
            return LLMFactory._create_anthropic_llm()
        elif provider == "openai":
            return LLMFactory._create_openai_llm()
        elif provider == "llamastack":
            return LLMFactory._create_llamastack_llm()
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")
    
    @staticmethod
    def _create_anthropic_llm() -> LLM:
        """Create Anthropic LLM instance."""
        try:
            from llama_index.llms.anthropic import Anthropic
            
            if not config.anthropic.api_key:
                raise ValueError("ANTHROPIC_API_KEY environment variable is required for Anthropic provider")
            
            llm = Anthropic(
                model=config.anthropic.model_name,
                api_key=config.anthropic.api_key,
                temperature=config.llm.temperature,
                max_tokens=config.llm.max_tokens,
            )
            
            logger.info(f"Created Anthropic LLM with model: {config.anthropic.model_name}")
            return llm
            
        except ImportError as e:
            logger.error(f"Failed to import Anthropic LLM: {e}")
            raise ValueError("llama-index-llms-anthropic package is required for Anthropic provider")
    
    @staticmethod
    def _create_openai_llm() -> LLM:
        """Create OpenAI LLM instance."""
        try:
            from llama_index.llms.openai import OpenAI
            
            if not config.openai.api_key:
                raise ValueError("OPENAI_API_KEY environment variable is required for OpenAI provider")
            
            llm = OpenAI(
                model=config.openai.model_name,
                api_key=config.openai.api_key,
                temperature=config.llm.temperature,
                max_tokens=config.llm.max_tokens,
            )
            
            logger.info(f"Created OpenAI LLM with model: {config.openai.model_name}")
            return llm
            
        except ImportError as e:
            logger.error(f"Failed to import OpenAI LLM: {e}")
            raise ValueError("llama-index-llms-openai package is required for OpenAI provider")
    
    @staticmethod
    def _create_llamastack_llm() -> LLM:
        """Create LlamaStack LLM instance."""
        try:
            from llama_stack_client import LlamaStackClient
            from src.rag.engine import LlamaStackLLM
            
            # Initialize LlamaStackClient with timeout configuration
            client = LlamaStackClient(
                base_url=config.llamastack.api_url,
                timeout=30.0,  # 30 second timeout
            )
            
            llm = LlamaStackLLM(
                client=client,
                model_id=config.llamastack.model_name
            )
            
            logger.info(f"Created LlamaStack LLM with model: {config.llamastack.model_name}")
            return llm
            
        except ImportError as e:
            logger.error(f"Failed to import LlamaStack client: {e}")
            raise ValueError("llama-stack-client package is required for LlamaStack provider")
    
    @staticmethod
    def setup_global_llm(provider: Optional[LLMProvider] = None) -> LLM:
        """Set up global LLM settings.
        
        Args:
            provider: LLM provider to use. If None, uses config.llm.provider
            
        Returns:
            LLM instance that was set globally
        """
        llm = LLMFactory.create_llm(provider)
        
        # Configure global settings
        Settings.llm = llm
        Settings.chunk_size = config.rag.chunk_size
        Settings.chunk_overlap = config.rag.chunk_overlap
        
        return llm


# Create a singleton factory instance
llm_factory = LLMFactory() 
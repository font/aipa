from typing import List, Dict, Any, Optional, AsyncGenerator, Generator
import logging

from llama_index.core import Document, VectorStoreIndex
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.core.settings import Settings
from llama_index.core.llms import LLM, ChatMessage, ChatResponse, CompletionResponse, LLMMetadata
from llama_stack_client import LlamaStackClient
from llama_stack_client.types import Model
from pydantic import Field

from src.core.config import config
from src.policy.loader import policy_loader

logger = logging.getLogger(__name__)


class LlamaStackLLM(LLM):
    """Wrapper for LlamaStack LLM to work with LlamaIndex."""
    
    client: LlamaStackClient = Field(description="LlamaStackClient instance")
    model_id: str = Field(description="Model ID to use")
    
    def __init__(self, client: LlamaStackClient, model_id: str):
        """Initialize the LlamaStack LLM wrapper.
        
        Args:
            client: LlamaStackClient instance
            model_id: Model ID to use
        """
        super().__init__(client=client, model_id=model_id)
    
    @property
    def metadata(self) -> LLMMetadata:
        """Get LLM metadata."""
        return LLMMetadata(
            model_name=self.model_id,
            is_chat_model=True,
            is_function_calling_model=False,
            context_window=4096,  # Default value, adjust based on your model
            num_output=2048,  # Default value, adjust based on your model
        )
    
    def complete(self, prompt: str, **kwargs) -> CompletionResponse:
        """Complete the prompt using LlamaStack.
        
        Args:
            prompt: The prompt to complete
            **kwargs: Additional arguments
            
        Returns:
            The completed text
        """
        response = self.client.inference.chat_completion(
            model_id=self.model_id,
            messages=[
                {"role": "system", "content": "You are a helpful AI assistant."},
                {"role": "user", "content": prompt}
            ]
        )
        return CompletionResponse(text=response.choices[0].message.content)
    
    def stream_complete(self, prompt: str, **kwargs) -> Generator[CompletionResponse, None, None]:
        """Stream complete the prompt using LlamaStack.
        
        Args:
            prompt: The prompt to complete
            **kwargs: Additional arguments
            
        Yields:
            The completed text chunks
        """
        response = self.client.inference.chat_completion(
            model_id=self.model_id,
            messages=[
                {"role": "system", "content": "You are a helpful AI assistant."},
                {"role": "user", "content": prompt}
            ],
            stream=True
        )
        for chunk in response:
            if chunk.choices[0].delta.content:
                yield CompletionResponse(text=chunk.choices[0].delta.content)
    
    def chat(self, messages: List[ChatMessage], **kwargs) -> ChatResponse:
        """Chat with the model using LlamaStack.
        
        Args:
            messages: List of chat messages
            **kwargs: Additional arguments
            
        Returns:
            The chat response
        """
        response = self.client.inference.chat_completion(
            model_id=self.model_id,
            messages=[{"role": msg.role.value, "content": msg.content} for msg in messages]
        )
        return ChatResponse(message=ChatMessage(role="assistant", content=response.choices[0].message.content))
    
    def stream_chat(self, messages: List[ChatMessage], **kwargs) -> Generator[ChatResponse, None, None]:
        """Stream chat with the model using LlamaStack.
        
        Args:
            messages: List of chat messages
            **kwargs: Additional arguments
            
        Yields:
            The chat response chunks
        """
        response = self.client.inference.chat_completion(
            model_id=self.model_id,
            messages=[{"role": msg.role.value, "content": msg.content} for msg in messages],
            stream=True
        )
        for chunk in response:
            if chunk.choices[0].delta.content:
                yield ChatResponse(message=ChatMessage(role="assistant", content=chunk.choices[0].delta.content))
    
    async def acomplete(self, prompt: str, **kwargs) -> CompletionResponse:
        """Async complete the prompt using LlamaStack.
        
        Note: LlamaStack's client doesn't have native async support, so this is a wrapper
        around the synchronous complete method. For true async support, we would need to
        implement this using an async HTTP client.
        
        Args:
            prompt: The prompt to complete
            **kwargs: Additional arguments
            
        Returns:
            The completed text
        """
        return self.complete(prompt, **kwargs)
    
    async def astream_complete(self, prompt: str, **kwargs) -> AsyncGenerator[CompletionResponse, None]:
        """Async stream complete the prompt using LlamaStack.
        
        Note: LlamaStack's client doesn't have native async support, so this is a wrapper
        around the synchronous stream_complete method. For true async support, we would need to
        implement this using an async HTTP client.
        
        Args:
            prompt: The prompt to complete
            **kwargs: Additional arguments
            
        Yields:
            The completed text chunks
        """
        for response in self.stream_complete(prompt, **kwargs):
            yield response
    
    async def achat(self, messages: List[ChatMessage], **kwargs) -> ChatResponse:
        """Async chat with the model using LlamaStack.
        
        Note: LlamaStack's client doesn't have native async support, so this is a wrapper
        around the synchronous chat method. For true async support, we would need to
        implement this using an async HTTP client.
        
        Args:
            messages: List of chat messages
            **kwargs: Additional arguments
            
        Returns:
            The chat response
        """
        return self.chat(messages, **kwargs)
    
    async def astream_chat(self, messages: List[ChatMessage], **kwargs) -> AsyncGenerator[ChatResponse, None]:
        """Async stream chat with the model using LlamaStack.
        
        Note: LlamaStack's client doesn't have native async support, so this is a wrapper
        around the synchronous stream_chat method. For true async support, we would need to
        implement this using an async HTTP client.
        
        Args:
            messages: List of chat messages
            **kwargs: Additional arguments
            
        Yields:
            The chat response chunks
        """
        for response in self.stream_chat(messages, **kwargs):
            yield response


class RagEngine:
    """Retrieval-Augmented Generation engine for policy queries."""
    
    def __init__(self, llm: Optional[LLM] = None):
        """Initialize the RAG engine.
        
        Args:
            llm: LLM instance to use. If None, will use llama-stack-client.
        """
        self.llm = llm
        self.index = None
        self._setup_llm()
    
    def _setup_llm(self):
        """Set up the LLM using LlamaStackClient."""
        try:
            # Initialize LlamaStackClient
            client = LlamaStackClient(
                base_url=config.llama.api_url,
            )
            
            # Create LLM instance using the inference API
            self.llm = self.llm or LlamaStackLLM(
                client=client,
                model_id=config.llama.model_name
            )
            
            # Configure global settings
            Settings.llm = self.llm
            Settings.chunk_size = config.rag.chunk_size
            Settings.chunk_overlap = config.rag.chunk_overlap
            
        except ImportError as e:
            logger.error(f"Failed to import llama-stack-client: {e}")
            raise
    
    def build_index(self):
        """Build the index from policy documents."""
        # Load policy documents
        policy_docs = policy_loader.load_policies()
        
        if not policy_docs:
            logger.warning("No policy documents found.")
            return
        
        # Convert to LlamaIndex documents
        documents = [
            Document(text=doc["content"], metadata={"source": doc["source"]})
            for doc in policy_docs
        ]
        
        # Create node parser
        node_parser = SimpleNodeParser.from_defaults(
            chunk_size=config.rag.chunk_size,
            chunk_overlap=config.rag.chunk_overlap,
        )
        
        # Build the index
        self.index = VectorStoreIndex.from_documents(
            documents,
            node_parser=node_parser,
        )
        
        logger.info(f"Built index with {len(documents)} policy documents.")
    
    def query(self, query_text: str) -> Dict[str, Any]:
        """Query the policy engine.
        
        Args:
            query_text: The query string.
            
        Returns:
            A dictionary containing the response, relevant documents, and metadata.
        """
        if not self.index:
            self.build_index()
        
        # Create query engine
        query_engine = self.index.as_query_engine(
            similarity_top_k=config.rag.similarity_top_k,
        )
        
        # Execute query
        response = query_engine.query(query_text)
        
        # Format result
        result = {
            "answer": str(response),
            "sources": [
                {"source": node.metadata.get("source", "unknown"), "text": node.get_text()}
                for node in getattr(response, "source_nodes", [])
            ],
            "metadata": {
                "model": config.llama.model_name,
                "provider": config.llama.provider,
            }
        }
        
        return result


# Create a singleton RAG engine instance
rag_engine = RagEngine() 
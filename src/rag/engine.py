from typing import List, Dict, Any, Optional, AsyncGenerator, Generator, Union
import logging
import yaml

from llama_index.core import Document, VectorStoreIndex
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.core.settings import Settings
from llama_index.core.llms import LLM, ChatMessage, ChatResponse, CompletionResponse, LLMMetadata
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_stack_client import LlamaStackClient
from llama_stack_client.types import Model
from pydantic import Field, BaseModel

from src.core.config import config
from src.policy.loader import policy_loader

logger = logging.getLogger(__name__)

class PolicyViolation(BaseModel):
    """Model for policy violations."""
    rule: str
    manifest_path: str
    violation: str
    severity: str = "error"

class K8sPolicyEnforcer:
    """Enforces Kubernetes manifest policies using natural language rules."""
    
    def __init__(self, rag_engine: 'RagEngine'):
        """Initialize the K8s policy enforcer.
        
        Args:
            rag_engine: RAG engine instance to use for policy queries
        """
        self.rag_engine = rag_engine
        self.policy_index = None
        self._build_policy_index()
    
    def _build_policy_index(self):
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
        self.policy_index = VectorStoreIndex.from_documents(
            documents,
            node_parser=node_parser,
        )
        
        logger.info(f"Built policy index with {len(documents)} policy documents.")
    
    def _parse_manifest(self, manifest: str) -> Dict[str, Any]:
        """Parse a Kubernetes manifest.
        
        Args:
            manifest: YAML manifest string
            
        Returns:
            Parsed manifest as a dictionary
        """
        try:
            return yaml.safe_load(manifest)
        except yaml.YAMLError as e:
            logger.error(f"Failed to parse manifest: {e}")
            raise ValueError(f"Invalid YAML manifest: {e}")
    
    def _format_manifest_for_prompt(self, manifest: Dict[str, Any]) -> str:
        """Format manifest for inclusion in prompt.
        
        Args:
            manifest: Parsed manifest dictionary
            
        Returns:
            Formatted manifest string
        """
        return yaml.dump(manifest, default_flow_style=False)
    
    def enforce_policy(self, manifest: Union[str, Dict[str, Any]]) -> List[PolicyViolation]:
        """Enforce policy on a Kubernetes manifest.
        
        Args:
            manifest: Kubernetes manifest as YAML string or parsed dictionary
            
        Returns:
            List of policy violations found
        """
        if isinstance(manifest, str):
            manifest = self._parse_manifest(manifest)
        
        formatted_manifest = self._format_manifest_for_prompt(manifest)
        
        # Create query engine
        query_engine = self.policy_index.as_query_engine(
            similarity_top_k=config.rag.similarity_top_k,
        )
        
        # Construct prompt for policy enforcement
        prompt = f"""Analyze this Kubernetes manifest against our company policies:

{formatted_manifest}

Please check if this manifest violates any of our policies. For each violation found, provide:
1. The specific policy rule that was violated
2. The exact part of the manifest that violates the rule
3. The severity of the violation (error or warning)

If no violations are found, respond with "No policy violations found."

Format your response as a list of violations, one per line, with each violation containing:
- Rule: [policy rule]
- Violation: [description of violation]
- Severity: [error/warning]
"""
        
        # Execute query
        response = query_engine.query(prompt)
        
        # Parse violations from response
        violations = []
        if "No policy violations found" not in str(response):
            # Parse the response to extract violations
            # This is a simple implementation - you might want to make it more robust
            lines = str(response).split('\n')
            current_violation = {}
            
            for line in lines:
                if line.startswith('- Rule:'):
                    if current_violation:
                        violations.append(PolicyViolation(**current_violation))
                    current_violation = {'rule': line[7:].strip()}
                elif line.startswith('- Violation:'):
                    current_violation['violation'] = line[12:].strip()
                elif line.startswith('- Severity:'):
                    current_violation['severity'] = line[11:].strip()
                    current_violation['manifest_path'] = 'root'  # You might want to make this more specific
            
            if current_violation:
                violations.append(PolicyViolation(**current_violation))
        
        return violations


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
        # Handle the response format from LlamaStack
        if hasattr(response, 'message'):
            content = response.message.content
        elif hasattr(response, 'content'):
            content = response.content
        else:
            raise ValueError("Unexpected response format from LlamaStack")
            
        return CompletionResponse(text=content)
    
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
            if hasattr(chunk, 'delta') and hasattr(chunk.delta, 'content') and chunk.delta.content:
                yield CompletionResponse(text=chunk.delta.content)
            elif hasattr(chunk, 'content') and chunk.content:
                yield CompletionResponse(text=chunk.content)
    
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
        
        # Debug logging
        logger.debug(f"LlamaStack response type: {type(response)}")
        logger.debug(f"LlamaStack response attributes: {dir(response)}")
        
        # Handle the response format from LlamaStack
        try:
            if hasattr(response, 'completion_message'):
                content = response.completion_message.content
            elif hasattr(response, 'message'):
                content = response.message.content
            elif hasattr(response, 'content'):
                content = response.content
            elif hasattr(response, 'text'):
                content = response.text
            elif hasattr(response, 'response'):
                content = response.response
            elif isinstance(response, str):
                content = response
            elif isinstance(response, dict):
                if 'completion_message' in response:
                    content = response['completion_message'].get('content', '')
                elif 'message' in response:
                    content = response['message'].get('content', '')
                elif 'content' in response:
                    content = response['content']
                elif 'text' in response:
                    content = response['text']
                elif 'response' in response:
                    content = response['response']
                else:
                    raise ValueError(f"Unexpected response dict format: {response}")
            else:
                raise ValueError(f"Unexpected response type: {type(response)}")
                
            if not content:
                raise ValueError("Empty response content")
                
            return ChatResponse(message=ChatMessage(role="assistant", content=content))
            
        except Exception as e:
            logger.error(f"Error processing LlamaStack response: {str(e)}")
            logger.error(f"Response object: {response}")
            raise ValueError(f"Failed to process LlamaStack response: {str(e)}")
    
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
            try:
                content = None
                if hasattr(chunk, 'completion_message'):
                    content = chunk.completion_message.content
                elif hasattr(chunk, 'delta') and hasattr(chunk.delta, 'content'):
                    content = chunk.delta.content
                elif hasattr(chunk, 'content'):
                    content = chunk.content
                elif hasattr(chunk, 'text'):
                    content = chunk.text
                elif hasattr(chunk, 'response'):
                    content = chunk.response
                elif isinstance(chunk, str):
                    content = chunk
                elif isinstance(chunk, dict):
                    if 'completion_message' in chunk:
                        content = chunk['completion_message'].get('content', '')
                    elif 'delta' in chunk and 'content' in chunk['delta']:
                        content = chunk['delta']['content']
                    elif 'content' in chunk:
                        content = chunk['content']
                    elif 'text' in chunk:
                        content = chunk['text']
                    elif 'response' in chunk:
                        content = chunk['response']
                
                if content:
                    yield ChatResponse(message=ChatMessage(role="assistant", content=content))
                    
            except Exception as e:
                logger.error(f"Error processing LlamaStack stream chunk: {str(e)}")
                logger.error(f"Chunk object: {chunk}")
                continue
    
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
        self._setup_embeddings()
    
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

    def _setup_embeddings(self):
        """Set up the embedding model."""
        try:
            # Initialize HuggingFace embedding model
            embed_model = HuggingFaceEmbedding(
                model_name="BAAI/bge-small-en-v1.5"
            )
            
            # Configure global settings
            Settings.embed_model = embed_model
            
        except ImportError as e:
            logger.error(f"Failed to import HuggingFace embedding model: {e}")
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
        
        # Format result and deduplicate sources
        seen_texts = set()
        unique_sources = []
        
        for node in getattr(response, "source_nodes", []):
            source_text = node.get_text()
            if source_text not in seen_texts:
                seen_texts.add(source_text)
                unique_sources.append({
                    "source": node.metadata.get("source", "unknown"),
                    "text": source_text
                })
        
        result = {
            "answer": str(response),
            "sources": unique_sources,
            "metadata": {
                "model": config.llama.model_name,
                "provider": config.llama.provider,
            }
        }
        
        return result


# Create a singleton RAG engine instance
rag_engine = RagEngine() 
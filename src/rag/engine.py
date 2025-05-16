from typing import List, Dict, Any, Optional
import logging

from llama_index import ServiceContext, VectorStoreIndex, Document
from llama_index.llms import LLM
from llama_index.node_parser import SimpleNodeParser

from src.core.config import config
from src.policy.loader import policy_loader

logger = logging.getLogger(__name__)


class RagEngine:
    """Retrieval-Augmented Generation engine for policy queries."""
    
    def __init__(self, llm: Optional[LLM] = None):
        """Initialize the RAG engine.
        
        Args:
            llm: LLM instance to use. If None, will use llama-stack-client.
        """
        self.llm = llm
        self.index = None
        self.service_context = None
        self._setup_service_context()
    
    def _setup_service_context(self):
        """Set up the service context for LlamaIndex."""
        try:
            from llama_stack_client import get_llm
            
            # Use llama-stack-client to create an LLM
            self.llm = self.llm or get_llm(
                model_name=config.llama.model_name,
                provider=config.llama.provider,
                temperature=config.llama.temperature,
                max_tokens=config.llama.max_tokens,
            )
            
            # Create node parser
            node_parser = SimpleNodeParser.from_defaults(
                chunk_size=config.rag.chunk_size,
                chunk_overlap=config.rag.chunk_overlap,
            )
            
            # Create service context
            self.service_context = ServiceContext.from_defaults(
                llm=self.llm,
                node_parser=node_parser,
            )
            
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
        
        # Build the index
        self.index = VectorStoreIndex.from_documents(
            documents, service_context=self.service_context
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
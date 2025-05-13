import os
from typing import Dict, List, Any, Optional
import logging

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain_community.llms import LlamaCpp
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain

# Import llama-stack-client
from llama_stack_client import LlamaClient
from llama_stack_client.langchain import LlamaLLM, LlamaEmbeddings

from app.config.settings import (
    OPENAI_API_KEY, LLM_MODEL, EMBEDDING_MODEL, VECTOR_DB_PATH,
    LLM_PROVIDER, EMBEDDING_PROVIDER, LLAMA_MODEL_PATH, LLAMA_N_CTX,
    LLAMA_N_GPU_LAYERS, SENTENCE_TRANSFORMER_MODEL,
    LLAMA_STACK_CLIENT_URL, LLAMA_STACK_CLIENT_PROVIDER, LLAMA_STACK_CLIENT_MODEL,
    LLAMA_STACK_CLIENT_TIMEOUT, LLAMA_STACK_CLIENT_EMBEDDING_MODEL
)
from app.utils.policy_loader import load_policy_document, parse_policy_sections

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PolicyEngine:
    """
    AI-based policy engine that enforces natural language policies using RAG.
    """
    
    def __init__(self, policy_file_path: str, vector_db_path: Optional[str] = None):
        """
        Initialize the policy engine.
        
        Args:
            policy_file_path: Path to the policy document
            vector_db_path: Path to store the vector database
        """
        self.policy_file_path = policy_file_path
        self.vector_db_path = vector_db_path or VECTOR_DB_PATH
        
        # Load policy document
        self.policy_text = load_policy_document(policy_file_path)
        self.policy_sections = parse_policy_sections(self.policy_text)
        
        # Initialize llama-stack client if needed
        self.llama_client = None
        if LLM_PROVIDER == "llama_stack" or EMBEDDING_PROVIDER == "llama_stack":
            self.llama_client = LlamaClient(
                base_url=LLAMA_STACK_CLIENT_URL,
                provider=LLAMA_STACK_CLIENT_PROVIDER,
                timeout=LLAMA_STACK_CLIENT_TIMEOUT
            )
        
        # Initialize components
        self.embeddings = self._initialize_embeddings()
        self.llm = self._initialize_llm()
        
        # Create vector store
        self._create_vector_store()
        
        # Set up the retriever with contextual compression
        compressor = LLMChainExtractor.from_llm(self.llm)
        self.retriever = ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=self.vector_store.as_retriever(search_kwargs={"k": 5})
        )
        
        # Create evaluation chain
        self.evaluation_chain = self._create_evaluation_chain()
    
    def _initialize_embeddings(self):
        """Initialize the embedding model based on configuration."""
        if EMBEDDING_PROVIDER == "openai":
            logger.info(f"Using OpenAI embeddings model: {EMBEDDING_MODEL}")
            return OpenAIEmbeddings(
                model=EMBEDDING_MODEL,
                openai_api_key=OPENAI_API_KEY
            )
        elif EMBEDDING_PROVIDER == "sentence_transformers":
            logger.info(f"Using Sentence Transformers embeddings model: {SENTENCE_TRANSFORMER_MODEL}")
            return HuggingFaceEmbeddings(model_name=SENTENCE_TRANSFORMER_MODEL)
        elif EMBEDDING_PROVIDER == "llama_stack":
            logger.info(f"Using llama-stack embeddings model: {LLAMA_STACK_CLIENT_EMBEDDING_MODEL}")
            return LlamaEmbeddings(
                client=self.llama_client,
                model=LLAMA_STACK_CLIENT_EMBEDDING_MODEL
            )
        else:
            raise ValueError(f"Unsupported embedding provider: {EMBEDDING_PROVIDER}")
    
    def _initialize_llm(self):
        """Initialize the LLM based on configuration."""
        if LLM_PROVIDER == "openai":
            logger.info(f"Using OpenAI LLM model: {LLM_MODEL}")
            return ChatOpenAI(
                model=LLM_MODEL,
                temperature=0,
                openai_api_key=OPENAI_API_KEY
            )
        elif LLM_PROVIDER == "llama":
            logger.info(f"Using Llama model from: {LLAMA_MODEL_PATH}")
            
            # Check if model file exists
            if not os.path.exists(LLAMA_MODEL_PATH):
                raise FileNotFoundError(f"Llama model file not found: {LLAMA_MODEL_PATH}")
            
            return LlamaCpp(
                model_path=LLAMA_MODEL_PATH,
                temperature=0,
                max_tokens=2048,
                n_ctx=LLAMA_N_CTX,
                n_gpu_layers=LLAMA_N_GPU_LAYERS,
                verbose=True
            )
        elif LLM_PROVIDER == "llama_stack":
            logger.info(f"Using llama-stack LLM model: {LLAMA_STACK_CLIENT_MODEL} via {LLAMA_STACK_CLIENT_PROVIDER}")
            return LlamaLLM(
                client=self.llama_client,
                model=LLAMA_STACK_CLIENT_MODEL,
                temperature=0
            )
        else:
            raise ValueError(f"Unsupported LLM provider: {LLM_PROVIDER}")
    
    def _create_vector_store(self):
        """Create or load the vector store from policy documents."""
        # Split the policy text into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )
        chunks = text_splitter.split_text(self.policy_text)
        
        # Check if vector store exists
        if os.path.exists(self.vector_db_path):
            logger.info(f"Loading existing vector store from {self.vector_db_path}")
            self.vector_store = Chroma(
                persist_directory=self.vector_db_path,
                embedding_function=self.embeddings
            )
        else:
            logger.info(f"Creating new vector store at {self.vector_db_path}")
            os.makedirs(os.path.dirname(self.vector_db_path), exist_ok=True)
            self.vector_store = Chroma.from_texts(
                chunks,
                self.embeddings,
                persist_directory=self.vector_db_path
            )
            self.vector_store.persist()
    
    def _create_evaluation_chain(self) -> LLMChain:
        """Create the LLM chain for policy evaluation."""
        template = """You are an AI policy enforcement system. Your job is to evaluate if content complies with the organization's policies.

POLICY CONTEXT:
{context}

CONTENT TO EVALUATE:
{content}

Evaluate if the content violates any of the policies in the context. 
If it does, specify which policy it violates and why.
If it doesn't violate any policies, state that the content complies with all policies.

Your evaluation should be structured as follows:
- Compliance: [COMPLIANT/NON-COMPLIANT]
- Violated Policy: [Policy section and rule that was violated, or "None" if compliant]
- Explanation: [Detailed explanation of why the content violates the policy or why it complies]

EVALUATION:"""
        
        prompt = ChatPromptTemplate.from_template(template)
        return LLMChain(llm=self.llm, prompt=prompt)
    
    def evaluate_content(self, content: str) -> Dict[str, Any]:
        """
        Evaluate if content complies with policies.
        
        Args:
            content: The content to evaluate
            
        Returns:
            Evaluation result with compliance status, violated policy, and explanation
        """
        # Retrieve relevant policy documents
        retrieved_docs = self.retriever.get_relevant_documents(content)
        context = "\n\n".join([doc.page_content for doc in retrieved_docs])
        
        # Evaluate content against policies
        result = self.evaluation_chain.invoke({
            "context": context,
            "content": content
        })
        
        # Parse the result
        evaluation_text = result["text"]
        
        # Extract compliance status
        compliance_status = "COMPLIANT"
        violated_policy = "None"
        explanation = ""
        
        for line in evaluation_text.split("\n"):
            line = line.strip()
            if line.startswith("- Compliance:"):
                compliance_status = line.split(":", 1)[1].strip()
            elif line.startswith("- Violated Policy:"):
                violated_policy = line.split(":", 1)[1].strip()
            elif line.startswith("- Explanation:"):
                explanation = line.split(":", 1)[1].strip()
        
        return {
            "compliance_status": compliance_status,
            "violated_policy": violated_policy,
            "explanation": explanation,
            "relevant_policies": context
        } 
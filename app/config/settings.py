import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Base directory
BASE_DIR = Path(__file__).resolve().parent.parent.parent

# API Keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
LLAMA_STACK_API_KEY = os.getenv("LLAMA_STACK_API_KEY")

# Policy settings
POLICY_DIR = os.path.join(BASE_DIR, "data", "policies")
DEFAULT_POLICY_FILE = os.path.join(POLICY_DIR, "sample_policy.txt")

# Vector DB settings
VECTOR_DB_PATH = os.path.join(BASE_DIR, "data", "vectordb")

# LLM settings
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")  # Options: "openai", "llama", "llama_stack"
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

# Llama model settings
LLAMA_MODEL_PATH = os.getenv("LLAMA_MODEL_PATH", "models/llama-3-8b-instruct.gguf")
LLAMA_N_CTX = int(os.getenv("LLAMA_N_CTX", "4096"))
LLAMA_N_GPU_LAYERS = int(os.getenv("LLAMA_N_GPU_LAYERS", "0"))

# Embedding model settings
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "openai")  # Options: "openai", "sentence_transformers"
SENTENCE_TRANSFORMER_MODEL = os.getenv("SENTENCE_TRANSFORMER_MODEL", "all-MiniLM-L6-v2")

# Llama Stack client settings
LLAMA_STACK_CLIENT_URL = os.getenv("LLAMA_STACK_CLIENT_URL", "http://localhost:11434")
LLAMA_STACK_CLIENT_PROVIDER = os.getenv("LLAMA_STACK_CLIENT_PROVIDER", "ollama")  # Options: "ollama", "vllm"
LLAMA_STACK_CLIENT_MODEL = os.getenv("LLAMA_STACK_CLIENT_MODEL", "llama3")
LLAMA_STACK_CLIENT_TIMEOUT = int(os.getenv("LLAMA_STACK_CLIENT_TIMEOUT", "120"))
LLAMA_STACK_CLIENT_EMBEDDING_MODEL = os.getenv("LLAMA_STACK_CLIENT_EMBEDDING_MODEL", "nomic-embed-text")

# Llama Stack integration
LLAMA_STACK_BASE_URL = os.getenv("LLAMA_STACK_BASE_URL", "https://api.llama-stack.com")
SHIELD_RESOURCE_ID = os.getenv("SHIELD_RESOURCE_ID")

# API settings
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))
API_DEBUG = os.getenv("API_DEBUG", "False").lower() == "true" 
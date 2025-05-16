import os
from pathlib import Path
from typing import Optional

from pydantic import BaseModel
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Base directory
BASE_DIR = Path(__file__).parent.parent.parent


class LlamaConfig(BaseModel):
    """Configuration for Llama Stack."""
    api_url: str = os.getenv("LLAMA_API_URL", "http://localhost:8000")
    model_name: str = os.getenv("LLAMA_MODEL", "llama2")
    provider: str = os.getenv("LLAMA_PROVIDER", "ollama")
    temperature: float = float(os.getenv("LLAMA_TEMPERATURE", "0.1"))
    max_tokens: int = int(os.getenv("LLAMA_MAX_TOKENS", "1024"))


class RagConfig(BaseModel):
    """Configuration for RAG."""
    chunk_size: int = int(os.getenv("RAG_CHUNK_SIZE", "512"))
    chunk_overlap: int = int(os.getenv("RAG_CHUNK_OVERLAP", "50"))
    similarity_top_k: int = int(os.getenv("RAG_TOP_K", "3"))


class PolicyConfig(BaseModel):
    """Configuration for Policy documents."""
    policy_dir: Path = Path(os.getenv("POLICY_DIR", str(BASE_DIR / "data")))


class Config(BaseModel):
    """Main configuration class."""
    llama: LlamaConfig = LlamaConfig()
    rag: RagConfig = RagConfig()
    policy: PolicyConfig = PolicyConfig()
    debug: bool = os.getenv("DEBUG", "False").lower() in ("true", "1", "t")


# Create a singleton config instance
config = Config() 
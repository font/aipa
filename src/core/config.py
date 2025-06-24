import os
from pathlib import Path
from typing import Optional, Literal

from pydantic import BaseModel
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Base directory
BASE_DIR = Path(__file__).parent.parent.parent

LLMProvider = Literal["llamastack", "anthropic", "openai"]


class LLMConfig(BaseModel):
    """Configuration for LLM providers."""
    provider: LLMProvider = os.getenv("LLM_PROVIDER", "llamastack")
    temperature: float = float(os.getenv("LLM_TEMPERATURE", "0.1"))
    max_tokens: int = int(os.getenv("LLM_MAX_TOKENS", "1024"))


class LlamaStackConfig(BaseModel):
    """Configuration for Llama Stack."""
    api_url: str = os.getenv("LLAMASTACK_API_URL", "http://localhost:8000")
    model_name: str = os.getenv("LLAMASTACK_MODEL", "llama2")


class AnthropicConfig(BaseModel):
    """Configuration for Anthropic."""
    api_key: str = os.getenv("ANTHROPIC_API_KEY", "")
    model_name: str = os.getenv("ANTHROPIC_MODEL", "claude-3-haiku-20240307")


class OpenAIConfig(BaseModel):
    """Configuration for OpenAI."""
    api_key: str = os.getenv("OPENAI_API_KEY", "")
    model_name: str = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")


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
    llm: LLMConfig = LLMConfig()
    llamastack: LlamaStackConfig = LlamaStackConfig()
    anthropic: AnthropicConfig = AnthropicConfig()
    openai: OpenAIConfig = OpenAIConfig()
    rag: RagConfig = RagConfig()
    policy: PolicyConfig = PolicyConfig()
    debug: bool = os.getenv("DEBUG", "False").lower() in ("true", "1", "t")


# Create a singleton config instance
config = Config() 
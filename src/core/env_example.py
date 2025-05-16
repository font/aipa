"""
Sample environment variables for configuration.

To use, create a .env file in the project root with the following variables:

# Llama Stack Configuration
LLAMA_API_URL=http://localhost:11434  # URL for the llama-stack API
LLAMA_MODEL=llama2                    # Model name to use
LLAMA_PROVIDER=ollama                 # Provider (ollama, vllm, etc.)
LLAMA_TEMPERATURE=0.1                 # Temperature for generation
LLAMA_MAX_TOKENS=1024                 # Maximum tokens to generate

# RAG Configuration
RAG_CHUNK_SIZE=512                   # Size of chunks for indexing
RAG_CHUNK_OVERLAP=50                 # Overlap between chunks
RAG_TOP_K=3                          # Number of chunks to retrieve

# Policy Configuration
POLICY_DIR=data                      # Directory containing policy documents

# Debug mode
DEBUG=false                          # Enable debug mode
""" 
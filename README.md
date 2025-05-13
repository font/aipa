# AIPA: AI Policy Automation

AIPA is an AI-based policy engine that uses natural language policies defined in text documents and provides them to an LLM that leverages RAG (Retrieval Augmented Generation) to enforce those policies. It integrates with llama-stack's Safety API associated with Shield resources for comprehensive policy enforcement.

## Features

- **Natural Language Policies**: Define policies in plain English in text documents
- **RAG-Based Policy Enforcement**: Uses LLMs with retrieval augmentation to understand and enforce policies
- **Multiple LLM Support**: Use OpenAI models, local llama models, or connect to Ollama/vLLM via llama-stack-client
- **llama-stack Integration**: Seamlessly integrates with llama-stack's Safety API
- **Combined Evaluation**: Evaluates content using both RAG-based policy engine and llama-stack Safety API
- **Policy Synchronization**: Sync local policy documents to llama-stack Shield resources
- **RESTful API**: Easy-to-use API for content evaluation and policy management

## Architecture

The system consists of the following main components:

1. **Policy Engine**: Core component that uses RAG to enforce policies
2. **llama-stack Integration**: Integration with llama-stack's Safety API
3. **Integrated Policy Engine**: Combines RAG-based policy enforcement with llama-stack Safety API
4. **API Layer**: FastAPI endpoints for content evaluation and policy management

## Getting Started

### Prerequisites

- Python 3.9+
- OpenAI API key (if using OpenAI models)
- llama-stack API key and Shield resource ID
- Ollama or vLLM server (if using llama-stack-client)

### Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/aipa.git
   cd aipa
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Copy the example environment file and update it with your API keys:
   ```
   cp env.example .env
   ```

4. Update the `.env` file with your API keys and configuration.

### LLM Configuration

AIPA supports multiple LLM providers. You can configure which to use in the `.env` file:

#### Using OpenAI (default)

```
LLM_PROVIDER=openai
LLM_MODEL=gpt-4o
EMBEDDING_PROVIDER=openai
EMBEDDING_MODEL=text-embedding-3-small
```

#### Using llama models locally

1. Download a llama model:
   ```
   python scripts/download_llama_model.py --model llama-3-8b-instruct
   ```

2. Configure your `.env` file:
   ```
   LLM_PROVIDER=llama
   LLAMA_MODEL_PATH=models/llama-3-8b-instruct.Q4_K_M.gguf
   LLAMA_N_GPU_LAYERS=0  # Set to higher number for GPU acceleration
   
   # You can also use SentenceTransformers for embeddings
   EMBEDDING_PROVIDER=sentence_transformers
   SENTENCE_TRANSFORMER_MODEL=all-MiniLM-L6-v2
   ```

#### Using Ollama or vLLM via llama-stack-client

1. Start your Ollama or vLLM server. For Ollama:
   ```
   ollama serve
   ```

2. Pull the models you want to use:
   ```
   # For Ollama
   ollama pull llama3
   ollama pull nomic-embed-text
   ```

3. Configure your `.env` file:
   ```
   LLM_PROVIDER=llama_stack
   LLAMA_STACK_CLIENT_URL=http://localhost:11434  # Default Ollama port
   LLAMA_STACK_CLIENT_PROVIDER=ollama  # or vllm
   LLAMA_STACK_CLIENT_MODEL=llama3
   
   # You can also use llama-stack for embeddings
   EMBEDDING_PROVIDER=llama_stack
   LLAMA_STACK_CLIENT_EMBEDDING_MODEL=nomic-embed-text
   ```

### Running the API

Start the API server:

```
python -m app.main
```

The API will be available at http://localhost:8000. API documentation is available at http://localhost:8000/docs.

## Usage

### Defining Policies

Create policy documents in the `data/policies` directory. Policy documents should be structured as follows:

```
# Policy Title

## Section Name
1. Policy rule one
2. Policy rule two
3. Policy rule three

## Another Section
1. Another policy rule
2. Yet another policy rule
```

### Evaluating Content

Send a POST request to `/api/v1/evaluate` with the content to evaluate:

```bash
curl -X POST "http://localhost:8000/api/v1/evaluate" \
     -H "Content-Type: application/json" \
     -d '{"content": "This is the content to evaluate", "metadata": {"user_id": "123"}}'
```

### Syncing Policies

Send a POST request to `/api/v1/sync-policy` to sync policies to llama-stack:

```bash
curl -X POST "http://localhost:8000/api/v1/sync-policy" \
     -H "Content-Type: application/json" \
     -d '{"policy_file_path": "data/policies/custom_policy.txt"}'
```

## Development

### Project Structure

```
aipa/
├── app/
│   ├── api/
│   │   ├── endpoints.py    # API endpoints
│   │   └── models.py       # Pydantic models for API
│   ├── config/
│   │   └── settings.py     # Configuration settings
│   ├── core/
│   │   ├── policy_engine.py          # RAG-based policy engine
│   │   └── integrated_policy_engine.py # Integrated policy engine
│   ├── integrations/
│   │   └── llama_stack/
│   │       └── safety_api.py # llama-stack Safety API integration
│   ├── utils/
│   │   └── policy_loader.py # Utilities for loading policies
│   └── main.py              # Main application entry point
├── data/
│   └── policies/            # Policy documents
│       └── sample_policy.txt # Sample policy document
├── models/                   # Directory for downloaded llama models (if using local models)
├── scripts/
│   └── download_llama_model.py # Script to download llama models (if using local models)
├── requirements.txt         # Project dependencies
├── env.example              # Example environment variables
└── README.md                # Project documentation
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.
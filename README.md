# AIPA (AI Policy Advisor)

A simple AI-based policy engine that uses natural language policies to make decisions.

## Features

- Uses RAG (Retrieval Augmented Generation) to provide context to LLMs
- Supports multiple LLM providers: LlamaStack, Anthropic Claude, OpenAI GPT
- Policy enforcement based on natural language policy documents
- Kubernetes manifest validation against company policies
- CLI and API interfaces
- Minimal viable implementation for extensibility

## Quick Start

1. Install dependencies:
   ```bash
   pip install -e .
   ```

2. Configure your LLM provider by copying the example config:
   ```bash
   cp .env.example .env
   # Edit .env with your API keys and preferences
   ```

3. Place your policy documents in the `data/` directory

4. Check available providers:
   ```bash
   python src/core/cli.py providers
   ```

5. Ask policy questions:
   ```bash
   python src/core/cli.py ask "Can I install software on my work laptop?"
   ```

6. Validate Kubernetes manifests:
   ```bash
   python src/core/cli.py validate-manifest deployment.yaml
   ```

## LLM Provider Configuration

### Anthropic Claude (Recommended)
```env
LLM_PROVIDER=anthropic
ANTHROPIC_API_KEY=your_api_key_here
ANTHROPIC_MODEL=claude-3-haiku-20240307
```

### OpenAI GPT
```env
LLM_PROVIDER=openai
OPENAI_API_KEY=your_api_key_here
OPENAI_MODEL=gpt-3.5-turbo
```

### LlamaStack (Local)
```env
LLM_PROVIDER=llamastack
LLAMASTACK_API_URL=http://localhost:8000
LLAMASTACK_MODEL=llama2
```

## CLI Usage

```bash
# Ask policy questions
python src/core/cli.py ask "What are the password requirements?"

# Use specific provider
python src/core/cli.py --provider anthropic ask "Security policy question"

# Validate Kubernetes manifest
python src/core/cli.py validate-manifest tests/deployment-with-violations.yaml

# Check provider status
python src/core/cli.py providers
```

## API Usage

Run the API server:
```bash
python -m src.api.main
```

Make policy queries:
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "Can I install software on my work laptop?"}'
```

## Architecture

- `src/core/` - Core functionality and config
- `src/policy/` - Policy loading and management
- `src/rag/` - Retrieval Augmented Generation engine
- `src/api/` - API interfaces for querying the policy engine
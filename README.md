# AIPA (AI Policy Advisor)

A simple AI-based policy engine that uses natural language policies to make decisions.

## Features

- Uses RAG (Retrieval Augmented Generation) to provide context to LLMs
- Built on the llama-stack ecosystem
- Policy enforcement based on natural language policy documents
- Minimal viable implementation for extensibility

## Quick Start

1. Install dependencies:
   ```
   pip install -e .
   ```

2. Place your policy documents in the `data/` directory

3. Run the API server:
   ```
   python -m src.api.main
   ```

4. Make policy queries:
   ```
   curl -X POST http://localhost:8000/query -H "Content-Type: application/json" -d '{"query": "Can I install software on my work laptop?"}'
   ```

## Architecture

- `src/core/` - Core functionality and config
- `src/policy/` - Policy loading and management
- `src/rag/` - Retrieval Augmented Generation engine
- `src/api/` - API interfaces for querying the policy engine
# Demo

AIPA

RAG based policy engine using natural language.

Ground the LLM using Retreival Augmented Generation (RAG) by creating an index
of the policy document by:

1. Embedding: first breaks down your documents into smaller pieces called
   Nodes.
2. Vector Embeddings: It then creates vector embeddings (numerical
   representations of text meaning) for each node using an LLM API.
3. Storing in a Vector Store: These vector embeddings, along with the
   corresponding nodes, are stored in a chosen vector store/vector database.

LlamaIndex leverages various vector stores/vector databases as its underlying
storage backend. By default, LlamaIndex uses a simple, in-memory for quick
experimentation e.g. SimpleVectorStore, Faiss, Hnslib, etc. This can be easily
persisted to disk.

## Setup

### System 1

#### Window 1

```bash
ollama run llama3.2:3b-instruct-fp16 --keepalive 60m
```

#### Window 2

```bash
export LLAMA_STACK_MODEL="meta-llama/Llama-3.2-3B-Instruct"
export INFERENCE_MODEL="meta-llama/Llama-3.2-3B-Instruct"
export LLAMA_STACK_PORT=8321
export LLAMA_STACK_SERVER=http://localhost:$LLAMA_STACK_PORT
export LLAMA_STACK_ENDPOINT=$LLAMA_STACK_SERVER
podman run -it -p $LLAMA_STACK_PORT:$LLAMA_STACK_PORT -v ~/.llama:/root/.llama:Z --network=host llamastack/distribution-ollama --port $LLAMA_STACK_PORT --env INFERENCE_MODEL=$LLAMA_STACK_MODEL --env OLLAMA_URL=http://localhos
t:11434
```

### System 2

## Demo

```bash
vi .env.example
vi data/company_policy.txt
python src/core/cli.py -p llamastack ask 'What software can I install?'
python src/core/cli.py -p llamastack ask "Can I use my work laptop for personal use? If so, how much?"
python src/core/cli.py -p llamastack ask "What OCI registries are approved?"
```

```bash
python -m src api
python src/core/cli.py --use-api ask "What OCI registries are approved?"
vi tests/deployment-with-violations.yaml
python src/core/cli.py --use-api validate-manifest tests/deployment-with-violations.yaml
vi tests/deployment-compliant.yaml
python src/core/cli.py --use-api validate-manifest tests/deployment-compliant.yaml
```


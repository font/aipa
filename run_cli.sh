#!/bin/bash

# AIPA CLI Runner
# This script sets up the environment and runs the CLI

# Default environment variables
export LLM_PROVIDER=${LLM_PROVIDER:-anthropic}
export LLM_TEMPERATURE=${LLM_TEMPERATURE:-0.1}
export LLM_MAX_TOKENS=${LLM_MAX_TOKENS:-1024}
export LLAMASTACK_API_URL=${LLAMASTACK_API_URL:-http://localhost:8000}
export LLAMASTACK_MODEL=${LLAMASTACK_MODEL:-llama2}
export ANTHROPIC_MODEL=${ANTHROPIC_MODEL:-claude-3-haiku-20240307}
export OPENAI_MODEL=${OPENAI_MODEL:-gpt-3.5-turbo}
export RAG_CHUNK_SIZE=${RAG_CHUNK_SIZE:-512}
export RAG_CHUNK_OVERLAP=${RAG_CHUNK_OVERLAP:-50}
export RAG_TOP_K=${RAG_TOP_K:-3}
export POLICY_DIR=${POLICY_DIR:-data}
export DEBUG=${DEBUG:-false}

# Check if API key is set for the chosen provider (local mode only)
if [ "$1" != "--use-api" ] && [ "$2" != "--use-api" ]; then
    if [ "$LLM_PROVIDER" = "anthropic" ] && [ -z "$ANTHROPIC_API_KEY" ]; then
        echo "Warning: ANTHROPIC_API_KEY is not set. Please set it to use Anthropic Claude."
        echo "export ANTHROPIC_API_KEY='your_api_key_here'"
        echo ""
        echo "Or use API mode: $0 --use-api [command]"
    fi

    if [ "$LLM_PROVIDER" = "openai" ] && [ -z "$OPENAI_API_KEY" ]; then
        echo "Warning: OPENAI_API_KEY is not set. Please set it to use OpenAI GPT."
        echo "export OPENAI_API_KEY='your_api_key_here'"
        echo ""
        echo "Or use API mode: $0 --use-api [command]"
    fi
fi

# Show usage examples if no arguments provided
if [ $# -eq 0 ]; then
    echo "AIPA CLI - AI Policy Advisor"
    echo ""
    echo "Usage examples:"
    echo ""
    echo "LOCAL MODE (processes policies locally):"
    echo "  $0 ask 'Can I use Docker Hub images?'"
    echo "  $0 validate-manifest tests/deployment-compliant.yaml"
    echo "  $0 --provider anthropic --model claude-3-5-sonnet-20241022 ask 'Security policies?'"
    echo ""
    echo "API MODE (queries running server, more efficient):"
    echo "  $0 --use-api ask 'Can I use Docker Hub images?'"
    echo "  $0 --use-api validate-manifest tests/deployment-compliant.yaml"
    echo "  $0 --use-api --api-url http://localhost:8001 ask 'Security policies?'"
    echo ""
    echo "Other commands:"
    echo "  $0 providers  # Show available LLM providers"
    echo ""
    echo "Note: API mode requires a running server (python -m src.api.main)"
    exit 1
fi

# Run the CLI with all arguments passed through
python src/core/cli.py "$@"

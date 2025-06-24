#!/usr/bin/env python3
"""Command-line interface for the policy engine."""

import logging
import click
import yaml
import httpx
from pathlib import Path
from typing import Optional

from src.rag.engine import RagEngine, K8sPolicyEnforcer
from src.core.config import config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _override_config(provider: str, model: Optional[str] = None):
    """Override configuration with CLI parameters."""
    import os
    
    # Override provider
    os.environ['LLM_PROVIDER'] = provider
    config.llm.provider = provider
    
    # Override model if specified
    if model:
        if provider == 'anthropic':
            os.environ['ANTHROPIC_MODEL'] = model
            config.anthropic.model_name = model
        elif provider == 'openai':
            os.environ['OPENAI_MODEL'] = model  
            config.openai.model_name = model
        elif provider == 'llamastack':
            os.environ['LLAMASTACK_MODEL'] = model
            config.llamastack.model_name = model

@click.group()
@click.option('--provider', '-p', 
              type=click.Choice(['llamastack', 'anthropic', 'openai']), 
              help='LLM provider to use (overrides config, local mode only)')
@click.option('--model', '-m', help='Model name to use (overrides config, local mode only)')
@click.option('--api-url', default='http://localhost:8000', 
              help='API server URL for remote mode (default: http://localhost:8000)')
@click.option('--use-api', is_flag=True, 
              help='Use API server instead of local processing (more efficient, avoids re-indexing)')
@click.pass_context
def cli(ctx, provider, model, api_url, use_api):
    """Policy engine CLI.
    
    Two modes of operation:
    
    1. LOCAL MODE (default): Processes policies locally, builds RAG index on each run
    
    2. API MODE (--use-api): Queries a running API server, more efficient as it
       reuses pre-built RAG indices and avoids re-parsing policy documents
    """
    # Store options in context for subcommands
    ctx.ensure_object(dict)
    ctx.obj['provider'] = provider
    ctx.obj['model'] = model
    ctx.obj['api_url'] = api_url
    ctx.obj['use_api'] = use_api

@cli.command()
@click.argument('query')
@click.pass_context
def ask(ctx, query: str):
    """Ask a question about company policies."""
    use_api = ctx.obj.get('use_api', False)
    api_url = ctx.obj.get('api_url', 'http://localhost:8000')
    
    if use_api:
        # Use API server
        try:
            with httpx.Client() as client:
                response = client.post(
                    f"{api_url}/query",
                    json={"query": query},
                    timeout=30.0
                )
                response.raise_for_status()
                result = response.json()
                
                click.echo("\nAnswer:")
                click.echo(result["answer"])
                
                if result.get("metadata"):
                    click.echo(f"\nMode: API Server ({api_url})")
                
                if result.get("sources"):
                    click.echo("\nSources:")
                    for source in result["sources"]:
                        click.echo(f"- {source['source']}")
                        
        except httpx.RequestError as e:
            logger.error(f"Error connecting to API server: {e}")
            raise click.ClickException(f"Failed to connect to API server at {api_url}: {e}")
        except httpx.HTTPStatusError as e:
            logger.error(f"API server error: {e}")
            raise click.ClickException(f"API server error: {e.response.status_code} - {e.response.text}")
        except Exception as e:
            logger.error(f"Error processing query via API: {e}")
            raise click.ClickException(f"Failed to process query via API: {e}")
    else:
        # Use local processing
        provider = ctx.obj.get('provider')
        model = ctx.obj.get('model')
        
        if provider:
            _override_config(provider, model)
        
        try:
            engine = RagEngine()
            result = engine.query(query)
            
            click.echo("\nAnswer:")
            click.echo(result["answer"])
            
            click.echo(f"\nUsed: {result['metadata']['provider']} - {result['metadata']['model']}")
            
            if result["sources"]:
                click.echo("\nSources:")
                for source in result["sources"]:
                    click.echo(f"- {source['source']}")
                    
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            click.echo(f"Error: {e}", err=True)
            raise click.ClickException(f"Failed to process query: {e}")

@cli.command()
@click.argument('manifest_path', type=click.Path(exists=True))
@click.option('--output', '-o', type=click.Path(), help='Output file for violations')
@click.pass_context
def validate_manifest(ctx, manifest_path: str, output: Optional[str]):
    """Validate a Kubernetes manifest against company policies."""
    use_api = ctx.obj.get('use_api', False)
    api_url = ctx.obj.get('api_url', 'http://localhost:8000')
    
    # Read manifest file
    try:
        with open(manifest_path, 'r') as f:
            manifest = f.read()
    except Exception as e:
        logger.error(f"Failed to read manifest file: {e}")
        raise click.ClickException(f"Failed to read manifest file: {e}")
    
    if use_api:
        # Use API server
        try:
            with httpx.Client() as client:
                response = client.post(
                    f"{api_url}/validate-manifest",
                    json={"manifest": manifest},
                    timeout=60.0  # Longer timeout for manifest validation
                )
                response.raise_for_status()
                result = response.json()
                
                violations = result.get("violations", [])
                
                if violations:
                    click.echo(f"\nFound {len(violations)} policy violations in {manifest_path}:")
                    for violation in violations:
                        click.echo(f"\nRule: {violation['rule']}")
                        click.echo(f"Violation: {violation['violation']}")
                        click.echo(f"Severity: {violation['severity']}")
                    
                    # Write violations to output file if specified
                    if output:
                        try:
                            with open(output, 'w') as f:
                                yaml.dump(violations, f)
                            click.echo(f"\nViolations written to {output}")
                        except Exception as e:
                            logger.error(f"Failed to write violations to output file: {e}")
                            raise click.ClickException(f"Failed to write violations to output file: {e}")
                    
                    click.echo(f"\nMode: API Server ({api_url})")
                    # Exit with error code if there are violations
                    raise click.ClickException("Policy violations found")
                else:
                    click.echo(f"\nNo policy violations found in {manifest_path}")
                    click.echo(f"Mode: API Server ({api_url})")
                    
        except httpx.RequestError as e:
            logger.error(f"Error connecting to API server: {e}")
            raise click.ClickException(f"Failed to connect to API server at {api_url}: {e}")
        except httpx.HTTPStatusError as e:
            logger.error(f"API server error: {e}")
            raise click.ClickException(f"API server error: {e.response.status_code} - {e.response.text}")
        except Exception as e:
            logger.error(f"Error validating manifest via API: {e}")
            raise click.ClickException(f"Failed to validate manifest via API: {e}")
    else:
        # Use local processing
        provider = ctx.obj.get('provider')
        model = ctx.obj.get('model')
        
        if provider:
            _override_config(provider, model)
        
        # Initialize RAG engine and policy enforcer
        engine = RagEngine()
        policy_enforcer = K8sPolicyEnforcer(engine)
        
        # Validate manifest
        try:
            violations = policy_enforcer.enforce_policy(manifest)
        except Exception as e:
            logger.error(f"Failed to validate manifest: {e}")
            raise click.ClickException(f"Failed to validate manifest: {e}")
        
        # Output results
        if violations:
            click.echo(f"\nFound {len(violations)} policy violations in {manifest_path}:")
            for violation in violations:
                click.echo(f"\nRule: {violation.rule}")
                click.echo(f"Violation: {violation.violation}")
                click.echo(f"Severity: {violation.severity}")
            
            # Write violations to output file if specified
            if output:
                try:
                    with open(output, 'w') as f:
                        yaml.dump([v.dict() for v in violations], f)
                    click.echo(f"\nViolations written to {output}")
                except Exception as e:
                    logger.error(f"Failed to write violations to output file: {e}")
                    raise click.ClickException(f"Failed to write violations to output file: {e}")
            
            # Exit with error code if there are violations
            raise click.ClickException("Policy violations found")
        else:
            click.echo(f"\nNo policy violations found in {manifest_path}")

@cli.command()
def providers():
    """List available LLM providers and their configuration status."""
    click.echo("Available LLM Providers:")
    click.echo("=" * 50)
    
    # Check LlamaStack
    click.echo("\n🦙 LlamaStack:")
    click.echo(f"   URL: {config.llamastack.api_url}")
    click.echo(f"   Model: {config.llamastack.model_name}")
    try:
        from llama_stack_client import LlamaStackClient
        client = LlamaStackClient(base_url=config.llamastack.api_url, timeout=5.0)
        # Try a simple health check - this might fail but that's ok
        click.echo("   Status: ✅ Client available")
    except Exception as e:
        click.echo(f"   Status: ❌ {str(e)[:50]}...")
    
    # Check Anthropic
    click.echo("\n🤖 Anthropic:")
    click.echo(f"   Model: {config.anthropic.model_name}")
    if config.anthropic.api_key:
        click.echo("   API Key: ✅ Set")
        try:
            from llama_index.llms.anthropic import Anthropic
            click.echo("   Status: ✅ Available")
        except ImportError:
            click.echo("   Status: ❌ Package not installed (llama-index-llms-anthropic)")
    else:
        click.echo("   API Key: ❌ Not set (ANTHROPIC_API_KEY)")
        click.echo("   Status: ❌ Not configured")
    
    # Check OpenAI
    click.echo("\n🧠 OpenAI:")
    click.echo(f"   Model: {config.openai.model_name}")
    if config.openai.api_key:
        click.echo("   API Key: ✅ Set")
        try:
            from llama_index.llms.openai import OpenAI
            click.echo("   Status: ✅ Available")
        except ImportError:
            click.echo("   Status: ❌ Package not installed (llama-index-llms-openai)")
    else:
        click.echo("   API Key: ❌ Not set (OPENAI_API_KEY)")
        click.echo("   Status: ❌ Not configured")
    
    click.echo(f"\nCurrent Provider: {config.llm.provider}")
    click.echo("=" * 50)

def main():
    """Main entry point for the CLI."""
    cli()

if __name__ == '__main__':
    main() 
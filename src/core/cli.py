#!/usr/bin/env python3
"""Command-line interface for the policy engine."""

import logging
import click
import yaml
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
              help='LLM provider to use (overrides config)')
@click.option('--model', '-m', help='Model name to use (overrides config)')
@click.pass_context
def cli(ctx, provider, model):
    """Policy engine CLI."""
    # Store provider and model in context for subcommands
    ctx.ensure_object(dict)
    ctx.obj['provider'] = provider
    ctx.obj['model'] = model

@cli.command()
@click.argument('query')
@click.pass_context
def ask(ctx, query: str):
    """Ask a question about company policies."""
    # Override config if provider/model specified
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
    # Override config if provider/model specified
    provider = ctx.obj.get('provider')
    model = ctx.obj.get('model')
    
    if provider:
        _override_config(provider, model)
    
    # Initialize RAG engine and policy enforcer
    engine = RagEngine()
    policy_enforcer = K8sPolicyEnforcer(engine)
    
    # Read manifest file
    try:
        with open(manifest_path, 'r') as f:
            manifest = f.read()
    except Exception as e:
        logger.error(f"Failed to read manifest file: {e}")
        raise click.ClickException(f"Failed to read manifest file: {e}")
    
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
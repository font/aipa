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

@click.group()
def cli():
    """Policy engine CLI."""
    pass

@cli.command()
@click.argument('query')
def ask(query: str):
    """Ask a question about company policies."""
    engine = RagEngine()
    result = engine.query(query)
    
    click.echo("\nAnswer:")
    click.echo(result["answer"])
    
    if result["sources"]:
        click.echo("\nSources:")
        for source in result["sources"]:
            click.echo(f"- {source['source']}")

@cli.command()
@click.argument('manifest_path', type=click.Path(exists=True))
@click.option('--output', '-o', type=click.Path(), help='Output file for violations')
def validate_manifest(manifest_path: str, output: Optional[str]):
    """Validate a Kubernetes manifest against company policies."""
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

def main():
    """Main entry point for the CLI."""
    cli()

if __name__ == '__main__':
    main() 
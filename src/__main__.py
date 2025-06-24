import argparse
import sys

from src.api.main import start as start_api
from src.core.cli import cli as start_cli


def main():
    """Main entry point for the application."""
    parser = argparse.ArgumentParser(
        description="AI Policy Advisor - A RAG-based policy engine using llama-stack"
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # API server command
    api_parser = subparsers.add_parser("api", help="Start the API server")
    
    # CLI command - pass through to Click
    cli_parser = subparsers.add_parser("cli", help="Run CLI commands")
    
    args, remaining = parser.parse_known_args()
    
    if args.command == "api":
        start_api()
    elif args.command == "cli":
        # Pass remaining args to Click CLI
        sys.argv = ["cli"] + remaining
        start_cli()
    else:
        parser.print_help()


if __name__ == "__main__":
    main() 
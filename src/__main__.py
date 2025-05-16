import argparse
import sys

from src.api.main import start as start_api
from src.core.cli import main as start_cli


def main():
    """Main entry point for the application."""
    parser = argparse.ArgumentParser(
        description="AI Policy Advisor - A RAG-based policy engine using llama-stack"
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # API server command
    api_parser = subparsers.add_parser("api", help="Start the API server")
    
    # CLI command
    cli_parser = subparsers.add_parser("cli", help="Run CLI query")
    cli_parser.add_argument(
        "query", nargs="?", help="The policy question to query"
    )
    cli_parser.add_argument(
        "--json", action="store_true", help="Output results in JSON format"
    )
    cli_parser.add_argument(
        "--file", "-f", help="Read query from file instead of command line"
    )
    
    args = parser.parse_args()
    
    if args.command == "api":
        start_api()
    elif args.command == "cli":
        # Pass remaining args to CLI
        sys.argv = [sys.argv[0]] + sys.argv[2:]
        start_cli()
    else:
        parser.print_help()


if __name__ == "__main__":
    main() 
import argparse
import json
import logging
import sys

from src.rag.engine import rag_engine
from src.core.config import config

# Configure logging
logging.basicConfig(
    level=logging.DEBUG if config.debug else logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="AI Policy Advisor - Query the policy engine from the command line."
    )
    parser.add_argument(
        "query", nargs="?", help="The policy question to query."
    )
    parser.add_argument(
        "--json", action="store_true", help="Output results in JSON format."
    )
    parser.add_argument(
        "--file", "-f", help="Read query from file instead of command line."
    )
    
    args = parser.parse_args()
    
    # Get the query from file or command line
    query = None
    if args.file:
        try:
            with open(args.file, "r") as f:
                query = f.read().strip()
        except Exception as e:
            logger.error(f"Error reading file: {e}")
            sys.exit(1)
    elif args.query:
        query = args.query
    else:
        # Interactive mode
        print("Enter your policy question (Ctrl+D to submit):")
        query = sys.stdin.read().strip()
    
    if not query:
        logger.error("No query provided.")
        sys.exit(1)
    
    try:
        result = rag_engine.query(query)
        
        if args.json:
            print(json.dumps(result, indent=2))
        else:
            print("\n" + "=" * 80)
            print(f"QUESTION: {query}")
            print("=" * 80)
            print(f"\nANSWER: {result['answer']}")
            
            if result["sources"]:
                print("\nSOURCES:")
                for i, source in enumerate(result["sources"], 1):
                    print(f"\n{i}. {source['source']}")
                    print(f"   {source['text'][:200]}...")
            
            print("\n" + "=" * 80)
            
    except Exception as e:
        logger.error(f"Error processing query: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main() 
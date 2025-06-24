#!/usr/bin/env python3
"""Start the AIPA API server."""

import os
import sys
import logging

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.api.main import start

if __name__ == "__main__":
    print("Starting AIPA API server...")
    print("Server will be available at: http://localhost:8000")
    print("API docs at: http://localhost:8000/docs")
    print("Press Ctrl+C to stop")
    start()

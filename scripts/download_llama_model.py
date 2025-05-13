#!/usr/bin/env python
"""
Script to download a llama model for use with the policy engine.
"""
import os
import sys
import argparse
import logging
import requests
from pathlib import Path
from tqdm import tqdm

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Default model URLs
MODEL_URLS = {
    "llama-3-8b-instruct": "https://huggingface.co/TheBloke/Llama-3-8B-Instruct-GGUF/resolve/main/llama-3-8b-instruct.Q4_K_M.gguf",
    "llama-2-7b-chat": "https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/resolve/main/llama-2-7b-chat.Q4_K_M.gguf",
    "mistral-7b-instruct": "https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/mistral-7b-instruct-v0.2.Q4_K_M.gguf"
}

def download_file(url, destination):
    """
    Download a file from a URL with a progress bar.
    
    Args:
        url: URL to download from
        destination: Path to save the file to
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    
    # Check if file already exists
    if os.path.exists(destination):
        logger.info(f"File already exists at {destination}")
        return
    
    # Download the file
    logger.info(f"Downloading from {url} to {destination}")
    
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024  # 1 KB
    
    with open(destination, 'wb') as f, tqdm(
        desc=os.path.basename(destination),
        total=total_size,
        unit='B',
        unit_scale=True,
        unit_divisor=1024,
    ) as progress_bar:
        for data in response.iter_content(block_size):
            f.write(data)
            progress_bar.update(len(data))
    
    logger.info(f"Download complete: {destination}")

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Download a llama model for use with the policy engine.")
    parser.add_argument(
        "--model", 
        choices=list(MODEL_URLS.keys()),
        default="llama-3-8b-instruct",
        help="Model to download"
    )
    parser.add_argument(
        "--output-dir", 
        default="models",
        help="Directory to save the model to"
    )
    parser.add_argument(
        "--custom-url", 
        help="Custom URL to download from"
    )
    
    args = parser.parse_args()
    
    # Determine URL and destination
    url = args.custom_url or MODEL_URLS[args.model]
    filename = os.path.basename(url)
    destination = os.path.join(args.output_dir, filename)
    
    # Download the file
    try:
        download_file(url, destination)
        
        # Update the .env file if it exists
        env_file = ".env"
        if os.path.exists(env_file):
            with open(env_file, 'r') as f:
                env_content = f.read()
            
            # Check if LLAMA_MODEL_PATH is already in the file
            if "LLAMA_MODEL_PATH" in env_content:
                # Update the existing line
                lines = env_content.split('\n')
                for i, line in enumerate(lines):
                    if line.startswith("LLAMA_MODEL_PATH="):
                        lines[i] = f"LLAMA_MODEL_PATH={destination}"
                        break
                
                with open(env_file, 'w') as f:
                    f.write('\n'.join(lines))
            else:
                # Append to the file
                with open(env_file, 'a') as f:
                    f.write(f"\n# Llama model path\nLLAMA_MODEL_PATH={destination}\n")
            
            logger.info(f"Updated .env file with LLAMA_MODEL_PATH={destination}")
        
        logger.info(f"To use this model, set LLAMA_MODEL_PATH={destination} in your .env file")
        logger.info(f"And set LLM_PROVIDER=llama in your .env file")
        
    except Exception as e:
        logger.error(f"Error downloading model: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 
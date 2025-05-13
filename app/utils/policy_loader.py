import os
from typing import List, Dict, Any
from pathlib import Path

def load_policy_document(file_path: str) -> str:
    """
    Load a policy document from a file.
    
    Args:
        file_path: Path to the policy document
        
    Returns:
        The content of the policy document as a string
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Policy file not found: {file_path}")
    
    with open(file_path, 'r') as f:
        return f.read()

def parse_policy_sections(policy_text: str) -> Dict[str, List[str]]:
    """
    Parse a policy document into sections and rules.
    
    Args:
        policy_text: The policy document text
        
    Returns:
        A dictionary mapping section names to lists of rules
    """
    sections = {}
    current_section = None
    
    for line in policy_text.split('\n'):
        line = line.strip()
        
        # Skip empty lines
        if not line:
            continue
        
        # Check for section headers (## Section Name)
        if line.startswith('## '):
            current_section = line[3:].strip()
            sections[current_section] = []
        
        # Check for numbered rules (1. Rule text)
        elif current_section and line[0].isdigit() and '. ' in line:
            rule = line[line.find('.')+1:].strip()
            sections[current_section].append(rule)
    
    return sections

def get_all_policy_files(policy_dir: str) -> List[str]:
    """
    Get all policy files in a directory.
    
    Args:
        policy_dir: Directory containing policy files
        
    Returns:
        List of paths to policy files
    """
    if not os.path.exists(policy_dir):
        raise FileNotFoundError(f"Policy directory not found: {policy_dir}")
    
    policy_files = []
    for file in os.listdir(policy_dir):
        file_path = os.path.join(policy_dir, file)
        if os.path.isfile(file_path) and file.endswith('.txt'):
            policy_files.append(file_path)
    
    return policy_files 
from pathlib import Path
from typing import List, Dict, Union

from src.core.config import config


class PolicyLoader:
    """Load policy documents from the policy directory."""
    
    def __init__(self, policy_dir: Path = None):
        """Initialize the policy loader.
        
        Args:
            policy_dir: Directory containing policy documents. Defaults to config value.
        """
        self.policy_dir = policy_dir or config.policy.policy_dir
    
    def load_policies(self) -> List[Dict[str, Union[str, Path]]]:
        """Load all policy documents from the policy directory.
        
        Returns:
            List of dictionaries containing policy document metadata and content.
        """
        policy_docs = []
        
        # Load all text files in the policy directory
        for policy_file in self.policy_dir.glob("*.txt"):
            with open(policy_file, "r") as f:
                content = f.read()
                
            policy_docs.append({
                "source": str(policy_file),
                "filename": policy_file.name,
                "content": content
            })
        
        return policy_docs
    
    def get_policy_count(self) -> int:
        """Get the number of policy documents.
        
        Returns:
            Number of policy documents.
        """
        return len(list(self.policy_dir.glob("*.txt")))


# Create a singleton policy loader instance
policy_loader = PolicyLoader() 
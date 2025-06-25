#!/usr/bin/env python3
"""Pytest configuration and shared fixtures."""

import pytest
import os
import sys
from unittest.mock import patch, MagicMock

# Add src to path for all tests
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


@pytest.fixture(autouse=True)
def mock_llm_setup():
    """Auto-use fixture to mock LLM setup for all tests to avoid external dependencies."""
    with patch('src.core.llm_factory.llm_factory.setup_global_llm') as mock_setup:
        mock_llm = MagicMock()
        mock_setup.return_value = mock_llm
        
        # Mock Settings to avoid import issues
        with patch('src.rag.engine.Settings') as mock_settings:
            yield mock_llm


@pytest.fixture(autouse=True)
def mock_embedding_setup():
    """Auto-use fixture to mock HuggingFace embeddings to avoid downloading models."""
    with patch('src.rag.engine.HuggingFaceEmbedding') as mock_embedding:
        mock_embed_instance = MagicMock()
        mock_embedding.return_value = mock_embed_instance
        yield mock_embed_instance


@pytest.fixture(autouse=True)
def mock_policy_loader():
    """Auto-use fixture to mock policy loader to avoid file system dependencies."""
    with patch('src.rag.engine.policy_loader') as mock_loader:
        mock_loader.load_policies.return_value = [
            {
                "content": "Test policy content for CI testing",
                "source": "test_policy.txt"
            }
        ]
        yield mock_loader


@pytest.fixture
def sample_k8s_manifest():
    """Fixture providing a sample Kubernetes manifest for testing."""
    return """
apiVersion: apps/v1
kind: Deployment
metadata:
  name: test-deployment
  namespace: default
spec:
  replicas: 2
  selector:
    matchLabels:
      app: test-app
  template:
    metadata:
      labels:
        app: test-app
    spec:
      containers:
      - name: test-container
        image: nginx:1.20
        ports:
        - containerPort: 80
        resources:
          requests:
            memory: "64Mi"
            cpu: "250m"
          limits:
            memory: "128Mi"
            cpu: "500m"
"""


@pytest.fixture
def sample_non_compliant_manifest():
    """Fixture providing a non-compliant Kubernetes manifest for testing."""
    return """
apiVersion: apps/v1
kind: Deployment
metadata:
  name: bad-deployment
spec:
  replicas: 1
  selector:
    matchLabels:
      app: bad-app
  template:
    metadata:
      labels:
        app: bad-app
    spec:
      containers:
      - name: bad-container
        image: nginx:latest
        securityContext:
          runAsUser: 0
"""


@pytest.fixture
def sample_policy_violations():
    """Fixture providing sample policy violations for testing."""
    from src.rag.engine import PolicyViolation
    
    return [
        PolicyViolation(
            rule="Container images must use specific tags",
            manifest_path="spec.template.spec.containers[0].image",
            violation="Using 'latest' tag is not allowed",
            severity="error"
        ),
        PolicyViolation(
            rule="Containers should not run as root",
            manifest_path="spec.template.spec.containers[0].securityContext.runAsUser",
            violation="Container is configured to run as root (UID 0)",
            severity="warning"
        )
    ] 
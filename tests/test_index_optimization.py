#!/usr/bin/env python3
"""Tests for index building optimization."""

import pytest
import unittest.mock as mock
from unittest.mock import patch, MagicMock
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.rag.engine import RagEngine, K8sPolicyEnforcer


class TestIndexOptimization:
    """Test suite for index building optimization."""

    def test_rag_engine_index_building(self):
        """Test that RagEngine builds index correctly."""
        engine = RagEngine()
        
        # Initially no index
        assert engine.index is None
        
        # Build index
        engine.build_index()
        
        # Index should now exist
        assert engine.index is not None

    def test_k8s_policy_enforcer_uses_rag_engine_index(self):
        """Test that K8sPolicyEnforcer uses the RAG engine's index instead of building its own."""
        engine = RagEngine()
        engine.build_index()
        
        # Create policy enforcer
        enforcer = K8sPolicyEnforcer(engine)
        
        # Should reference the same RAG engine
        assert enforcer.rag_engine is engine
        
        # Should not have its own policy_index attribute
        assert not hasattr(enforcer, 'policy_index')

    def test_api_server_startup_sequence(self):
        """Test that API server builds index once at startup."""
        # Mock the rag_engine singleton to track build_index calls
        with patch('src.rag.engine.rag_engine') as mock_rag_engine:
            mock_rag_engine.index = None
            mock_rag_engine.build_index = MagicMock()
            
            # Import the API main module (simulates startup)
            with patch('src.api.main.rag_engine', mock_rag_engine):
                import src.api.main
                
                # Verify build_index was called during module import
                mock_rag_engine.build_index.assert_called_once()

    def test_cli_backwards_compatibility(self):
        """Test that CLI mode still works with on-demand index building."""
        # Create new RAG engine instance (simulates CLI usage)
        engine = RagEngine()
        
        # Create policy enforcer
        enforcer = K8sPolicyEnforcer(engine)
        
        # Mock the build_index method to track calls
        with patch.object(engine, 'build_index', wraps=engine.build_index) as mock_build:
            with patch.object(engine, 'index', None):
                # Test manifest
                test_manifest = """
apiVersion: apps/v1
kind: Deployment
metadata:
  name: test
spec:
  replicas: 1
  template:
    spec:
      containers:
      - name: test
        image: nginx:latest
"""
                
                # Mock the index to avoid actual LLM calls
                mock_index = MagicMock()
                mock_query_engine = MagicMock()
                mock_index.as_query_engine.return_value = mock_query_engine
                mock_query_engine.query.return_value = "No policy violations found."
                
                # Set up the mock to return our mock index when build_index is called
                def mock_build_side_effect():
                    engine.index = mock_index
                
                mock_build.side_effect = mock_build_side_effect
                
                # Call enforce_policy (should trigger index building)
                violations = enforcer.enforce_policy(test_manifest)
                
                # Verify build_index was called
                mock_build.assert_called_once()
                
                # Should return empty violations for "No policy violations found"
                assert isinstance(violations, list)

    def test_policy_enforcement_reuses_existing_index(self):
        """Test that policy enforcement reuses existing index without rebuilding."""
        engine = RagEngine()
        
        # Mock the index and build_index
        mock_index = MagicMock()
        mock_query_engine = MagicMock()
        mock_index.as_query_engine.return_value = mock_query_engine
        mock_query_engine.query.return_value = "No policy violations found."
        
        # Set the index (simulates already built)
        engine.index = mock_index
        
        enforcer = K8sPolicyEnforcer(engine)
        
        # Mock build_index to track calls
        with patch.object(engine, 'build_index') as mock_build:
            test_manifest = """
apiVersion: apps/v1
kind: Deployment
metadata:
  name: test
spec:
  replicas: 1
  template:
    spec:
      containers:
      - name: test
        image: nginx:latest
"""
            
            # Call enforce_policy
            violations = enforcer.enforce_policy(test_manifest)
            
            # build_index should NOT be called since index already exists
            mock_build.assert_not_called()
            
            # Should still work correctly
            assert isinstance(violations, list)

    def test_multiple_policy_enforcements_single_index(self):
        """Test that multiple policy enforcements use the same index."""
        engine = RagEngine()
        
        # Mock the index
        mock_index = MagicMock()
        mock_query_engine = MagicMock()
        mock_index.as_query_engine.return_value = mock_query_engine
        mock_query_engine.query.return_value = "No policy violations found."
        
        # Build index once
        with patch.object(engine, 'build_index') as mock_build:
            def mock_build_side_effect():
                engine.index = mock_index
            mock_build.side_effect = mock_build_side_effect
            
            enforcer = K8sPolicyEnforcer(engine)
            
            test_manifest = """
apiVersion: apps/v1
kind: Deployment
metadata:
  name: test
spec:
  replicas: 1
  template:
    spec:
      containers:
      - name: test
        image: nginx:latest
"""
            
            # First enforcement - should build index
            violations1 = enforcer.enforce_policy(test_manifest)
            assert mock_build.call_count == 1
            
            # Second enforcement - should reuse existing index
            violations2 = enforcer.enforce_policy(test_manifest)
            assert mock_build.call_count == 1  # Still only called once
            
            # Both should work
            assert isinstance(violations1, list)
            assert isinstance(violations2, list)

    def test_manifest_parsing_integration(self):
        """Test that manifest parsing works correctly with the optimized index."""
        engine = RagEngine()
        enforcer = K8sPolicyEnforcer(engine)
        
        # Test valid YAML manifest
        valid_manifest = """
apiVersion: apps/v1
kind: Deployment
metadata:
  name: test-app
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
        image: nginx:latest
"""
        
        # Should parse without errors
        parsed = enforcer._parse_manifest(valid_manifest)
        assert len(parsed) == 1
        assert parsed[0]['kind'] == 'Deployment'
        assert parsed[0]['metadata']['name'] == 'test-app'

    def test_invalid_manifest_handling(self):
        """Test that invalid manifests are handled correctly."""
        engine = RagEngine()
        enforcer = K8sPolicyEnforcer(engine)
        
        # Test invalid YAML
        invalid_manifest = """
apiVersion: apps/v1
kind: Deployment
metadata:
  name: test-app
spec:
  replicas: 2
  invalid_yaml: [unclosed
"""
        
        # Should raise ValueError for invalid YAML
        with pytest.raises(ValueError, match="Invalid YAML manifest"):
            enforcer._parse_manifest(invalid_manifest)

    def test_empty_manifest_handling(self):
        """Test that empty manifests are handled correctly."""
        engine = RagEngine()
        enforcer = K8sPolicyEnforcer(engine)
        
        # Test empty manifest
        empty_manifest = ""
        
        # Should raise ValueError for empty manifest
        with pytest.raises(ValueError, match="No valid documents found"):
            enforcer._parse_manifest(empty_manifest)


if __name__ == '__main__':
    pytest.main([__file__]) 
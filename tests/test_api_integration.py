#!/usr/bin/env python3
"""Integration tests for API endpoints with index optimization."""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


class TestAPIIntegration:
    """Test suite for API integration with index optimization."""

    def test_startup_builds_index(self):
        """Test that API startup builds the index."""
        mock_rag_engine = MagicMock()
        mock_rag_engine.build_index = MagicMock()
        
        with patch('src.rag.engine.rag_engine', mock_rag_engine):
            with patch('src.api.main.rag_engine', mock_rag_engine):
                # Clear module cache
                modules_to_clear = [m for m in sys.modules.keys() if m.startswith('src.api.main')]
                for module in modules_to_clear:
                    if module in sys.modules:
                        del sys.modules[module]
                
                # Import the app (simulates startup)
                from src.api.main import app
                
                # Verify build_index was called at least once during startup
                assert mock_rag_engine.build_index.call_count >= 1

    def test_health_endpoint_simple(self):
        """Test the /health endpoint works."""
        with patch('src.rag.engine.rag_engine') as mock_rag_engine:
            with patch('src.api.main.rag_engine', mock_rag_engine):
                # Clear module cache
                modules_to_clear = [m for m in sys.modules.keys() if m.startswith('src.api.main')]
                for module in modules_to_clear:
                    if module in sys.modules:
                        del sys.modules[module]
                
                from src.api.main import app
                client = TestClient(app)
                
                response = client.get("/health")
                assert response.status_code == 200
                data = response.json()
                assert data["status"] == "healthy"

    def test_query_endpoint_integration(self):
        """Test query endpoint with actual integration."""
        mock_rag_engine = MagicMock()
        mock_rag_engine.build_index = MagicMock()
        mock_rag_engine.query.return_value = {
            "answer": "Test policy answer",
            "sources": [{"source": "test.txt", "text": "Test content"}],
            "metadata": {"provider": "test", "model": "test-model"}
        }
        
        with patch('src.rag.engine.rag_engine', mock_rag_engine):
            with patch('src.api.main.rag_engine', mock_rag_engine):
                # Clear module cache
                modules_to_clear = [m for m in sys.modules.keys() if m.startswith('src.api.main')]
                for module in modules_to_clear:
                    if module in sys.modules:
                        del sys.modules[module]
                
                from src.api.main import app
                client = TestClient(app)
                
                response = client.post("/query", json={"query": "What is the password policy?"})
                
                assert response.status_code == 200
                data = response.json()
                assert "answer" in data
                assert "sources" in data
                assert "metadata" in data
                
                # Verify the RAG engine query was called
                mock_rag_engine.query.assert_called_with("What is the password policy?")

    def test_validate_manifest_endpoint_structure(self):
        """Test that the validate manifest endpoint has the correct structure."""
        with patch('src.rag.engine.rag_engine') as mock_rag_engine:
            with patch('src.api.main.rag_engine', mock_rag_engine):
                # Clear module cache
                modules_to_clear = [m for m in sys.modules.keys() if m.startswith('src.api.main')]
                for module in modules_to_clear:
                    if module in sys.modules:
                        del sys.modules[module]
                
                from src.api.main import app
                client = TestClient(app)
                
                test_manifest = """
apiVersion: apps/v1
kind: Deployment
metadata:
  name: test-app
spec:
  replicas: 1
  template:
    spec:
      containers:
      - name: test
        image: nginx:latest
"""
                
                response = client.post("/validate-manifest", json={"manifest": test_manifest})
                
                assert response.status_code == 200
                data = response.json()
                
                # Check response structure
                assert "violations" in data
                assert "compliant" in data
                assert "metadata" in data
                assert isinstance(data["violations"], list)
                assert isinstance(data["compliant"], bool)
                assert isinstance(data["metadata"], dict)
                assert "violation_count" in data["metadata"]
                assert "error_count" in data["metadata"]
                assert "warning_count" in data["metadata"]

    def test_query_endpoint_handles_errors(self):
        """Test query endpoint error handling."""
        mock_rag_engine = MagicMock()
        mock_rag_engine.build_index = MagicMock()
        mock_rag_engine.query.side_effect = Exception("Test error")
        
        with patch('src.rag.engine.rag_engine', mock_rag_engine):
            with patch('src.api.main.rag_engine', mock_rag_engine):
                # Clear module cache
                modules_to_clear = [m for m in sys.modules.keys() if m.startswith('src.api.main')]
                for module in modules_to_clear:
                    if module in sys.modules:
                        del sys.modules[module]
                
                from src.api.main import app
                client = TestClient(app)
                
                response = client.post("/query", json={"query": "test query"})
                
                assert response.status_code == 500
                assert "Error processing query" in response.json()["detail"]

    def test_multiple_query_requests_reuse_index(self):
        """Test that multiple query requests reuse the same index without rebuilding."""
        mock_rag_engine = MagicMock()
        mock_rag_engine.build_index = MagicMock()
        mock_rag_engine.query.return_value = {
            "answer": "Test answer",
            "sources": [],
            "metadata": {}
        }
        
        with patch('src.rag.engine.rag_engine', mock_rag_engine):
            with patch('src.api.main.rag_engine', mock_rag_engine):
                # Clear module cache
                modules_to_clear = [m for m in sys.modules.keys() if m.startswith('src.api.main')]
                for module in modules_to_clear:
                    if module in sys.modules:
                        del sys.modules[module]
                
                from src.api.main import app
                client = TestClient(app)
                
                # Reset call count after startup
                mock_rag_engine.build_index.reset_mock()
                
                # Make multiple requests
                client.post("/query", json={"query": "First query"})
                client.post("/query", json={"query": "Second query"})
                client.post("/query", json={"query": "Third query"})
                
                # build_index should not be called again after startup
                mock_rag_engine.build_index.assert_not_called()
                
                # But the engine query method should be called
                assert mock_rag_engine.query.call_count == 3

    def test_api_endpoints_exist(self):
        """Test that all expected API endpoints exist and return valid responses."""
        with patch('src.rag.engine.rag_engine') as mock_rag_engine:
            mock_rag_engine.query.return_value = {
                "answer": "Test",
                "sources": [],
                "metadata": {}
            }
            
            with patch('src.api.main.rag_engine', mock_rag_engine):
                # Clear module cache
                modules_to_clear = [m for m in sys.modules.keys() if m.startswith('src.api.main')]
                for module in modules_to_clear:
                    if module in sys.modules:
                        del sys.modules[module]
                
                from src.api.main import app
                client = TestClient(app)
                
                # Test health endpoint
                response = client.get("/health")
                assert response.status_code == 200
                
                # Test query endpoint
                response = client.post("/query", json={"query": "test"})
                assert response.status_code == 200
                
                # Test validate manifest endpoint
                response = client.post("/validate-manifest", json={"manifest": "apiVersion: v1\nkind: Pod"})
                assert response.status_code == 200

    def test_invalid_requests_handled(self):
        """Test that invalid requests are handled properly."""
        with patch('src.rag.engine.rag_engine') as mock_rag_engine:
            with patch('src.api.main.rag_engine', mock_rag_engine):
                # Clear module cache
                modules_to_clear = [m for m in sys.modules.keys() if m.startswith('src.api.main')]
                for module in modules_to_clear:
                    if module in sys.modules:
                        del sys.modules[module]
                
                from src.api.main import app
                client = TestClient(app)
                
                # Test query endpoint with missing data
                response = client.post("/query", json={})
                assert response.status_code == 422  # Validation error
                
                # Test validate manifest endpoint with missing data
                response = client.post("/validate-manifest", json={})
                assert response.status_code == 422  # Validation error


if __name__ == '__main__':
    pytest.main([__file__]) 
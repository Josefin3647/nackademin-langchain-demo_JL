"""
Utilities for getting an embeddings using Ollama.

DO NOT CHANGE THIS FILE.
"""

import os
from langchain_ollama import OllamaEmbeddings
from util.models import AvailableModels, DEFAULT_MODEL


def get_embeddings(model: AvailableModels = DEFAULT_MODEL) -> OllamaEmbeddings:
    """Initialize and return an embeddings model using Ollama.
    
    Args:
        model: Model identifier for embeddings (AvailableModels enum)
    
    Returns:
        An initialized OllamaEmbeddings instance.
    """
    base_url = os.getenv("OLLAMA_BASE_URL")
    bearer_token = os.getenv("OLLAMA_BEARER_TOKEN")
    
    if not bearer_token:
        raise ValueError("OLLAMA_BEARER_TOKEN must be set in .env")
    
    return OllamaEmbeddings(
        model=model.value,
        base_url=base_url,
        client_kwargs={
            "headers": {"Authorization": f"Bearer {bearer_token}"}
        },
    )

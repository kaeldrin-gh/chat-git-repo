"""Embedding generation using HuggingFace sentence transformers."""

from typing import List

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from ..config import DEFAULT_EMBEDDING_MODEL, NUMPY_SEED, RANDOM_SEED


class CodeEmbedder:
    """Generates embeddings for code chunks using sentence transformers."""

    def __init__(self, model_name: str = DEFAULT_EMBEDDING_MODEL, device: str | None = None) -> None:
        """Initialize the embedder.
        
        Args:
            model_name: Name of the sentence transformer model.
            device: Device to run the model on. Auto-detected if None.
        """
        self.model_name = model_name
        
        # Set random seeds for reproducibility
        np.random.seed(NUMPY_SEED)
        torch.manual_seed(RANDOM_SEED)
        
        # Determine device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        
        # Load the model
        self.model = SentenceTransformer(model_name, device=device)
        
    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for a list of texts.
        
        Args:
            texts: List of text strings to embed.
            
        Returns:
            NumPy array of embeddings with shape (n_texts, embedding_dim).
        """
        if not texts:
            return np.array([])
        
        # Generate embeddings
        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,  # Normalize for cosine similarity
            show_progress_bar=len(texts) > 10,
        )
        
        return embeddings
    
    def embed_single(self, text: str) -> np.ndarray:
        """Generate embedding for a single text.
        
        Args:
            text: Text string to embed.
            
        Returns:
            NumPy array embedding with shape (embedding_dim,).
        """
        embeddings = self.embed_texts([text])
        return embeddings[0] if len(embeddings) > 0 else np.array([])
    
    @property
    def embedding_dimension(self) -> int:
        """Get the embedding dimension of the model."""
        return self.model.get_sentence_embedding_dimension()
    
    def get_model_info(self) -> dict:
        """Get information about the loaded model.
        
        Returns:
            Dictionary with model information.
        """
        return {
            "model_name": self.model_name,
            "device": str(self.device),
            "embedding_dimension": self.embedding_dimension,
            "max_sequence_length": getattr(self.model, "max_seq_length", "unknown"),
        }

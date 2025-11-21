"""
Embedding Generator Module - Phase 4.3

This module provides efficient batch embedding generation for job chunks
using sentence transformers or Ollama embedding models.

Author: Mert Alp Aydin
Date: 2025-11-21
"""

import logging
import numpy as np
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import time
import json

logger = logging.getLogger(__name__)


class ChunkEmbeddingGenerator:
    """
    Generates embeddings for job chunks efficiently.

    Supports multiple embedding models:
    - sentence-transformers (all-MiniLM-L6-v2, all-mpnet-base-v2)
    - Ollama embedding models
    """

    AVAILABLE_MODELS = {
        'all-MiniLM-L6-v2': {
            'dim': 384,
            'description': 'Fast, lightweight (22M params)',
            'type': 'sentence-transformers'
        },
        'all-mpnet-base-v2': {
            'dim': 768,
            'description': 'Higher quality (110M params)',
            'type': 'sentence-transformers'
        },
        'google/embeddinggemma-300m': {
            'dim': 768,
            'description': 'Best quality for retrieval (300M params), asymmetric encoding',
            'type': 'embeddinggemma'
        }
    }

    def __init__(
        self,
        model_name: str = 'google/embeddinggemma-300m',
        batch_size: int = 32,
        cache_dir: Optional[str] = None
    ):
        """
        Initialize the embedding generator.

        Args:
            model_name: Name of the embedding model to use
            batch_size: Batch size for embedding generation
            cache_dir: Directory to cache embeddings (optional)
        """
        self.model_name = model_name
        self.batch_size = batch_size
        self.model = None
        self.embedding_dim = None

        if cache_dir:
            self.cache_dir = Path(cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.cache_dir = None

        self._load_model()
        logger.info(f"ChunkEmbeddingGenerator initialized with {model_name} (dim={self.embedding_dim})")

    def _load_model(self):
        """Load the embedding model."""
        try:
            from sentence_transformers import SentenceTransformer

            self.model = SentenceTransformer(self.model_name)
            self.embedding_dim = self.model.get_sentence_embedding_dimension()
            logger.info(f"Loaded model {self.model_name} with dimension {self.embedding_dim}")

        except ImportError:
            logger.error("sentence-transformers not installed. Run: pip install sentence-transformers")
            raise

    def embed_text(self, text: str, is_query: bool = True) -> np.ndarray:
        """
        Generate embedding for a single text.

        Args:
            text: Input text
            is_query: If True, use query encoding (for EmbeddingGemma).
                      If False, use document encoding.

        Returns:
            Embedding vector as numpy array
        """
        model_info = self.AVAILABLE_MODELS.get(self.model_name, {})

        # Use asymmetric encoding for embeddinggemma
        if model_info.get('type') == 'embeddinggemma':
            if is_query:
                embedding = self.model.encode_query(text)
            else:
                embedding = self.model.encode_document(text)
        else:
            embedding = self.model.encode(text, convert_to_numpy=True)

        return np.array(embedding, dtype='float32')

    def embed_texts(self, texts: List[str], show_progress: bool = True, is_query: bool = False) -> np.ndarray:
        """
        Generate embeddings for multiple texts.

        Args:
            texts: List of input texts
            show_progress: Show progress bar
            is_query: If True, use query encoding. Default False (document encoding).

        Returns:
            Array of embeddings (n_texts, embedding_dim)
        """
        start_time = time.time()

        model_info = self.AVAILABLE_MODELS.get(self.model_name, {})

        # Use asymmetric encoding for embeddinggemma
        if model_info.get('type') == 'embeddinggemma':
            if is_query:
                embeddings = self.model.encode_query(texts, show_progress_bar=show_progress)
            else:
                embeddings = self.model.encode_document(texts, show_progress_bar=show_progress)
            embeddings = np.array(embeddings, dtype='float32')
        else:
            embeddings = self.model.encode(
                texts,
                batch_size=self.batch_size,
                show_progress_bar=show_progress,
                convert_to_numpy=True
            )

        elapsed = time.time() - start_time
        logger.info(f"Embedded {len(texts)} texts in {elapsed:.2f}s ({len(texts)/elapsed:.1f} texts/sec)")

        return embeddings

    def embed_chunks(
        self,
        chunks: List['JobChunk'],
        show_progress: bool = True
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Generate embeddings for job chunks.

        Args:
            chunks: List of JobChunk objects
            show_progress: Show progress bar

        Returns:
            Tuple of (embeddings array, list of chunk_ids)
        """
        texts = [chunk.text for chunk in chunks]
        chunk_ids = [chunk.chunk_id for chunk in chunks]

        embeddings = self.embed_texts(texts, show_progress)

        return embeddings, chunk_ids

    def save_embeddings(
        self,
        embeddings: np.ndarray,
        chunk_ids: List[str],
        path: str
    ):
        """
        Save embeddings and metadata to disk.

        Args:
            embeddings: Embedding array
            chunk_ids: List of chunk IDs
            path: Directory path for saving
        """
        save_path = Path(path)
        save_path.mkdir(parents=True, exist_ok=True)

        # Save embeddings as numpy
        np.save(save_path / "embeddings.npy", embeddings)

        # Save chunk IDs
        with open(save_path / "chunk_ids.json", 'w') as f:
            json.dump(chunk_ids, f)

        # Save metadata
        metadata = {
            'model_name': self.model_name,
            'embedding_dim': self.embedding_dim,
            'num_embeddings': len(chunk_ids)
        }
        with open(save_path / "embedding_metadata.json", 'w') as f:
            json.dump(metadata, f)

        logger.info(f"Saved {len(chunk_ids)} embeddings to {path}")

    def load_embeddings(self, path: str) -> Tuple[np.ndarray, List[str], Dict]:
        """
        Load embeddings from disk.

        Args:
            path: Directory path containing saved embeddings

        Returns:
            Tuple of (embeddings, chunk_ids, metadata)
        """
        load_path = Path(path)

        embeddings = np.load(load_path / "embeddings.npy")

        with open(load_path / "chunk_ids.json", 'r') as f:
            chunk_ids = json.load(f)

        with open(load_path / "embedding_metadata.json", 'r') as f:
            metadata = json.load(f)

        logger.info(f"Loaded {len(chunk_ids)} embeddings from {path}")
        return embeddings, chunk_ids, metadata

    def get_model_info(self) -> Dict:
        """Get information about the current model."""
        return {
            'model_name': self.model_name,
            'embedding_dim': self.embedding_dim,
            'batch_size': self.batch_size,
            'info': self.AVAILABLE_MODELS.get(self.model_name, {})
        }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Test embedding generation
    generator = ChunkEmbeddingGenerator(model_name='all-MiniLM-L6-v2')

    texts = [
        "Senior Software Engineer with Python experience",
        "Machine Learning Engineer position at tech company",
        "Data Scientist role requiring TensorFlow and PyTorch"
    ]

    embeddings = generator.embed_texts(texts, show_progress=False)

    print(f"\nModel Info: {generator.get_model_info()}")
    print(f"Embeddings shape: {embeddings.shape}")
    print(f"Sample embedding (first 10 dims): {embeddings[0][:10]}")

    # Test similarity
    from numpy.linalg import norm
    similarity = np.dot(embeddings[0], embeddings[1]) / (norm(embeddings[0]) * norm(embeddings[1]))
    print(f"\nSimilarity between texts 0 and 1: {similarity:.4f}")
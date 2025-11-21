"""
Vector Store Module - Phase 4.4 & 4.5

This module provides FAISS-based vector storage and retrieval for job chunks,
with Maximum Marginal Relevance (MMR) for diverse results and automatic
deduplication by job ID.

Author: Mert Alp Aydin
Date: 2025-11-21
"""

import logging
import numpy as np
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import json
import pickle

logger = logging.getLogger(__name__)


class ChunkVectorStore:
    """
    FAISS-based vector store for job chunk embeddings.

    Features:
    - Fast similarity search with FAISS
    - Maximum Marginal Relevance (MMR) for diverse results
    - Automatic deduplication by job ID
    - Persistence to disk
    """

    def __init__(
        self,
        embedding_dim: int = 768,
        index_type: str = 'flat'
    ):
        """
        Initialize the vector store.

        Args:
            embedding_dim: Dimension of embedding vectors
            index_type: Type of FAISS index ('flat', 'ivf')
        """
        self.embedding_dim = embedding_dim
        self.index_type = index_type
        self.index = None
        self.chunk_ids: List[str] = []
        self.chunk_metadata: Dict[str, Dict] = {}

        self._create_index()
        logger.info(f"ChunkVectorStore initialized: dim={embedding_dim}, type={index_type}")

    def _create_index(self):
        """Create FAISS index."""
        try:
            import faiss

            if self.index_type == 'flat':
                # Exact search - best for smaller datasets
                self.index = faiss.IndexFlatIP(self.embedding_dim)  # Inner product (cosine after normalization)
            elif self.index_type == 'ivf':
                # Approximate search - better for large datasets
                quantizer = faiss.IndexFlatIP(self.embedding_dim)
                self.index = faiss.IndexIVFFlat(quantizer, self.embedding_dim, 100)

            logger.info(f"Created FAISS {self.index_type} index")

        except ImportError:
            logger.error("FAISS not installed. Run: pip install faiss-cpu")
            raise

    def add_embeddings(
        self,
        embeddings: np.ndarray,
        chunk_ids: List[str],
        metadata: Optional[List[Dict]] = None
    ):
        """
        Add embeddings to the index.

        Args:
            embeddings: Array of embeddings (n, dim)
            chunk_ids: List of chunk IDs
            metadata: Optional list of metadata dicts for each chunk
        """
        import faiss

        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)

        # Train IVF index if needed
        if self.index_type == 'ivf' and not self.index.is_trained:
            self.index.train(embeddings)

        # Add to index
        self.index.add(embeddings)

        # Store chunk IDs and metadata
        start_idx = len(self.chunk_ids)
        self.chunk_ids.extend(chunk_ids)

        if metadata:
            for i, meta in enumerate(metadata):
                self.chunk_metadata[chunk_ids[i]] = meta

        logger.info(f"Added {len(chunk_ids)} embeddings to index (total: {self.index.ntotal})")

    def search_chunks(
        self,
        query_embedding: np.ndarray,
        k: int = 50
    ) -> List[Tuple[str, float]]:
        """
        Search for similar chunks.

        Args:
            query_embedding: Query embedding vector
            k: Number of results to return

        Returns:
            List of (chunk_id, score) tuples
        """
        import faiss

        # Normalize query
        query = query_embedding.reshape(1, -1).astype('float32')
        faiss.normalize_L2(query)

        # Search
        scores, indices = self.index.search(query, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0 and idx < len(self.chunk_ids):
                results.append((self.chunk_ids[idx], float(score)))

        return results

    def search_mmr(
        self,
        query_embedding: np.ndarray,
        k: int = 10,
        fetch_k: int = 50,
        lambda_mult: float = 0.5
    ) -> List[Tuple[str, float]]:
        """
        Search using Maximum Marginal Relevance for diverse results.

        MMR balances relevance to query with diversity among results.

        Args:
            query_embedding: Query embedding vector
            k: Number of final results to return
            fetch_k: Number of candidates to fetch for reranking
            lambda_mult: Balance between relevance (1) and diversity (0)

        Returns:
            List of (chunk_id, score) tuples
        """
        import faiss

        # Normalize query
        query = query_embedding.reshape(1, -1).astype('float32')
        faiss.normalize_L2(query)

        # Fetch candidates
        scores, indices = self.index.search(query, fetch_k)

        # Get embeddings for candidates
        candidates = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0 and idx < len(self.chunk_ids):
                # Reconstruct embedding from index
                embedding = np.zeros((1, self.embedding_dim), dtype='float32')
                self.index.reconstruct(int(idx), embedding[0])
                candidates.append({
                    'idx': idx,
                    'chunk_id': self.chunk_ids[idx],
                    'score': float(score),
                    'embedding': embedding[0]
                })

        if not candidates:
            return []

        # MMR selection
        selected = []
        remaining = candidates.copy()

        while len(selected) < k and remaining:
            mmr_scores = []

            for candidate in remaining:
                # Relevance to query
                relevance = candidate['score']

                # Max similarity to already selected
                if selected:
                    similarities = [
                        np.dot(candidate['embedding'], s['embedding'])
                        for s in selected
                    ]
                    max_sim = max(similarities)
                else:
                    max_sim = 0

                # MMR score
                mmr = lambda_mult * relevance - (1 - lambda_mult) * max_sim
                mmr_scores.append((candidate, mmr))

            # Select highest MMR score
            best = max(mmr_scores, key=lambda x: x[1])
            selected.append(best[0])
            remaining.remove(best[0])

        return [(s['chunk_id'], s['score']) for s in selected]

    def search_with_job_dedup(
        self,
        query_embedding: np.ndarray,
        job_id_mapper: 'JobIDMapper',
        k_jobs: int = 10,
        k_chunks: int = 50,
        use_mmr: bool = True,
        lambda_mult: float = 0.5
    ) -> List[Tuple['JobPosting', float, List[Tuple[str, float]]]]:
        """
        Search and return unique jobs with deduplication.

        Args:
            query_embedding: Query embedding vector
            job_id_mapper: JobIDMapper for deduplication
            k_jobs: Number of unique jobs to return
            k_chunks: Number of chunks to search
            use_mmr: Use MMR for diversity
            lambda_mult: MMR lambda parameter

        Returns:
            List of (JobPosting, best_score, matching_chunks) tuples
        """
        # Search for chunks
        if use_mmr:
            chunk_results = self.search_mmr(
                query_embedding, k=k_chunks, fetch_k=k_chunks * 2, lambda_mult=lambda_mult
            )
        else:
            chunk_results = self.search_chunks(query_embedding, k=k_chunks)

        # Group by job_id
        job_chunks: Dict[str, List[Tuple[str, float]]] = {}

        for chunk_id, score in chunk_results:
            job_id = job_id_mapper.get_job_id(chunk_id)
            if job_id:
                if job_id not in job_chunks:
                    job_chunks[job_id] = []
                job_chunks[job_id].append((chunk_id, score))

        # Calculate best score per job and sort
        job_scores = []
        for job_id, chunks in job_chunks.items():
            best_score = max(score for _, score in chunks)
            job = job_id_mapper.get_job(job_id)
            if job:
                job_scores.append((job, best_score, chunks))

        # Sort by best score and take top k_jobs
        job_scores.sort(key=lambda x: x[1], reverse=True)
        return job_scores[:k_jobs]

    def save(self, path: str):
        """
        Save index and metadata to disk.

        Args:
            path: Directory path for saving
        """
        import faiss

        save_path = Path(path)
        save_path.mkdir(parents=True, exist_ok=True)

        # Save FAISS index
        faiss.write_index(self.index, str(save_path / "faiss.index"))

        # Save chunk IDs
        with open(save_path / "chunk_ids.json", 'w') as f:
            json.dump(self.chunk_ids, f)

        # Save metadata
        with open(save_path / "chunk_metadata.pkl", 'wb') as f:
            pickle.dump(self.chunk_metadata, f)

        # Save config
        config = {
            'embedding_dim': self.embedding_dim,
            'index_type': self.index_type,
            'num_vectors': self.index.ntotal
        }
        with open(save_path / "config.json", 'w') as f:
            json.dump(config, f)

        logger.info(f"Vector store saved to {path}")

    def load(self, path: str):
        """
        Load index and metadata from disk.

        Args:
            path: Directory path containing saved files
        """
        import faiss

        load_path = Path(path)

        # Load FAISS index
        self.index = faiss.read_index(str(load_path / "faiss.index"))

        # Load chunk IDs
        with open(load_path / "chunk_ids.json", 'r') as f:
            self.chunk_ids = json.load(f)

        # Load metadata
        with open(load_path / "chunk_metadata.pkl", 'rb') as f:
            self.chunk_metadata = pickle.load(f)

        # Load config
        with open(load_path / "config.json", 'r') as f:
            config = json.load(f)
            self.embedding_dim = config['embedding_dim']
            self.index_type = config['index_type']

        logger.info(f"Vector store loaded from {path}: {self.index.ntotal} vectors")

    def get_stats(self) -> Dict:
        """Get statistics about the vector store."""
        return {
            'num_vectors': self.index.ntotal if self.index else 0,
            'num_chunk_ids': len(self.chunk_ids),
            'embedding_dim': self.embedding_dim,
            'index_type': self.index_type
        }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Test vector store
    print("Testing ChunkVectorStore...")

    store = ChunkVectorStore(embedding_dim=384)

    # Create sample embeddings
    np.random.seed(42)
    embeddings = np.random.randn(100, 384).astype('float32')
    chunk_ids = [f"chunk_{i}" for i in range(100)]

    # Add to store
    store.add_embeddings(embeddings, chunk_ids)

    # Test search
    query = np.random.randn(384).astype('float32')
    results = store.search_chunks(query, k=5)
    print(f"\nRegular search results:")
    for chunk_id, score in results:
        print(f"  {chunk_id}: {score:.4f}")

    # Test MMR search
    mmr_results = store.search_mmr(query, k=5, lambda_mult=0.5)
    print(f"\nMMR search results:")
    for chunk_id, score in mmr_results:
        print(f"  {chunk_id}: {score:.4f}")

    print(f"\nStats: {store.get_stats()}")
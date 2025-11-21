"""
Generate Embeddings for Job Database - Phase 4

This script generates embeddings for the full job database using the
winner configuration from Phase 4 experiments:
- Embedding Model: google/embeddinggemma-300m
- Vector Store: FAISS IndexFlatIP (768-dim)

Usage:
    python src/generate_job_embeddings.py [--limit N]

Author: Mert Alp Aydin
Date: 2025-11-21
"""

import sys
import argparse
import time
import pickle
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from job_chunker import JobChunker, load_jobs_from_jsonl
from job_id_mapper import JobIDMapper
from embedding_generator import ChunkEmbeddingGenerator
from vector_store import ChunkVectorStore
from logging_utils import setup_logger

logger = setup_logger(__name__)


def main(limit: int = None):
    """Generate embeddings for job database."""

    # Paths
    DATA_PATH = Path(__file__).parent.parent / "data" / "techmap-jobs_us_2023-05-05.json"
    OUTPUT_DIR = Path(__file__).parent.parent / "data" / "embeddings"

    logger.info("=" * 70)
    logger.info("JOB EMBEDDING GENERATION - Phase 4")
    logger.info("=" * 70)

    # Load jobs
    logger.info(f"Loading jobs from {DATA_PATH}...")
    start_time = time.time()
    jobs = load_jobs_from_jsonl(str(DATA_PATH), limit=limit)
    load_time = time.time() - start_time
    logger.info(f"Loaded {len(jobs)} jobs in {load_time:.1f}s")
    print(f"Loaded {len(jobs)} jobs in {load_time:.1f}s")

    # Chunk jobs
    logger.info("Chunking jobs...")
    start_time = time.time()
    chunker = JobChunker()
    mapper = JobIDMapper()
    all_chunks = []

    for i, job in enumerate(jobs):
        if i % 1000 == 0 and i > 0:
            print(f"  Chunked {i}/{len(jobs)} jobs...")
        chunks = chunker.chunk_job(job)
        mapper.add_job(job, chunks)
        all_chunks.extend(chunks)

    chunk_time = time.time() - start_time
    avg_chunks = len(all_chunks) / len(jobs) if jobs else 0
    logger.info(f"Created {len(all_chunks)} chunks in {chunk_time:.1f}s ({avg_chunks:.1f} avg/job)")
    print(f"Created {len(all_chunks)} chunks ({avg_chunks:.1f} avg/job)")

    # Generate embeddings
    logger.info("Initializing embedding generator (google/embeddinggemma-300m)...")
    print("\nInitializing embedding model (this may take a moment on first run)...")
    generator = ChunkEmbeddingGenerator()  # Uses embeddinggemma by default

    logger.info(f"Generating embeddings for {len(all_chunks)} chunks...")
    print(f"\nGenerating embeddings for {len(all_chunks)} chunks...")
    start_time = time.time()
    embeddings, chunk_ids = generator.embed_chunks(all_chunks, show_progress=True)
    embed_time = time.time() - start_time

    chunks_per_sec = len(all_chunks) / embed_time
    logger.info(f"Embeddings generated in {embed_time:.1f}s ({chunks_per_sec:.1f} chunks/sec)")
    print(f"Embeddings generated in {embed_time:.1f}s ({chunks_per_sec:.1f} chunks/sec)")

    # Build vector store
    logger.info("Building FAISS vector store...")
    print("\nBuilding FAISS vector store...")
    start_time = time.time()
    store = ChunkVectorStore(embedding_dim=generator.embedding_dim)
    store.add_embeddings(embeddings, chunk_ids)
    store_time = time.time() - start_time
    logger.info(f"Vector store built in {store_time:.1f}s")

    # Save everything
    logger.info(f"Saving to {OUTPUT_DIR}...")
    print(f"\nSaving to {OUTPUT_DIR}...")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Save embeddings
    generator.save_embeddings(embeddings, chunk_ids, str(OUTPUT_DIR))

    # Save vector store (to same directory - chunk_ids.json shared)
    store.save(str(OUTPUT_DIR))

    # Save job mapper
    mapper_path = OUTPUT_DIR / "job_mapper.pkl"
    with open(mapper_path, 'wb') as f:
        pickle.dump(mapper, f)
    logger.info(f"Saved job mapper to {mapper_path}")

    # Summary
    total_time = load_time + chunk_time + embed_time + store_time
    memory_mb = embeddings.nbytes / (1024 * 1024)

    logger.info("=" * 70)
    logger.info("EMBEDDING GENERATION COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Jobs: {len(jobs)}")
    logger.info(f"Chunks: {len(all_chunks)}")
    logger.info(f"Embedding dim: {generator.embedding_dim}")
    logger.info(f"Memory: {memory_mb:.1f} MB")
    logger.info(f"Total time: {total_time:.1f}s")
    logger.info(f"Output: {OUTPUT_DIR}")

    print("\n" + "=" * 50)
    print("EMBEDDING GENERATION COMPLETE")
    print("=" * 50)
    print(f"Jobs processed:    {len(jobs)}")
    print(f"Chunks created:    {len(all_chunks)}")
    print(f"Embedding dim:     {generator.embedding_dim}")
    print(f"Memory usage:      {memory_mb:.1f} MB")
    print(f"Total time:        {total_time:.1f}s")
    print(f"Output directory:  {OUTPUT_DIR}")
    print("=" * 50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate embeddings for job database")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of jobs (default: all)")
    args = parser.parse_args()

    main(limit=args.limit)
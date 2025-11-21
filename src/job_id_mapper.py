"""
Job ID Mapper Module - Phase 4.2

This module provides bidirectional mapping between chunk IDs and job IDs,
with automatic deduplication when retrieving search results.

Author: Mert Alp Aydin
Date: 2025-11-21
"""

import logging
import json
import pickle
from typing import List, Dict, Optional, Set, Tuple
from dataclasses import dataclass
from pathlib import Path

from job_chunker import JobChunk, JobPosting

logger = logging.getLogger(__name__)


@dataclass
class ChunkResult:
    """A search result containing chunk info and score."""
    chunk_id: str
    job_id: str
    score: float
    section: str
    text: str


class JobIDMapper:
    """
    Bidirectional mapping between chunk IDs and job IDs.

    Provides:
    - chunk_id -> job_id mapping
    - job_id -> list of chunk_ids mapping
    - Full job retrieval by chunk_id
    - Automatic deduplication of search results
    """

    def __init__(self):
        """Initialize the mapper."""
        # chunk_id -> job_id
        self.chunk_to_job: Dict[str, str] = {}

        # job_id -> list of chunk_ids
        self.job_to_chunks: Dict[str, List[str]] = {}

        # job_id -> JobPosting
        self.jobs: Dict[str, JobPosting] = {}

        # chunk_id -> JobChunk
        self.chunks: Dict[str, JobChunk] = {}

        logger.info("JobIDMapper initialized")

    def add_job(self, job: JobPosting, chunks: List[JobChunk]):
        """
        Add a job and its chunks to the mapper.

        Args:
            job: JobPosting object
            chunks: List of JobChunk objects for this job
        """
        # Store job
        self.jobs[job.job_id] = job

        # Initialize chunk list for job
        self.job_to_chunks[job.job_id] = []

        # Add each chunk
        for chunk in chunks:
            self.chunk_to_job[chunk.chunk_id] = job.job_id
            self.job_to_chunks[job.job_id].append(chunk.chunk_id)
            self.chunks[chunk.chunk_id] = chunk

    def add_jobs_batch(self, jobs_and_chunks: List[Tuple[JobPosting, List[JobChunk]]]):
        """
        Add multiple jobs and their chunks.

        Args:
            jobs_and_chunks: List of (JobPosting, List[JobChunk]) tuples
        """
        for job, chunks in jobs_and_chunks:
            self.add_job(job, chunks)

        logger.info(f"Added {len(jobs_and_chunks)} jobs with {len(self.chunks)} total chunks")

    def get_job_id(self, chunk_id: str) -> Optional[str]:
        """Get job_id for a chunk_id."""
        return self.chunk_to_job.get(chunk_id)

    def get_chunk_ids(self, job_id: str) -> List[str]:
        """Get all chunk_ids for a job_id."""
        return self.job_to_chunks.get(job_id, [])

    def get_job(self, job_id: str) -> Optional[JobPosting]:
        """Get full job posting by job_id."""
        return self.jobs.get(job_id)

    def get_job_by_chunk_id(self, chunk_id: str) -> Optional[JobPosting]:
        """Get full job posting by chunk_id."""
        job_id = self.get_job_id(chunk_id)
        return self.get_job(job_id) if job_id else None

    def get_chunk(self, chunk_id: str) -> Optional[JobChunk]:
        """Get chunk by chunk_id."""
        return self.chunks.get(chunk_id)

    def get_unique_jobs(
        self,
        chunk_results: List[ChunkResult],
        top_k: int = 10,
        aggregate: str = 'max'
    ) -> List[Tuple[JobPosting, float, List[ChunkResult]]]:
        """
        Deduplicate search results and return unique jobs.

        Args:
            chunk_results: List of ChunkResult from vector search
            top_k: Number of unique jobs to return
            aggregate: How to aggregate scores ('max', 'mean', 'sum')

        Returns:
            List of (JobPosting, aggregated_score, matching_chunks) tuples
        """
        # Group chunks by job_id
        job_chunks: Dict[str, List[ChunkResult]] = {}

        for result in chunk_results:
            job_id = result.job_id
            if job_id not in job_chunks:
                job_chunks[job_id] = []
            job_chunks[job_id].append(result)

        # Calculate aggregated scores
        job_scores: List[Tuple[str, float, List[ChunkResult]]] = []

        for job_id, chunks in job_chunks.items():
            scores = [c.score for c in chunks]

            if aggregate == 'max':
                agg_score = max(scores)
            elif aggregate == 'mean':
                agg_score = sum(scores) / len(scores)
            elif aggregate == 'sum':
                agg_score = sum(scores)
            else:
                agg_score = max(scores)

            job_scores.append((job_id, agg_score, chunks))

        # Sort by score and take top_k
        job_scores.sort(key=lambda x: x[1], reverse=True)
        job_scores = job_scores[:top_k]

        # Build result with full job data
        results = []
        for job_id, score, matching_chunks in job_scores:
            job = self.get_job(job_id)
            if job:
                results.append((job, score, matching_chunks))

        logger.debug(f"Deduplicated {len(chunk_results)} chunks to {len(results)} unique jobs")
        return results

    def save(self, path: str):
        """
        Save mapper state to disk.

        Args:
            path: Directory path to save files
        """
        save_path = Path(path)
        save_path.mkdir(parents=True, exist_ok=True)

        # Save mappings
        with open(save_path / "chunk_to_job.json", 'w') as f:
            json.dump(self.chunk_to_job, f)

        with open(save_path / "job_to_chunks.json", 'w') as f:
            json.dump(self.job_to_chunks, f)

        # Save jobs and chunks as pickle (for dataclass support)
        with open(save_path / "jobs.pkl", 'wb') as f:
            pickle.dump(self.jobs, f)

        with open(save_path / "chunks.pkl", 'wb') as f:
            pickle.dump(self.chunks, f)

        logger.info(f"JobIDMapper saved to {path}")

    def load(self, path: str):
        """
        Load mapper state from disk.

        Args:
            path: Directory path containing saved files
        """
        load_path = Path(path)

        with open(load_path / "chunk_to_job.json", 'r') as f:
            self.chunk_to_job = json.load(f)

        with open(load_path / "job_to_chunks.json", 'r') as f:
            self.job_to_chunks = json.load(f)

        with open(load_path / "jobs.pkl", 'rb') as f:
            self.jobs = pickle.load(f)

        with open(load_path / "chunks.pkl", 'rb') as f:
            self.chunks = pickle.load(f)

        logger.info(f"JobIDMapper loaded from {path}: {len(self.jobs)} jobs, {len(self.chunks)} chunks")

    def get_stats(self) -> Dict:
        """Get statistics about the mapper."""
        return {
            'total_jobs': len(self.jobs),
            'total_chunks': len(self.chunks),
            'avg_chunks_per_job': len(self.chunks) / len(self.jobs) if self.jobs else 0
        }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    from job_chunker import JobChunker, load_jobs_from_jsonl

    # Test with sample data
    data_path = Path(__file__).parent.parent / "data" / "techmap-jobs_us_2023-05-05.json"

    if data_path.exists():
        # Load and chunk jobs
        jobs = load_jobs_from_jsonl(str(data_path), limit=10)
        chunker = JobChunker()
        mapper = JobIDMapper()

        for job in jobs:
            chunks = chunker.chunk_job(job)
            mapper.add_job(job, chunks)

        print(f"\nMapper Stats: {mapper.get_stats()}")

        # Test retrieval
        test_chunk_id = list(mapper.chunks.keys())[0]
        print(f"\nTest chunk_id: {test_chunk_id}")
        print(f"Job ID: {mapper.get_job_id(test_chunk_id)}")
        print(f"Full Job: {mapper.get_job_by_chunk_id(test_chunk_id).title}")
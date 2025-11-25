"""
Matching Engine Module - Phase 5.1

This module provides resume-to-job matching functionality, combining
resume parsing, embedding generation, and vector search to find relevant jobs.

Author: Mert Alp Aydin
Date: 2025-11-21
"""

import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
import time

from resume_extractor import ResumeExtractor
from resume_schema import Resume
from embedding_generator import ChunkEmbeddingGenerator
from vector_store import ChunkVectorStore
from job_id_mapper import JobIDMapper
from job_chunker import JobPosting
from skill_extractor import SkillExtractor

logger = logging.getLogger(__name__)


@dataclass
class JobMatch:
    """Represents a matched job with relevance details."""
    job: JobPosting
    similarity_score: float
    matching_chunks: List[Tuple[str, float]] = field(default_factory=list)
    skill_match: Optional[Dict[str, Any]] = None
    rank: int = 0


@dataclass
class MatchingResult:
    """Complete result from the matching engine."""
    resume: Resume
    matches: List[JobMatch]
    total_jobs_searched: int
    search_time_seconds: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class MatchingEngine:
    """
    Resume-to-job matching engine.

    Combines:
    - Resume parsing and extraction
    - Embedding generation
    - Vector similarity search with MMR
    - Skill-based matching
    """

    def __init__(
        self,
        embeddings_path: str = "data/embeddings",
        embedding_model: str = "google/embeddinggemma-300m",
        use_mmr: bool = True,
        mmr_lambda: float = 0.5
    ):
        """
        Initialize the matching engine.

        Args:
            embeddings_path: Path to pre-computed job embeddings
            embedding_model: Model for generating resume embeddings
            use_mmr: Whether to use MMR for diverse results
            mmr_lambda: Lambda parameter for MMR (0.5 = balanced)
        """
        self.embeddings_path = Path(embeddings_path)
        self.use_mmr = use_mmr
        self.mmr_lambda = mmr_lambda

        logger.info("Initializing MatchingEngine...")

        # Initialize components
        self.resume_extractor = ResumeExtractor(model_name="gemma3:4b")
        self.embedding_generator = ChunkEmbeddingGenerator(model_name=embedding_model)
        self.skill_extractor = SkillExtractor(top_n=30)

        # Load vector store and mapper
        self.vector_store = ChunkVectorStore(embedding_dim=768)
        self.job_mapper = JobIDMapper()

        self._load_embeddings()

        logger.info("MatchingEngine initialized successfully")

    def _load_embeddings(self):
        """Load pre-computed job embeddings and mappings."""
        import pickle

        try:
            # Load vector store from flat directory structure
            self.vector_store.load(str(self.embeddings_path))

            # Load job mapper from pickle file
            mapper_path = self.embeddings_path / "job_mapper.pkl"
            with open(mapper_path, 'rb') as f:
                self.job_mapper = pickle.load(f)

            logger.info(f"Loaded {self.vector_store.get_stats()['num_vectors']} job embeddings")
            logger.info(f"Loaded {self.job_mapper.get_stats()['total_jobs']} jobs")
        except FileNotFoundError:
            logger.warning(f"No embeddings found at {self.embeddings_path}. Run generate_job_embeddings.py first.")

    def match_resume_pdf(
        self,
        pdf_path: Path,
        top_k: int = 10,
        include_skill_analysis: bool = True
    ) -> MatchingResult:
        """
        Match a PDF resume against job postings.

        Args:
            pdf_path: Path to resume PDF
            top_k: Number of top matches to return
            include_skill_analysis: Whether to include skill gap analysis

        Returns:
            MatchingResult with ranked job matches
        """
        start_time = time.time()
        logger.info(f"Matching resume: {pdf_path.name}")

        # Step 1: Extract resume
        extraction_result = self.resume_extractor.extract_from_pdf(pdf_path)

        if not extraction_result['success']:
            raise ValueError(f"Resume extraction failed: {extraction_result['error']}")

        resume = extraction_result['resume']
        logger.info(f"Extracted resume for: {resume.full_name}")

        # Step 2: Match using extracted resume
        return self.match_resume(
            resume=resume,
            top_k=top_k,
            include_skill_analysis=include_skill_analysis,
            start_time=start_time
        )

    def match_resume(
        self,
        resume: Resume,
        top_k: int = 10,
        include_skill_analysis: bool = True,
        start_time: Optional[float] = None
    ) -> MatchingResult:
        """
        Match an already-extracted resume against job postings.

        Args:
            resume: Extracted Resume object
            top_k: Number of top matches to return
            include_skill_analysis: Whether to include skill gap analysis
            start_time: Optional start time for timing

        Returns:
            MatchingResult with ranked job matches
        """
        if start_time is None:
            start_time = time.time()

        # Build resume text for embedding
        resume_text = self._build_resume_text(resume)

        # Step 2: Generate resume embedding
        logger.info("Generating resume embedding...")
        resume_embedding = self.embedding_generator.embed_text(resume_text, is_query=True)

        # Step 3: Search for similar jobs
        logger.info(f"Searching for top {top_k} matching jobs...")
        job_results = self.vector_store.search_with_job_dedup(
            query_embedding=resume_embedding,
            job_id_mapper=self.job_mapper,
            k_jobs=top_k,
            k_chunks=top_k * 5,
            use_mmr=self.use_mmr,
            lambda_mult=self.mmr_lambda
        )

        # Step 4: Build job matches with skill analysis
        matches = []
        for rank, (job, score, chunks) in enumerate(job_results, 1):
            match = JobMatch(
                job=job,
                similarity_score=score,
                matching_chunks=[(c, s) for c, s in chunks] if chunks else [],
                rank=rank
            )

            # Optional skill analysis
            if include_skill_analysis:
                match.skill_match = self._analyze_skills(resume, job)

            matches.append(match)

        end_time = time.time()
        search_time = end_time - start_time

        logger.info(f"Found {len(matches)} job matches in {search_time:.2f}s")

        return MatchingResult(
            resume=resume,
            matches=matches,
            total_jobs_searched=self.vector_store.get_stats()['num_vectors'],
            search_time_seconds=round(search_time, 2),
            metadata={
                'embedding_model': self.embedding_generator.model_name,
                'use_mmr': self.use_mmr,
                'mmr_lambda': self.mmr_lambda
            }
        )

    def _build_resume_text(self, resume: Resume) -> str:
        """Build searchable text from resume for embedding."""
        parts = []

        # Add title/summary
        if resume.summary:
            parts.append(resume.summary)

        # Add skills
        if resume.skills:
            parts.append(f"Skills: {', '.join(resume.skills)}")

        # Add experience
        for exp in resume.experience:
            exp_text = f"{exp.position} at {exp.company}"
            if exp.description:
                exp_text += f": {exp.description}"
            parts.append(exp_text)

        # Add education
        for edu in resume.education:
            edu_text = f"{edu.degree}"
            if edu.field:
                edu_text += f" in {edu.field}"
            edu_text += f" from {edu.institution}"
            parts.append(edu_text)

        return " ".join(parts)

    def _analyze_skills(self, resume: Resume, job: JobPosting) -> Dict[str, Any]:
        """
        Analyze skill match between resume and job using semantic similarity.

        Uses EmbeddingGemma to match skills semantically rather than exact string matching.
        Threshold: 0.65 cosine similarity (Decision #7, Experiment #7)
        """
        import numpy as np

        # Get resume skills (from extraction)
        resume_skills_raw = resume.skills
        resume_skills = [s.lower() for s in resume_skills_raw]

        # Extract job skills from description
        job_text = f"{job.title} {job.description}"
        job_skills_extracted = self.skill_extractor.extract_skills(
            job_text, method="rake_llm", top_n=10
        )
        job_skills = [s.lower() for s in job_skills_extracted]

        # If either list is empty, return early
        if not resume_skills or not job_skills:
            return {
                'matched_skills': [],
                'missing_skills': job_skills,
                'extra_skills': resume_skills,
                'match_percentage': 0.0,
                'total_job_skills': len(job_skills),
                'total_resume_skills': len(resume_skills)
            }

        # SEMANTIC MATCHING (Decision #7)
        # Threshold: 0.65 based on Experiment #7 results
        threshold = 0.65

        # Generate embeddings using EmbeddingGemma
        resume_embeddings = self.embedding_generator.model.encode(resume_skills)
        job_embeddings = self.embedding_generator.model.encode(job_skills)

        matched = []
        missing = []
        matched_resume_indices = set()

        # For each job skill, find best matching resume skill
        for job_idx, job_skill in enumerate(job_skills):
            job_emb = job_embeddings[job_idx]

            best_score = 0
            best_match_idx = -1

            for resume_idx, resume_skill in enumerate(resume_skills):
                resume_emb = resume_embeddings[resume_idx]

                # Cosine similarity
                similarity = np.dot(job_emb, resume_emb) / (
                    np.linalg.norm(job_emb) * np.linalg.norm(resume_emb)
                )

                if similarity > best_score:
                    best_score = similarity
                    best_match_idx = resume_idx

            # Match if above threshold
            if best_score >= threshold:
                matched.append(job_skill)
                matched_resume_indices.add(best_match_idx)
            else:
                missing.append(job_skill)

        # Extra skills = resume skills not matched
        extra = [s for idx, s in enumerate(resume_skills) if idx not in matched_resume_indices]

        match_pct = (len(matched) / len(job_skills) * 100) if job_skills else 0.0

        return {
            'matched_skills': sorted(matched),
            'missing_skills': sorted(missing),
            'extra_skills': sorted(extra),
            'match_percentage': round(match_pct, 1),
            'total_job_skills': len(job_skills),
            'total_resume_skills': len(resume_skills)
        }

    def rerank_by_skills(
        self,
        result: MatchingResult,
        skill_weight: float = 0.3
    ) -> MatchingResult:
        """
        Rerank matches by combining similarity and skill match scores.

        Args:
            result: Original matching result
            skill_weight: Weight for skill match (0-1)

        Returns:
            New MatchingResult with reranked matches
        """
        for match in result.matches:
            if match.skill_match:
                skill_score = match.skill_match['match_percentage'] / 100.0
                combined = (1 - skill_weight) * match.similarity_score + skill_weight * skill_score
                match.similarity_score = combined

        # Re-sort and update ranks
        result.matches.sort(key=lambda m: m.similarity_score, reverse=True)
        for i, match in enumerate(result.matches, 1):
            match.rank = i

        return result


def test_matching_engine():
    """Test the matching engine with sample resume."""
    logging.basicConfig(level=logging.INFO)

    print("=" * 70)
    print("MATCHING ENGINE TEST - Phase 5.1")
    print("=" * 70)

    # Initialize engine
    engine = MatchingEngine(embeddings_path="data/embeddings")

    # Test with sample resume
    samples_dir = Path(__file__).parent.parent / "data" / "cv_samples"
    pdf_files = list(samples_dir.glob("*.pdf"))

    if not pdf_files:
        print("[WARN] No sample PDFs found")
        return

    test_pdf = pdf_files[0]
    print(f"\nTesting with: {test_pdf.name}")

    try:
        result = engine.match_resume_pdf(test_pdf, top_k=5)

        print(f"\n[OK] Found {len(result.matches)} matches in {result.search_time_seconds}s")
        print(f"\nTop matches for {result.resume.full_name}:\n")

        for match in result.matches:
            print(f"{match.rank}. {match.job.title} @ {match.job.company}")
            print(f"   Score: {match.similarity_score:.3f}")
            if match.skill_match:
                print(f"   Skill Match: {match.skill_match['match_percentage']:.1f}%")
                print(f"   Matched: {', '.join(match.skill_match['matched_skills'][:5])}")
            print()

    except Exception as e:
        print(f"[FAIL] Error: {e}")


if __name__ == "__main__":
    test_matching_engine()
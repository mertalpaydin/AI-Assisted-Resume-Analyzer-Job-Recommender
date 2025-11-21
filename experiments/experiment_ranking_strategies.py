"""
Experiment: Ranking Strategy Comparison

This experiment tests different ranking strategies for resume-to-job matching.

Strategies tested:
1. Pure Similarity: Embedding-based cosine similarity only
2. Pure Skill Match: Skill match percentage only
3. Weighted Combined: similarity * (1-w) + skill_match * w (w = 0.3, 0.5, 0.7)
4. MMR Variants: Different lambda values (0.3, 0.5, 0.7, 1.0)

Metrics:
- Result diversity (unique companies, job types)
- Skill alignment quality
- Ranking stability across different resumes

Author: Mert Alp Aydin
Date: 2025-11-21
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import logging
import time
import json
from typing import List, Dict, Any
from datetime import datetime
from dataclasses import dataclass, asdict

from logging_utils import setup_logger
from matching_engine import MatchingEngine, MatchingResult, JobMatch
from resume_schema import Resume

logger = setup_logger(__name__, level=logging.DEBUG)


@dataclass
class RankingExperimentResult:
    """Results from a single ranking strategy test."""
    strategy_name: str
    params: Dict[str, Any]
    top_5_jobs: List[Dict[str, Any]]
    avg_similarity: float
    avg_skill_match: float
    unique_companies: int
    unique_titles: int
    execution_time: float


def calculate_diversity_metrics(matches: List[JobMatch]) -> Dict[str, Any]:
    """Calculate diversity metrics for a set of matches."""
    companies = set()
    titles = set()

    for m in matches[:10]:  # Top 10
        companies.add(m.job.company)
        titles.add(m.job.title.lower())

    return {
        'unique_companies': len(companies),
        'unique_titles': len(titles),
        'company_diversity': len(companies) / min(10, len(matches)) if matches else 0
    }


def run_ranking_experiment():
    """Run the ranking strategy comparison experiment."""

    logger.info("=" * 70)
    logger.info("EXPERIMENT: Ranking Strategy Comparison")
    logger.info(f"Date: {datetime.now().isoformat()}")
    logger.info("=" * 70)

    print("=" * 70)
    print("EXPERIMENT: Ranking Strategy Comparison")
    print("=" * 70)

    # Load sample resume for testing
    samples_dir = Path(__file__).parent.parent / "data" / "cv_samples"
    pdf_files = list(samples_dir.glob("*.pdf"))

    if not pdf_files:
        print("[ERROR] No sample PDFs found in data/cv_samples")
        return

    test_pdf = pdf_files[0]
    print(f"Testing with: {test_pdf.name}")

    # Step 1: Extract resume ONCE
    print("\n--- Extracting Resume (once) ---")
    from resume_extractor import ResumeExtractor
    extractor = ResumeExtractor(model_name="gemma3:4b")
    extraction_result = extractor.extract_from_pdf(test_pdf)

    if not extraction_result['success']:
        print(f"[ERROR] Resume extraction failed: {extraction_result['error']}")
        return

    resume = extraction_result['resume']
    print(f"Extracted resume for: {resume.full_name}")
    print(f"Skills: {resume.skills[:5]}...")

    results = {}

    # Initialize ONE engine with MMR support
    print("\n--- Initializing Matching Engine ---")
    engine = MatchingEngine(
        embeddings_path="data/embeddings",
        use_mmr=True,
        mmr_lambda=0.5
    )

    # Strategy 1: Pure Similarity (no MMR)
    print("\n--- Strategy 1: Pure Similarity (No MMR) ---")
    engine.use_mmr = False
    start = time.time()
    result_pure = engine.match_resume(resume, top_k=10, include_skill_analysis=True)
    time_pure = time.time() - start

    results['pure_similarity'] = {
        'matches': [(m.job.title, m.job.company, m.similarity_score,
                     m.skill_match.get('match_percentage', 0) if m.skill_match else 0)
                    for m in result_pure.matches[:5]],
        'diversity': calculate_diversity_metrics(result_pure.matches),
        'avg_similarity': sum(m.similarity_score for m in result_pure.matches) / len(result_pure.matches),
        'avg_skill_match': sum(m.skill_match.get('match_percentage', 0) if m.skill_match else 0
                               for m in result_pure.matches) / len(result_pure.matches),
        'time': time_pure
    }
    print(f"Time: {time_pure:.2f}s, Diversity: {results['pure_similarity']['diversity']['unique_companies']}")

    # Strategy 2: MMR Balanced (lambda=0.5)
    print("\n--- Strategy 2: MMR Balanced (lambda=0.5) ---")
    engine.use_mmr = True
    engine.mmr_lambda = 0.5
    start = time.time()
    result_mmr = engine.match_resume(resume, top_k=10, include_skill_analysis=True)
    time_mmr = time.time() - start

    results['mmr_balanced'] = {
        'matches': [(m.job.title, m.job.company, m.similarity_score,
                     m.skill_match.get('match_percentage', 0) if m.skill_match else 0)
                    for m in result_mmr.matches[:5]],
        'diversity': calculate_diversity_metrics(result_mmr.matches),
        'avg_similarity': sum(m.similarity_score for m in result_mmr.matches) / len(result_mmr.matches),
        'avg_skill_match': sum(m.skill_match.get('match_percentage', 0) if m.skill_match else 0
                               for m in result_mmr.matches) / len(result_mmr.matches),
        'time': time_mmr
    }
    print(f"Time: {time_mmr:.2f}s, Diversity: {results['mmr_balanced']['diversity']['unique_companies']}")

    # Strategy 3: MMR High Diversity (lambda=0.3)
    print("\n--- Strategy 3: MMR High Diversity (lambda=0.3) ---")
    engine.mmr_lambda = 0.3
    start = time.time()
    result_diverse = engine.match_resume(resume, top_k=10, include_skill_analysis=True)
    time_diverse = time.time() - start

    results['mmr_diverse'] = {
        'matches': [(m.job.title, m.job.company, m.similarity_score,
                     m.skill_match.get('match_percentage', 0) if m.skill_match else 0)
                    for m in result_diverse.matches[:5]],
        'diversity': calculate_diversity_metrics(result_diverse.matches),
        'avg_similarity': sum(m.similarity_score for m in result_diverse.matches) / len(result_diverse.matches),
        'avg_skill_match': sum(m.skill_match.get('match_percentage', 0) if m.skill_match else 0
                               for m in result_diverse.matches) / len(result_diverse.matches),
        'time': time_diverse
    }
    print(f"Time: {time_diverse:.2f}s, Diversity: {results['mmr_diverse']['diversity']['unique_companies']}")

    # Strategy 4: Skill-weighted Reranking (w=0.3)
    print("\n--- Strategy 4: Skill Reranking (w=0.3) ---")
    engine.mmr_lambda = 0.5
    start = time.time()
    result_rerank_03 = engine.match_resume(resume, top_k=10, include_skill_analysis=True)
    result_rerank_03 = engine.rerank_by_skills(result_rerank_03, skill_weight=0.3)
    time_rerank_03 = time.time() - start

    results['skill_rerank_03'] = {
        'matches': [(m.job.title, m.job.company, m.similarity_score,
                     m.skill_match.get('match_percentage', 0) if m.skill_match else 0)
                    for m in result_rerank_03.matches[:5]],
        'diversity': calculate_diversity_metrics(result_rerank_03.matches),
        'avg_similarity': sum(m.similarity_score for m in result_rerank_03.matches) / len(result_rerank_03.matches),
        'avg_skill_match': sum(m.skill_match.get('match_percentage', 0) if m.skill_match else 0
                               for m in result_rerank_03.matches) / len(result_rerank_03.matches),
        'time': time_rerank_03
    }
    print(f"Time: {time_rerank_03:.2f}s, Diversity: {results['skill_rerank_03']['diversity']['unique_companies']}")

    # Strategy 5: Skill-weighted Reranking (w=0.5)
    print("\n--- Strategy 5: Skill Reranking (w=0.5) ---")
    start = time.time()
    result_rerank_05 = engine.match_resume(resume, top_k=10, include_skill_analysis=True)
    result_rerank_05 = engine.rerank_by_skills(result_rerank_05, skill_weight=0.5)
    time_rerank_05 = time.time() - start

    results['skill_rerank_05'] = {
        'matches': [(m.job.title, m.job.company, m.similarity_score,
                     m.skill_match.get('match_percentage', 0) if m.skill_match else 0)
                    for m in result_rerank_05.matches[:5]],
        'diversity': calculate_diversity_metrics(result_rerank_05.matches),
        'avg_similarity': sum(m.similarity_score for m in result_rerank_05.matches) / len(result_rerank_05.matches),
        'avg_skill_match': sum(m.skill_match.get('match_percentage', 0) if m.skill_match else 0
                               for m in result_rerank_05.matches) / len(result_rerank_05.matches),
        'time': time_rerank_05
    }
    print(f"Time: {time_rerank_05:.2f}s, Diversity: {results['skill_rerank_05']['diversity']['unique_companies']}")

    # Summary
    print("\n" + "=" * 70)
    print("EXPERIMENT SUMMARY")
    print("=" * 70)

    logger.info("=" * 70)
    logger.info("EXPERIMENT SUMMARY")

    print(f"\n{'Strategy':<25} {'Avg Sim':>10} {'Avg Skill%':>12} {'Unique Co.':>12} {'Time':>8}")
    print("-" * 70)

    for name, data in results.items():
        print(f"{name:<25} {data['avg_similarity']:>10.3f} {data['avg_skill_match']:>12.1f} "
              f"{data['diversity']['unique_companies']:>12} {data['time']:>8.2f}s")

        logger.info(f"{name}: avg_sim={data['avg_similarity']:.3f}, "
                   f"avg_skill={data['avg_skill_match']:.1f}%, "
                   f"diversity={data['diversity']['unique_companies']}, "
                   f"time={data['time']:.2f}s")

    # Determine winner
    print("\n" + "=" * 70)
    print("ANALYSIS")
    print("=" * 70)

    # Best diversity
    best_diversity = max(results.items(), key=lambda x: x[1]['diversity']['unique_companies'])
    print(f"Best Diversity: {best_diversity[0]} ({best_diversity[1]['diversity']['unique_companies']} unique companies)")

    # Best skill match
    best_skill = max(results.items(), key=lambda x: x[1]['avg_skill_match'])
    print(f"Best Skill Match: {best_skill[0]} ({best_skill[1]['avg_skill_match']:.1f}%)")

    # Best similarity
    best_sim = max(results.items(), key=lambda x: x[1]['avg_similarity'])
    print(f"Best Similarity: {best_sim[0]} ({best_sim[1]['avg_similarity']:.3f})")

    logger.info("=" * 70)
    logger.info("EXPERIMENT COMPLETE")
    logger.info(f"Results: {json.dumps(results, indent=2, default=str)}")

    return results


if __name__ == "__main__":
    results = run_ranking_experiment()
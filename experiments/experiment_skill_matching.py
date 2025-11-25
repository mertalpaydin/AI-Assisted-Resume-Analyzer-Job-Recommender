"""
Experiment #7: Skill Matching Strategy Comparison
Phase 6.5 - Critical Fix

Objective: Fix 0% skill matching issue by testing different matching algorithms
Test Case: Peter Boyd (Data Scientist resume)

Current Problem:
- Exact string matching fails for similar skills
- Examples: "SQL database management" vs "sql queries" → No match
- Examples: "Machine learning frameworks" vs "machine learning" → No match

Options to Test:
1. Fuzzy String Matching (Levenshtein Distance)
2. Semantic Similarity (EmbeddingGemma)
3. Keyword/N-gram Overlap
4. Hybrid Approach (Fuzzy + Keyword)

Author: Mert Alp Aydin
Date: 2025-11-25
"""

import sys
import logging
from pathlib import Path
import time
import json
from typing import List, Dict, Set, Tuple, Any
from collections import defaultdict

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from resume_extractor import ResumeExtractor
from matching_engine import MatchingEngine
from skill_extractor import SkillExtractor, SkillNormalizer

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ====================
# SKILL MATCHING STRATEGIES
# ====================

class SkillMatcher:
    """Base class for skill matching strategies."""

    def __init__(self):
        self.normalizer = SkillNormalizer()

    def match_skills(
        self,
        resume_skills: List[str],
        job_skills: List[str],
        threshold: float = 0.7
    ) -> Dict[str, Any]:
        """
        Match resume skills against job skills.

        Returns:
            Dict with matched, missing, extra skills and match percentage
        """
        raise NotImplementedError


class ExactMatcher(SkillMatcher):
    """Current implementation - exact string matching."""

    def match_skills(
        self,
        resume_skills: List[str],
        job_skills: List[str],
        threshold: float = 0.7
    ) -> Dict[str, Any]:
        resume_set = set(s.lower() for s in resume_skills)
        job_set = set(s.lower() for s in job_skills)

        matched = resume_set & job_set
        missing = job_set - resume_set
        extra = resume_set - job_set

        match_pct = (len(matched) / len(job_set) * 100) if job_set else 0.0

        return {
            'matched': sorted(list(matched)),
            'missing': sorted(list(missing)),
            'extra': sorted(list(extra)),
            'match_percentage': round(match_pct, 1),
            'details': []
        }


class FuzzyMatcher(SkillMatcher):
    """Fuzzy string matching using Levenshtein distance."""

    def match_skills(
        self,
        resume_skills: List[str],
        job_skills: List[str],
        threshold: float = 0.7
    ) -> Dict[str, Any]:
        try:
            from fuzzywuzzy import fuzz
        except ImportError:
            logger.warning("fuzzywuzzy not installed. Install with: pip install fuzzywuzzy python-Levenshtein")
            return ExactMatcher().match_skills(resume_skills, job_skills, threshold)

        matched = []
        missing = []
        match_details = []

        # Normalize inputs
        resume_lower = [s.lower() for s in resume_skills]
        job_lower = [s.lower() for s in job_skills]

        # Track which resume skills were matched
        matched_resume_indices = set()

        # For each job skill, find best matching resume skill
        for job_skill in job_lower:
            best_score = 0
            best_match = None
            best_idx = -1

            for idx, resume_skill in enumerate(resume_lower):
                # Use token_sort_ratio for word-order independence
                score = fuzz.token_sort_ratio(resume_skill, job_skill) / 100.0

                if score > best_score:
                    best_score = score
                    best_match = resume_skill
                    best_idx = idx

            if best_score >= threshold:
                matched.append(job_skill)
                matched_resume_indices.add(best_idx)
                match_details.append({
                    'job_skill': job_skill,
                    'resume_skill': best_match,
                    'similarity': round(best_score, 3)
                })
            else:
                missing.append(job_skill)

        # Extra skills = resume skills not matched
        extra = [s for idx, s in enumerate(resume_lower) if idx not in matched_resume_indices]

        match_pct = (len(matched) / len(job_lower) * 100) if job_lower else 0.0

        return {
            'matched': sorted(matched),
            'missing': sorted(missing),
            'extra': sorted(extra),
            'match_percentage': round(match_pct, 1),
            'details': match_details
        }


class SemanticMatcher(SkillMatcher):
    """Semantic similarity using sentence transformers."""

    def __init__(self):
        super().__init__()
        from sentence_transformers import SentenceTransformer
        import numpy as np

        self.model = SentenceTransformer('google/embeddinggemma-300m')
        self.np = np
        logger.info("Loaded EmbeddingGemma for semantic matching")

    def match_skills(
        self,
        resume_skills: List[str],
        job_skills: List[str],
        threshold: float = 0.65
    ) -> Dict[str, Any]:
        import numpy as np

        matched = []
        missing = []
        match_details = []

        # Normalize and encode
        resume_lower = [s.lower() for s in resume_skills]
        job_lower = [s.lower() for s in job_skills]

        if not resume_lower or not job_lower:
            return {
                'matched': [],
                'missing': job_lower,
                'extra': resume_lower,
                'match_percentage': 0.0,
                'details': []
            }

        # Generate embeddings
        resume_embeddings = self.model.encode(resume_lower)
        job_embeddings = self.model.encode(job_lower)

        # Track matched resume skills
        matched_resume_indices = set()

        # For each job skill, find best semantic match
        for job_idx, job_skill in enumerate(job_lower):
            job_emb = job_embeddings[job_idx]

            best_score = 0
            best_match = None
            best_idx = -1

            for resume_idx, resume_skill in enumerate(resume_lower):
                resume_emb = resume_embeddings[resume_idx]

                # Cosine similarity
                similarity = np.dot(job_emb, resume_emb) / (
                    np.linalg.norm(job_emb) * np.linalg.norm(resume_emb)
                )

                if similarity > best_score:
                    best_score = similarity
                    best_match = resume_skill
                    best_idx = resume_idx

            if best_score >= threshold:
                matched.append(job_skill)
                matched_resume_indices.add(best_idx)
                match_details.append({
                    'job_skill': job_skill,
                    'resume_skill': best_match,
                    'similarity': round(float(best_score), 3)
                })
            else:
                missing.append(job_skill)

        # Extra skills
        extra = [s for idx, s in enumerate(resume_lower) if idx not in matched_resume_indices]

        match_pct = (len(matched) / len(job_lower) * 100) if job_lower else 0.0

        return {
            'matched': sorted(matched),
            'missing': sorted(missing),
            'extra': sorted(extra),
            'match_percentage': round(match_pct, 1),
            'details': match_details
        }


class KeywordMatcher(SkillMatcher):
    """Keyword/n-gram overlap matching."""

    def _extract_keywords(self, skill: str) -> Set[str]:
        """Extract keywords from skill string."""
        import re

        # Remove special chars, split on whitespace
        words = re.findall(r'\b\w+\b', skill.lower())

        # Filter stopwords
        stopwords = {'the', 'a', 'an', 'and', 'or', 'of', 'in', 'to', 'for', 'with'}
        keywords = {w for w in words if w not in stopwords and len(w) > 2}

        return keywords

    def match_skills(
        self,
        resume_skills: List[str],
        job_skills: List[str],
        threshold: float = 0.5
    ) -> Dict[str, Any]:
        matched = []
        missing = []
        match_details = []

        resume_lower = [s.lower() for s in resume_skills]
        job_lower = [s.lower() for s in job_skills]

        matched_resume_indices = set()

        # For each job skill, find best keyword overlap
        for job_skill in job_lower:
            job_keywords = self._extract_keywords(job_skill)

            if not job_keywords:  # Single word or all stopwords
                # Fallback to exact match
                if job_skill in resume_lower:
                    matched.append(job_skill)
                    matched_resume_indices.add(resume_lower.index(job_skill))
                    match_details.append({
                        'job_skill': job_skill,
                        'resume_skill': job_skill,
                        'similarity': 1.0
                    })
                else:
                    missing.append(job_skill)
                continue

            best_score = 0
            best_match = None
            best_idx = -1

            for idx, resume_skill in enumerate(resume_lower):
                resume_keywords = self._extract_keywords(resume_skill)

                if not resume_keywords:
                    continue

                # Jaccard similarity
                intersection = len(job_keywords & resume_keywords)
                union = len(job_keywords | resume_keywords)
                similarity = intersection / union if union > 0 else 0.0

                if similarity > best_score:
                    best_score = similarity
                    best_match = resume_skill
                    best_idx = idx

            if best_score >= threshold:
                matched.append(job_skill)
                matched_resume_indices.add(best_idx)
                match_details.append({
                    'job_skill': job_skill,
                    'resume_skill': best_match,
                    'similarity': round(best_score, 3)
                })
            else:
                missing.append(job_skill)

        extra = [s for idx, s in enumerate(resume_lower) if idx not in matched_resume_indices]

        match_pct = (len(matched) / len(job_lower) * 100) if job_lower else 0.0

        return {
            'matched': sorted(matched),
            'missing': sorted(missing),
            'extra': sorted(extra),
            'match_percentage': round(match_pct, 1),
            'details': match_details
        }


class HybridMatcher(SkillMatcher):
    """Hybrid: Fuzzy (70%) + Keyword (30%)."""

    def __init__(self):
        super().__init__()
        self.fuzzy = FuzzyMatcher()
        self.keyword = KeywordMatcher()

    def match_skills(
        self,
        resume_skills: List[str],
        job_skills: List[str],
        threshold: float = 0.65
    ) -> Dict[str, Any]:
        try:
            from fuzzywuzzy import fuzz
        except ImportError:
            logger.warning("fuzzywuzzy not available, using keyword-only")
            return self.keyword.match_skills(resume_skills, job_skills, threshold)

        matched = []
        missing = []
        match_details = []

        resume_lower = [s.lower() for s in resume_skills]
        job_lower = [s.lower() for s in job_skills]

        matched_resume_indices = set()

        # For each job skill, compute hybrid score
        for job_skill in job_lower:
            job_keywords = self.keyword._extract_keywords(job_skill)

            best_score = 0
            best_match = None
            best_idx = -1

            for idx, resume_skill in enumerate(resume_lower):
                # Fuzzy score (70% weight)
                fuzzy_score = fuzz.token_sort_ratio(resume_skill, job_skill) / 100.0 * 0.7

                # Keyword overlap score (30% weight)
                resume_keywords = self.keyword._extract_keywords(resume_skill)
                if job_keywords and resume_keywords:
                    intersection = len(job_keywords & resume_keywords)
                    union = len(job_keywords | resume_keywords)
                    keyword_score = (intersection / union if union > 0 else 0.0) * 0.3
                else:
                    keyword_score = 0.0

                # Combined score
                combined = fuzzy_score + keyword_score

                if combined > best_score:
                    best_score = combined
                    best_match = resume_skill
                    best_idx = idx

            if best_score >= threshold:
                matched.append(job_skill)
                matched_resume_indices.add(best_idx)
                match_details.append({
                    'job_skill': job_skill,
                    'resume_skill': best_match,
                    'similarity': round(best_score, 3)
                })
            else:
                missing.append(job_skill)

        extra = [s for idx, s in enumerate(resume_lower) if idx not in matched_resume_indices]

        match_pct = (len(matched) / len(job_lower) * 100) if job_lower else 0.0

        return {
            'matched': sorted(matched),
            'missing': sorted(missing),
            'extra': sorted(extra),
            'match_percentage': round(match_pct, 1),
            'details': match_details
        }


# ====================
# TEST RUNNER
# ====================

def run_experiment():
    """Run skill matching experiment on Peter Boyd (Data Scientist) resume."""

    print("=" * 80)
    print("EXPERIMENT #7: SKILL MATCHING STRATEGY COMPARISON")
    print("Test Case: Peter Boyd - Data Scientist Resume")
    print("=" * 80)
    print()

    # Test resume
    test_resume_path = Path(__file__).parent.parent / "data" / "cv_samples" / "Resume - Data Scientist.pdf"

    if not test_resume_path.exists():
        print(f"[ERROR] Resume not found: {test_resume_path}")
        return

    print(f"Test Resume: {test_resume_path.name}")
    print()

    # Step 1: Extract resume
    print("Step 1: Extracting resume...")
    extractor = ResumeExtractor(model_name="gemma3:4b")
    result = extractor.extract_from_pdf(test_resume_path)

    if not result['success']:
        print(f"[ERROR] Resume extraction failed: {result['error']}")
        return

    resume = result['resume']
    resume_skills = resume.skills

    print(f"[OK] Extracted {len(resume_skills)} skills from resume")
    print(f"  Skills: {', '.join(resume_skills[:10])}{'...' if len(resume_skills) > 10 else ''}")
    print()

    # Step 2: Get top matched job for testing
    print("Step 2: Finding top matched job...")
    engine = MatchingEngine(embeddings_path="data/embeddings")
    matching_result = engine.match_resume(resume, top_k=1, include_skill_analysis=True)

    if not matching_result.matches:
        print("[ERROR] No job matches found")
        return

    top_match = matching_result.matches[0]
    job = top_match.job

    print(f"[OK] Top Job: {job.title} @ {job.company}")
    print()

    # Step 3: Extract job skills
    print("Step 3: Extracting job skills...")
    skill_extractor = SkillExtractor(top_n=15)
    job_text = f"{job.title} {job.description}"
    job_skills = skill_extractor.extract_skills(job_text, method="rake_llm", top_n=15)

    print(f"[OK] Extracted {len(job_skills)} skills from job")
    print(f"  Skills: {', '.join(job_skills[:10])}{'...' if len(job_skills) > 10 else ''}")
    print()

    # Step 4: Test all matching strategies
    print("=" * 80)
    print("TESTING MATCHING STRATEGIES")
    print("=" * 80)
    print()

    strategies = [
        ("Exact Match (Current)", ExactMatcher(), 0.0),  # No threshold for exact
        ("Fuzzy Match (Levenshtein)", FuzzyMatcher(), 0.70),
        ("Semantic Match (EmbeddingGemma)", SemanticMatcher(), 0.65),
        ("Keyword Overlap", KeywordMatcher(), 0.50),
        ("Hybrid (Fuzzy+Keyword)", HybridMatcher(), 0.65),
    ]

    results = []

    for name, matcher, threshold in strategies:
        print(f"Testing: {name}")
        print(f"Threshold: {threshold:.2f}")

        start_time = time.time()
        match_result = matcher.match_skills(resume_skills, job_skills, threshold)
        elapsed = time.time() - start_time

        print(f"[OK] Match Percentage: {match_result['match_percentage']:.1f}%")
        print(f"  Matched: {len(match_result['matched'])} skills")
        print(f"  Missing: {len(match_result['missing'])} skills")
        print(f"  Latency: {elapsed*1000:.1f}ms")

        if match_result['details'] and len(match_result['details']) <= 5:
            print(f"  Examples:")
            for detail in match_result['details'][:3]:
                print(f"    '{detail['job_skill']}' <- '{detail['resume_skill']}' ({detail['similarity']:.2f})")

        print()

        results.append({
            'name': name,
            'threshold': threshold,
            'match_percentage': match_result['match_percentage'],
            'matched_count': len(match_result['matched']),
            'missing_count': len(match_result['missing']),
            'latency_ms': round(elapsed * 1000, 1),
            'matched_skills': match_result['matched'],
            'missing_skills': match_result['missing'],
            'details': match_result['details'][:5]  # Top 5 matches
        })

    # Step 5: Summary
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print()

    print(f"{'Strategy':<35} {'Match %':<12} {'Matched':<10} {'Missing':<10} {'Latency'}")
    print("-" * 80)

    for r in results:
        print(f"{r['name']:<35} {r['match_percentage']:>6.1f}%     {r['matched_count']:>3}/{len(job_skills):<6} {r['missing_count']:>3}/{len(job_skills):<6} {r['latency_ms']:>6.1f}ms")

    print()

    # Find winner
    best = max(results, key=lambda x: x['match_percentage'])
    print(f"[WINNER] {best['name']} with {best['match_percentage']:.1f}% match")
    print()

    # Step 6: Save results
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(exist_ok=True)

    output_file = output_dir / "experiment_7_skill_matching.json"

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            'experiment': 'Skill Matching Strategy Comparison',
            'date': time.strftime('%Y-%m-%d %H:%M:%S'),
            'test_case': {
                'resume': test_resume_path.name,
                'resume_skills_count': len(resume_skills),
                'resume_skills': resume_skills,
                'job_title': job.title,
                'job_company': job.company,
                'job_skills_count': len(job_skills),
                'job_skills': job_skills
            },
            'results': results,
            'winner': {
                'strategy': best['name'],
                'match_percentage': best['match_percentage'],
                'threshold': best['threshold']
            }
        }, f, indent=2)

    print(f"[OK] Results saved to: {output_file}")
    print()

    return results, best


if __name__ == "__main__":
    try:
        results, winner = run_experiment()

        print("=" * 80)
        print("EXPERIMENT COMPLETE")
        print("=" * 80)
        print()
        print("Next steps:")
        print("1. Review results in experiments/results/experiment_7_skill_matching.json")
        print("2. Document findings in logs/experiment_log.md")
        print("3. Update logs/decisions.md with chosen strategy")
        print("4. Implement winning strategy in matching_engine.py")

    except Exception as e:
        logger.error(f"Experiment failed: {e}", exc_info=True)
        print(f"\n[ERROR] Experiment failed: {e}")
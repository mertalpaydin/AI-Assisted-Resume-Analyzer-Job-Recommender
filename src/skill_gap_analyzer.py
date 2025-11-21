
"""
Skill Gap Analyzer Module - Phase 5.2

Provides detailed skill gap analysis between resumes and job postings.
Designed to work with ANY profession - not limited to IT/tech roles.

Uses dynamic skill categorization based on job context rather than
hardcoded skill databases.

Author: Mert Alp Aydin
Date: 2025-11-21
"""

import logging
from typing import List, Dict, Set, Optional, Any, Tuple
from dataclasses import dataclass, field
from difflib import SequenceMatcher

from skill_extractor import SkillExtractor, SkillNormalizer
from resume_schema import Resume
from job_chunker import JobPosting

logger = logging.getLogger(__name__)


@dataclass
class SkillGapResult:
    """Result of skill gap analysis."""
    # Core matching results
    matched_skills: List[str] = field(default_factory=list)
    missing_skills: List[str] = field(default_factory=list)
    extra_skills: List[str] = field(default_factory=list)
    match_percentage: float = 0.0

    # Fuzzy matches (similar but not exact)
    fuzzy_matches: List[Tuple[str, str, float]] = field(default_factory=list)  # (resume_skill, job_skill, similarity)

    # Priority rankings based on frequency in job text
    high_priority_missing: List[str] = field(default_factory=list)
    medium_priority_missing: List[str] = field(default_factory=list)
    low_priority_missing: List[str] = field(default_factory=list)

    # Statistics
    total_job_skills: int = 0
    total_resume_skills: int = 0

    # Job context
    job_title: str = ""
    job_company: str = ""


class SkillGapAnalyzer:
    """
    Analyzes skill gaps between resumes and job postings.

    This analyzer is profession-agnostic and works with any type of resume:
    - IT/Software Engineering
    - Healthcare/Medical
    - Finance/Accounting
    - Marketing/Sales
    - Education
    - Legal
    - Manufacturing
    - Any other profession

    Features:
    - Exact skill matching with normalization
    - Fuzzy matching for similar skills (e.g., "project management" vs "managing projects")
    - Priority ranking of missing skills based on job text frequency
    - No hardcoded skill databases - works dynamically with extracted skills
    """

    def __init__(
        self,
        skill_extractor: Optional[SkillExtractor] = None,
        fuzzy_threshold: float = 0.8
    ):
        """
        Initialize the analyzer.

        Args:
            skill_extractor: Optional pre-initialized SkillExtractor
            fuzzy_threshold: Minimum similarity score for fuzzy matching (0-1)
        """
        self.skill_extractor = skill_extractor or SkillExtractor(top_n=30)
        self.normalizer = SkillNormalizer()
        self.fuzzy_threshold = fuzzy_threshold

        logger.info("SkillGapAnalyzer initialized (profession-agnostic)")

    def analyze(
        self,
        resume: Resume,
        job: JobPosting,
        extract_job_skills: bool = True,
        use_fuzzy_matching: bool = True
    ) -> SkillGapResult:
        """
        Perform comprehensive skill gap analysis.

        Args:
            resume: Resume object with extracted skills
            job: Job posting to compare against
            extract_job_skills: Whether to extract skills from job text
            use_fuzzy_matching: Whether to use fuzzy matching for similar skills

        Returns:
            SkillGapResult with detailed analysis
        """
        # Get resume skills (normalized)
        resume_skills = self._normalize_skills(resume.skills)
        resume_skills_list = list(resume_skills)

        # Get job skills
        job_text = self._build_job_text(job)

        if extract_job_skills:
            job_skills_raw = self.skill_extractor.extract_skills(
                job_text, method="rake", top_n=30
            )
            job_skills = self._normalize_skills(job_skills_raw)
        else:
            job_skills = self._normalize_skills(job.requirements or [])

        job_skills_list = list(job_skills)

        # Exact matching
        exact_matched = resume_skills & job_skills
        exact_missing = job_skills - resume_skills
        exact_extra = resume_skills - job_skills

        # Fuzzy matching for remaining skills
        fuzzy_matches = []
        additional_matches = set()

        if use_fuzzy_matching and exact_missing and exact_extra:
            fuzzy_matches, additional_matches = self._find_fuzzy_matches(
                list(exact_extra), list(exact_missing)
            )

        # Final results
        all_matched = exact_matched | additional_matches
        final_missing = exact_missing - {m[1] for m in fuzzy_matches}
        final_extra = exact_extra - {m[0] for m in fuzzy_matches}

        # Calculate match percentage
        match_pct = (len(all_matched) / len(job_skills) * 100) if job_skills else 100.0

        # Prioritize missing skills
        high_priority, medium_priority, low_priority = self._prioritize_missing_skills(
            final_missing, job_text
        )

        return SkillGapResult(
            matched_skills=sorted(list(all_matched)),
            missing_skills=sorted(list(final_missing)),
            extra_skills=sorted(list(final_extra)),
            match_percentage=round(match_pct, 1),
            fuzzy_matches=fuzzy_matches,
            high_priority_missing=high_priority,
            medium_priority_missing=medium_priority,
            low_priority_missing=low_priority,
            total_job_skills=len(job_skills),
            total_resume_skills=len(resume_skills),
            job_title=job.title,
            job_company=job.company
        )

    def analyze_multiple_jobs(
        self,
        resume: Resume,
        jobs: List[JobPosting]
    ) -> Dict[str, SkillGapResult]:
        """
        Analyze skill gaps against multiple jobs.

        Args:
            resume: Resume to analyze
            jobs: List of job postings

        Returns:
            Dict mapping job_id to SkillGapResult
        """
        results = {}
        for job in jobs:
            results[job.job_id] = self.analyze(resume, job)
        return results

    def get_aggregate_skill_gaps(
        self,
        results: Dict[str, SkillGapResult]
    ) -> Dict[str, Any]:
        """
        Aggregate skill gaps across multiple job analyses.

        Useful for identifying which skills are most commonly missing
        across target job postings.

        Args:
            results: Dict of job_id -> SkillGapResult

        Returns:
            Aggregated analysis with skill frequencies
        """
        missing_frequency = {}
        matched_frequency = {}

        for job_id, result in results.items():
            for skill in result.missing_skills:
                missing_frequency[skill] = missing_frequency.get(skill, 0) + 1
            for skill in result.matched_skills:
                matched_frequency[skill] = matched_frequency.get(skill, 0) + 1

        # Sort by frequency
        sorted_missing = sorted(missing_frequency.items(), key=lambda x: x[1], reverse=True)
        sorted_matched = sorted(matched_frequency.items(), key=lambda x: x[1], reverse=True)

        return {
            'most_commonly_missing': sorted_missing[:10],
            'most_commonly_matched': sorted_matched[:10],
            'total_jobs_analyzed': len(results),
            'average_match_percentage': sum(r.match_percentage for r in results.values()) / len(results) if results else 0
        }

    def get_skill_development_recommendations(
        self,
        result: SkillGapResult
    ) -> List[Dict[str, Any]]:
        """
        Generate skill development recommendations based on gap analysis.

        Args:
            result: SkillGapResult from analyze()

        Returns:
            List of recommendation dicts with priority and reasoning
        """
        recommendations = []

        # High priority skills
        for skill in result.high_priority_missing:
            recommendations.append({
                'skill': skill,
                'priority': 'high',
                'reasoning': f"Frequently mentioned in job posting for {result.job_title}"
            })

        # Medium priority
        for skill in result.medium_priority_missing:
            recommendations.append({
                'skill': skill,
                'priority': 'medium',
                'reasoning': "Listed in job requirements"
            })

        # Low priority
        for skill in result.low_priority_missing:
            recommendations.append({
                'skill': skill,
                'priority': 'low',
                'reasoning': "Nice-to-have skill"
            })

        return recommendations

    def _normalize_skills(self, skills: List[str]) -> Set[str]:
        """Normalize a list of skills."""
        normalized = set()
        for s in skills:
            if s:
                # Basic normalization: lowercase, strip whitespace
                norm = s.lower().strip()
                # Try skill normalizer for known aliases
                norm = self.normalizer.normalize(norm)
                normalized.add(norm)
        return normalized

    def _build_job_text(self, job: JobPosting) -> str:
        """Build text for skill extraction from job posting."""
        parts = [job.title, job.description]
        if job.requirements:
            parts.extend(job.requirements)
        return " ".join(filter(None, parts))

    def _find_fuzzy_matches(
        self,
        resume_skills: List[str],
        job_skills: List[str]
    ) -> Tuple[List[Tuple[str, str, float]], Set[str]]:
        """
        Find fuzzy matches between resume and job skills.

        Args:
            resume_skills: Skills from resume (not exactly matched)
            job_skills: Skills from job (not exactly matched)

        Returns:
            Tuple of (fuzzy_matches list, set of matched job skills)
        """
        fuzzy_matches = []
        matched_job_skills = set()

        for resume_skill in resume_skills:
            best_match = None
            best_score = 0

            for job_skill in job_skills:
                if job_skill in matched_job_skills:
                    continue

                # Calculate similarity
                similarity = self._skill_similarity(resume_skill, job_skill)

                if similarity >= self.fuzzy_threshold and similarity > best_score:
                    best_match = job_skill
                    best_score = similarity

            if best_match:
                fuzzy_matches.append((resume_skill, best_match, round(best_score, 2)))
                matched_job_skills.add(best_match)

        return fuzzy_matches, matched_job_skills

    def _skill_similarity(self, skill1: str, skill2: str) -> float:
        """
        Calculate similarity between two skills.

        Uses SequenceMatcher for string similarity, which handles:
        - Word order differences ("project management" vs "management of projects")
        - Partial matches
        - Typos
        """
        # Direct sequence matching
        ratio = SequenceMatcher(None, skill1.lower(), skill2.lower()).ratio()

        # Also check if one contains the other
        s1, s2 = skill1.lower(), skill2.lower()
        if s1 in s2 or s2 in s1:
            containment_ratio = min(len(s1), len(s2)) / max(len(s1), len(s2))
            ratio = max(ratio, containment_ratio)

        return ratio

    def _prioritize_missing_skills(
        self,
        missing: Set[str],
        job_text: str
    ) -> Tuple[List[str], List[str], List[str]]:
        """
        Prioritize missing skills based on frequency in job text.

        Returns:
            Tuple of (high_priority, medium_priority, low_priority) lists
        """
        job_text_lower = job_text.lower()
        high, medium, low = [], [], []

        for skill in missing:
            # Count occurrences in job text
            frequency = job_text_lower.count(skill.lower())

            if frequency >= 3:
                high.append(skill)
            elif frequency >= 1:
                medium.append(skill)
            else:
                low.append(skill)

        return sorted(high), sorted(medium), sorted(low)


def test_skill_gap_analyzer():
    """Test the skill gap analyzer with various professions."""
    logging.basicConfig(level=logging.INFO)

    print("=" * 70)
    print("SKILL GAP ANALYZER TEST - Phase 5.2")
    print("Profession-agnostic testing")
    print("=" * 70)

    analyzer = SkillGapAnalyzer()

    # Test 1: IT/Software role
    print("\n--- Test 1: Software Engineer ---")

    class MockResumeIT:
        skills = ['python', 'javascript', 'react', 'sql', 'aws', 'docker', 'machine learning']

    class MockJobIT:
        job_id = "it-001"
        title = "Senior Software Engineer"
        company = "Tech Corp"
        description = """
        We are looking for a Senior Software Engineer with expertise in Python, Java, and React.
        Required: Python, Java, React, Node.js, PostgreSQL, AWS, Kubernetes, Docker.
        """
        requirements = ['Python', 'Java', 'React', 'Node.js', 'PostgreSQL', 'AWS', 'Kubernetes']
        location = "Remote"

    result_it = analyzer.analyze(MockResumeIT(), MockJobIT())
    print(f"Match: {result_it.match_percentage}%")
    print(f"Matched: {result_it.matched_skills}")
    print(f"Missing: {result_it.missing_skills}")

    # Test 2: Healthcare role
    print("\n--- Test 2: Registered Nurse ---")

    class MockResumeNurse:
        skills = ['patient care', 'medication administration', 'vital signs', 'electronic health records',
                  'CPR certified', 'wound care', 'IV therapy', 'communication']

    class MockJobNurse:
        job_id = "nurse-001"
        title = "Registered Nurse - ICU"
        company = "City Hospital"
        description = """
        ICU Registered Nurse needed. Must have critical care experience, ventilator management,
        medication administration, patient assessment, electronic health records (Epic preferred).
        BLS and ACLS certification required. Strong communication and teamwork skills.
        """
        requirements = ['critical care', 'ventilator management', 'BLS', 'ACLS', 'Epic']
        location = "New York"

    result_nurse = analyzer.analyze(MockResumeNurse(), MockJobNurse())
    print(f"Match: {result_nurse.match_percentage}%")
    print(f"Matched: {result_nurse.matched_skills}")
    print(f"Missing: {result_nurse.missing_skills}")
    print(f"Fuzzy matches: {result_nurse.fuzzy_matches}")

    # Test 3: Marketing role
    print("\n--- Test 3: Marketing Manager ---")

    class MockResumeMarketing:
        skills = ['social media marketing', 'content strategy', 'SEO', 'Google Analytics',
                  'brand management', 'copywriting', 'email marketing', 'Adobe Creative Suite']

    class MockJobMarketing:
        job_id = "mkt-001"
        title = "Digital Marketing Manager"
        company = "Brand Agency"
        description = """
        Digital Marketing Manager to lead our online presence. Experience with social media,
        content marketing, SEO/SEM, Google Ads, marketing automation, and analytics required.
        Must have strong project management and communication skills.
        """
        requirements = ['social media', 'SEO', 'Google Ads', 'marketing automation', 'analytics']
        location = "Chicago"

    result_mkt = analyzer.analyze(MockResumeMarketing(), MockJobMarketing())
    print(f"Match: {result_mkt.match_percentage}%")
    print(f"Matched: {result_mkt.matched_skills}")
    print(f"Missing: {result_mkt.missing_skills}")

    # Test 4: Finance role
    print("\n--- Test 4: Financial Analyst ---")

    class MockResumeFinance:
        skills = ['financial modeling', 'Excel', 'SQL', 'financial analysis', 'budgeting',
                  'forecasting', 'variance analysis', 'PowerPoint', 'SAP']

    class MockJobFinance:
        job_id = "fin-001"
        title = "Senior Financial Analyst"
        company = "Investment Bank"
        description = """
        Senior Financial Analyst for corporate finance team. Must have strong financial modeling,
        Excel (advanced), Bloomberg Terminal, financial statement analysis, valuation,
        M&A experience. CFA preferred. Strong presentation and communication skills.
        """
        requirements = ['financial modeling', 'Excel', 'Bloomberg', 'valuation', 'CFA']
        location = "New York"

    result_fin = analyzer.analyze(MockResumeFinance(), MockJobFinance())
    print(f"Match: {result_fin.match_percentage}%")
    print(f"Matched: {result_fin.matched_skills}")
    print(f"Missing: {result_fin.missing_skills}")

    print("\n" + "=" * 70)
    print("All profession tests completed!")


if __name__ == "__main__":
    test_skill_gap_analyzer()
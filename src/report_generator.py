"""
Report Generator Module - Phase 5.3 & 5.4

Generates comprehensive matching reports in multiple formats:
JSON, Markdown, and HTML.

Author: Mert Alp Aydin
Date: 2025-11-21
"""

import logging
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field, asdict
from datetime import datetime

from matching_engine import MatchingResult, JobMatch
from skill_gap_analyzer import SkillGapResult, SkillGapAnalyzer

logger = logging.getLogger(__name__)


@dataclass
class RecommendationResult:
    """
    Complete recommendation result for a single job match.
    JSON-serializable structure as specified in project plan.
    """
    job_id: str
    job_title: str
    company: str
    location: str
    similarity_score: float
    skill_match_percentage: float
    matched_skills: List[str]
    missing_skills: List[str]
    candidate_extra_skills: List[str]
    reasoning: str
    rank: int = 0


@dataclass
class MatchingReport:
    """Complete matching report with all data."""
    candidate_name: str
    candidate_email: Optional[str]
    candidate_skills: List[str]
    report_date: str
    total_jobs_searched: int
    search_time_seconds: float
    recommendations: List[RecommendationResult]
    skill_profile_summary: Dict[str, Any] = field(default_factory=dict)
    development_recommendations: List[str] = field(default_factory=list)


class ReportGenerator:
    """
    Generates matching reports in multiple formats.

    Supports:
    - JSON: For programmatic use
    - Markdown: For readability
    - HTML: For visualization
    """

    def __init__(self, skill_analyzer: Optional[SkillGapAnalyzer] = None):
        """Initialize report generator."""
        self.skill_analyzer = skill_analyzer or SkillGapAnalyzer()
        logger.info("ReportGenerator initialized")

    def generate_report(
        self,
        result: MatchingResult,
        detailed_skill_analysis: bool = True
    ) -> MatchingReport:
        """
        Generate a complete matching report from engine results.

        Args:
            result: MatchingResult from MatchingEngine
            detailed_skill_analysis: Whether to include detailed skill analysis

        Returns:
            MatchingReport object
        """
        resume = result.resume
        recommendations = []

        for match in result.matches:
            # Get skill data
            skill_match = match.skill_match or {}
            matched = skill_match.get('matched_skills', [])
            missing = skill_match.get('missing_skills', [])
            extra = skill_match.get('extra_skills', [])
            match_pct = skill_match.get('match_percentage', 0.0)

            # Generate reasoning
            reasoning = self._generate_reasoning(match, skill_match)

            rec = RecommendationResult(
                job_id=match.job.job_id,
                job_title=match.job.title,
                company=match.job.company,
                location=match.job.location or "Not specified",
                similarity_score=round(match.similarity_score, 3),
                skill_match_percentage=match_pct,
                matched_skills=matched[:10],  # Top 10
                missing_skills=missing[:10],
                candidate_extra_skills=extra[:10],
                reasoning=reasoning,
                rank=match.rank
            )
            recommendations.append(rec)

        # Build skill profile summary
        skill_profile = self._build_skill_profile(resume.skills, result.matches)

        # Generate development recommendations
        dev_recs = self._generate_development_recommendations(result.matches)

        return MatchingReport(
            candidate_name=resume.full_name,
            candidate_email=resume.email,
            candidate_skills=resume.skills,
            report_date=datetime.now().isoformat(),
            total_jobs_searched=result.total_jobs_searched,
            search_time_seconds=result.search_time_seconds,
            recommendations=recommendations,
            skill_profile_summary=skill_profile,
            development_recommendations=dev_recs
        )

    def _generate_reasoning(
        self,
        match: JobMatch,
        skill_match: Dict[str, Any]
    ) -> str:
        """Generate human-readable reasoning for a job match."""
        score = match.similarity_score
        match_pct = skill_match.get('match_percentage', 0)
        matched = skill_match.get('matched_skills', [])

        if score > 0.8 and match_pct > 70:
            quality = "Excellent"
        elif score > 0.6 and match_pct > 50:
            quality = "Good"
        elif score > 0.4:
            quality = "Moderate"
        else:
            quality = "Potential"

        reasoning = f"{quality} match based on semantic similarity ({score:.1%}) "
        reasoning += f"and skill alignment ({match_pct:.0f}%). "

        if matched:
            reasoning += f"Key matching skills: {', '.join(matched[:3])}."

        return reasoning

    def _build_skill_profile(
        self,
        resume_skills: List[str],
        matches: List[JobMatch]
    ) -> Dict[str, Any]:
        """Build a summary of the candidate's skill profile."""
        # Count how often each skill matches across jobs
        skill_demand = {}
        for match in matches:
            if match.skill_match:
                for skill in match.skill_match.get('matched_skills', []):
                    skill_demand[skill] = skill_demand.get(skill, 0) + 1

        # Most valuable skills (appear most in matches)
        sorted_skills = sorted(skill_demand.items(), key=lambda x: x[1], reverse=True)

        return {
            'total_skills': len(resume_skills),
            'most_valuable_skills': [s for s, _ in sorted_skills[:5]],
            'skill_demand_breakdown': dict(sorted_skills[:10]),
            'skills_list': resume_skills
        }

    def _generate_development_recommendations(
        self,
        matches: List[JobMatch]
    ) -> List[str]:
        """Generate skill development recommendations."""
        # Aggregate missing skills across all matches
        missing_frequency = {}
        for match in matches:
            if match.skill_match:
                for skill in match.skill_match.get('missing_skills', []):
                    missing_frequency[skill] = missing_frequency.get(skill, 0) + 1

        # Sort by frequency
        sorted_missing = sorted(missing_frequency.items(), key=lambda x: x[1], reverse=True)

        recommendations = []
        for skill, freq in sorted_missing[:5]:
            recommendations.append(
                f"Consider developing '{skill}' - missing in {freq}/{len(matches)} matched jobs"
            )

        return recommendations

    def to_json(self, report: MatchingReport) -> str:
        """Export report to JSON format."""
        data = {
            'candidate_name': report.candidate_name,
            'candidate_email': report.candidate_email,
            'candidate_skills': report.candidate_skills,
            'report_date': report.report_date,
            'total_jobs_searched': report.total_jobs_searched,
            'search_time_seconds': report.search_time_seconds,
            'recommendations': [asdict(r) for r in report.recommendations],
            'skill_profile_summary': report.skill_profile_summary,
            'development_recommendations': report.development_recommendations
        }
        return json.dumps(data, indent=2)

    def to_markdown(self, report: MatchingReport) -> str:
        """Export report to Markdown format."""
        lines = [
            f"# Job Match Report for {report.candidate_name}",
            f"",
            f"**Generated:** {report.report_date}",
            f"**Jobs Searched:** {report.total_jobs_searched}",
            f"**Search Time:** {report.search_time_seconds}s",
            f"",
            f"## Candidate Skills",
            f"",
            f"{', '.join(report.candidate_skills[:15])}{'...' if len(report.candidate_skills) > 15 else ''}",
            f"",
            f"## Top Job Matches",
            f""
        ]

        for rec in report.recommendations:
            lines.extend([
                f"### {rec.rank}. {rec.job_title}",
                f"**Company:** {rec.company}  ",
                f"**Location:** {rec.location}  ",
                f"**Similarity:** {rec.similarity_score:.1%}  ",
                f"**Skill Match:** {rec.skill_match_percentage:.0f}%",
                f"",
                f"**Matched Skills:** {', '.join(rec.matched_skills[:5])}",
                f"",
                f"**Missing Skills:** {', '.join(rec.missing_skills[:5])}",
                f"",
                f"_{rec.reasoning}_",
                f"",
                f"---",
                f""
            ])

        if report.development_recommendations:
            lines.extend([
                f"## Skill Development Recommendations",
                f""
            ])
            for rec in report.development_recommendations:
                lines.append(f"- {rec}")

        return "\n".join(lines)

    def to_html(self, report: MatchingReport) -> str:
        """Export report to HTML format."""
        html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Job Match Report - {report.candidate_name}</title>
    <style>
        body {{ font-family: Arial, sans-serif; max-width: 900px; margin: 0 auto; padding: 20px; }}
        h1 {{ color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px; }}
        h2 {{ color: #34495e; margin-top: 30px; }}
        .job-card {{ background: #f8f9fa; border-radius: 8px; padding: 20px; margin: 15px 0; border-left: 4px solid #3498db; }}
        .job-title {{ color: #2c3e50; margin: 0 0 10px 0; }}
        .company {{ color: #7f8c8d; font-size: 0.9em; }}
        .score {{ display: inline-block; padding: 5px 10px; border-radius: 4px; margin: 5px 5px 5px 0; }}
        .score-high {{ background: #27ae60; color: white; }}
        .score-med {{ background: #f39c12; color: white; }}
        .score-low {{ background: #e74c3c; color: white; }}
        .skills {{ margin: 10px 0; }}
        .skill-tag {{ display: inline-block; padding: 3px 8px; margin: 2px; border-radius: 3px; font-size: 0.85em; }}
        .matched {{ background: #d4edda; color: #155724; }}
        .missing {{ background: #f8d7da; color: #721c24; }}
        .reasoning {{ font-style: italic; color: #666; margin-top: 10px; }}
        .recommendations {{ background: #e8f4f8; padding: 15px; border-radius: 8px; }}
    </style>
</head>
<body>
    <h1>Job Match Report</h1>
    <p><strong>Candidate:</strong> {report.candidate_name}</p>
    <p><strong>Generated:</strong> {report.report_date}</p>
    <p><strong>Jobs Searched:</strong> {report.total_jobs_searched} | <strong>Time:</strong> {report.search_time_seconds}s</p>

    <h2>Top Job Matches</h2>
"""
        for rec in report.recommendations:
            score_class = 'score-high' if rec.similarity_score > 0.7 else ('score-med' if rec.similarity_score > 0.5 else 'score-low')
            matched_tags = ''.join(f'<span class="skill-tag matched">{s}</span>' for s in rec.matched_skills[:5])
            missing_tags = ''.join(f'<span class="skill-tag missing">{s}</span>' for s in rec.missing_skills[:5])

            html += f"""
    <div class="job-card">
        <h3 class="job-title">{rec.rank}. {rec.job_title}</h3>
        <p class="company">{rec.company} - {rec.location}</p>
        <span class="score {score_class}">Similarity: {rec.similarity_score:.1%}</span>
        <span class="score {'score-high' if rec.skill_match_percentage > 70 else 'score-med'}">Skills: {rec.skill_match_percentage:.0f}%</span>
        <div class="skills">
            <strong>Matched:</strong> {matched_tags}
        </div>
        <div class="skills">
            <strong>Missing:</strong> {missing_tags}
        </div>
        <p class="reasoning">{rec.reasoning}</p>
    </div>
"""

        if report.development_recommendations:
            html += """
    <h2>Skill Development Recommendations</h2>
    <div class="recommendations">
        <ul>
"""
            for rec in report.development_recommendations:
                html += f"            <li>{rec}</li>\n"
            html += """        </ul>
    </div>
"""

        html += """
</body>
</html>"""
        return html

    def save_report(
        self,
        report: MatchingReport,
        output_dir: str,
        formats: List[str] = None
    ) -> Dict[str, str]:
        """
        Save report in multiple formats.

        Args:
            report: MatchingReport to save
            output_dir: Directory to save files
            formats: List of formats ('json', 'md', 'html'). Default: all

        Returns:
            Dict mapping format to file path
        """
        if formats is None:
            formats = ['json', 'md', 'html']

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Generate filename from candidate name
        safe_name = "".join(c if c.isalnum() else "_" for c in report.candidate_name)
        base_name = f"job_matches_{safe_name}"

        saved_files = {}

        if 'json' in formats:
            json_path = output_path / f"{base_name}.json"
            json_path.write_text(self.to_json(report))
            saved_files['json'] = str(json_path)
            logger.info(f"Saved JSON report: {json_path}")

        if 'md' in formats:
            md_path = output_path / f"{base_name}.md"
            md_path.write_text(self.to_markdown(report))
            saved_files['md'] = str(md_path)
            logger.info(f"Saved Markdown report: {md_path}")

        if 'html' in formats:
            html_path = output_path / f"{base_name}.html"
            html_path.write_text(self.to_html(report))
            saved_files['html'] = str(html_path)
            logger.info(f"Saved HTML report: {html_path}")

        return saved_files


def test_report_generator():
    """Test the report generator."""
    logging.basicConfig(level=logging.INFO)
    print("Report generator module ready for testing with MatchingEngine results.")


if __name__ == "__main__":
    test_report_generator()
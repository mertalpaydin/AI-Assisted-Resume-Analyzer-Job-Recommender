"""
Job Skill Extraction Method Comparison Test
Phase 6.5 - Skill Quality Improvement

Objective: Compare 3 methods for extracting skills from job descriptions
1. gemma3:4b (LLM only)
2. granite4:micro (LLM only)
3. RAKE → LLM refinement (existing method, but max 10 not forced)

Test Case: Real matched job from Peter Boyd results
Metrics: Extracted skills quality, latency (4 runs, discard 1st, average rest)

Author: Mert Alp Aydin
Date: 2025-11-25
"""

import sys
import logging
from pathlib import Path
import time
import json
from typing import List, Dict, Any
import statistics

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from matching_engine import MatchingEngine
from skill_extractor import SkillExtractor
from langchain_ollama import ChatOllama

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ====================
# TEST METHODS
# ====================

class JobSkillExtractor:
    """Test different skill extraction methods for job descriptions."""

    def __init__(self):
        self.skill_extractor = SkillExtractor()

    def extract_gemma3(self, job_text: str, runs: int = 4) -> Dict[str, Any]:
        """Extract skills using gemma3:4b."""
        llm = ChatOllama(model="gemma3:4b", temperature=0.1)

        prompt = f"""Extract ONLY technical skills from this job description.
Focus on:
- Programming languages
- Technologies and frameworks
- Tools and platforms
- Technical methodologies
- Domain-specific technical skills

DO NOT include:
- Soft skills (communication, teamwork, etc.)
- Job duties or responsibilities
- Company benefits
- Work arrangements (remote, hybrid, etc.)

Return ONLY a JSON array of skills, nothing else.

Job Description:
{job_text}

JSON array of technical skills:"""

        latencies = []
        all_skills = []

        for i in range(runs):
            start = time.time()
            response = llm.invoke(prompt)
            elapsed = time.time() - start
            latencies.append(elapsed)

            # Parse response
            content = response.content.strip()
            if content.startswith('```'):
                # Remove markdown code blocks
                content = content.split('```')[1]
                if content.startswith('json'):
                    content = content[4:]
            content = content.strip()

            try:
                skills = json.loads(content)
                if isinstance(skills, list):
                    all_skills.append([s.lower().strip() for s in skills if isinstance(s, str)])
                else:
                    all_skills.append([])
            except:
                all_skills.append([])

            logger.info(f"  Run {i+1}/{runs}: {elapsed:.2f}s, {len(all_skills[-1])} skills")

        # Use skills from last run, average latency excluding first
        avg_latency = statistics.mean(latencies[1:]) if len(latencies) > 1 else latencies[0]

        return {
            'method': 'gemma3:4b',
            'skills': all_skills[-1],  # Last run
            'latency': avg_latency,
            'all_runs': all_skills,
            'all_latencies': latencies
        }

    def extract_granite4(self, job_text: str, runs: int = 4) -> Dict[str, Any]:
        """Extract skills using granite4:micro."""
        llm = ChatOllama(model="granite4:micro", temperature=0.1)

        prompt = f"""Extract ONLY technical skills from this job description.
Focus on:
- Programming languages
- Technologies and frameworks
- Tools and platforms
- Technical methodologies
- Domain-specific technical skills

DO NOT include:
- Soft skills (communication, teamwork, etc.)
- Job duties or responsibilities
- Company benefits
- Work arrangements (remote, hybrid, etc.)

Return ONLY a JSON array of skills, nothing else.

Job Description:
{job_text}

JSON array of technical skills:"""

        latencies = []
        all_skills = []

        for i in range(runs):
            start = time.time()
            response = llm.invoke(prompt)
            elapsed = time.time() - start
            latencies.append(elapsed)

            # Parse response
            content = response.content.strip()
            if content.startswith('```'):
                content = content.split('```')[1]
                if content.startswith('json'):
                    content = content[4:]
            content = content.strip()

            try:
                skills = json.loads(content)
                if isinstance(skills, list):
                    all_skills.append([s.lower().strip() for s in skills if isinstance(s, str)])
                else:
                    all_skills.append([])
            except:
                all_skills.append([])

            logger.info(f"  Run {i+1}/{runs}: {elapsed:.2f}s, {len(all_skills[-1])} skills")

        avg_latency = statistics.mean(latencies[1:]) if len(latencies) > 1 else latencies[0]

        return {
            'method': 'granite4:micro',
            'skills': all_skills[-1],
            'latency': avg_latency,
            'all_runs': all_skills,
            'all_latencies': latencies
        }

    def extract_rake_llm(self, job_text: str, runs: int = 4) -> Dict[str, Any]:
        """Extract skills using RAKE → granite4:micro refinement (existing method)."""
        from rake_nltk import Rake

        # Step 1: RAKE extraction (20-30 candidates)
        rake = Rake()
        rake.extract_keywords_from_text(job_text)
        rake_candidates = rake.get_ranked_phrases()[:30]  # Top 30 candidates

        logger.info(f"  RAKE extracted {len(rake_candidates)} candidates")

        # Step 2: LLM refinement to select best technical skills
        llm = ChatOllama(model="granite4:micro", temperature=0.1)

        prompt = f"""From the following list of keyword phrases extracted from a job description,
select ONLY the ones that are actual technical skills.

Focus on:
- Programming languages
- Technologies and frameworks
- Tools and platforms
- Technical methodologies
- Domain-specific technical skills

REMOVE:
- Soft skills
- Job duties or responsibilities
- Company benefits
- Work arrangements
- Generic business terms

Return UP TO 10 of the BEST technical skills as a JSON array.
If there are fewer than 10 valid skills, return only the valid ones.

Candidates:
{json.dumps(rake_candidates, indent=2)}

JSON array of technical skills (max 10, can be less):"""

        latencies = []
        all_skills = []

        for i in range(runs):
            start = time.time()
            response = llm.invoke(prompt)
            elapsed = time.time() - start
            latencies.append(elapsed)

            # Parse response
            content = response.content.strip()
            if content.startswith('```'):
                content = content.split('```')[1]
                if content.startswith('json'):
                    content = content[4:]
            content = content.strip()

            try:
                skills = json.loads(content)
                if isinstance(skills, list):
                    all_skills.append([s.lower().strip() for s in skills if isinstance(s, str)])
                else:
                    all_skills.append([])
            except:
                all_skills.append([])

            logger.info(f"  Run {i+1}/{runs}: {elapsed:.2f}s, {len(all_skills[-1])} skills")

        avg_latency = statistics.mean(latencies[1:]) if len(latencies) > 1 else latencies[0]

        return {
            'method': 'RAKE -> granite4:micro',
            'rake_candidates': rake_candidates[:15],  # Show first 15 for reference
            'skills': all_skills[-1],
            'latency': avg_latency,
            'all_runs': all_skills,
            'all_latencies': latencies
        }


# ====================
# TEST RUNNER
# ====================

def run_test():
    """Run comparison test on real job posting."""

    print("=" * 80)
    print("JOB SKILL EXTRACTION METHOD COMPARISON")
    print("=" * 80)
    print()

    # Load a real matched job from Peter Boyd results
    print("Loading test job posting...")

    # Use matching engine to get a real job
    engine = MatchingEngine(embeddings_path="data/embeddings")

    # Get job from vector store
    # Using job_id from Peter Boyd's 3rd match: Data Analyst at OrangePeople
    test_job_idx = 2  # 0-indexed, so 3rd job

    # Load one of Peter's matched jobs
    from resume_extractor import ResumeExtractor

    resume_path = Path("data/cv_samples/Resume - Data Scientist.pdf")
    extractor = ResumeExtractor()
    result = extractor.extract_from_pdf(resume_path)
    resume = result['resume']

    # Get matches
    matches = engine.match_resume(resume, top_k=5, include_skill_analysis=False)
    test_job = matches.matches[test_job_idx].job

    print(f"\n[TEST JOB]")
    print(f"Title: {test_job.title}")
    print(f"Company: {test_job.company}")
    print(f"Description length: {len(test_job.description)} chars")
    print()

    job_text = f"{test_job.title}\n\n{test_job.description}"

    # Initialize extractor
    skill_extractor = JobSkillExtractor()

    # Run tests
    results = []

    print("=" * 80)
    print("METHOD 1: gemma3:4b (LLM Only)")
    print("=" * 80)
    result1 = skill_extractor.extract_gemma3(job_text, runs=4)
    results.append(result1)
    print()

    print("=" * 80)
    print("METHOD 2: granite4:micro (LLM Only)")
    print("=" * 80)
    result2 = skill_extractor.extract_granite4(job_text, runs=4)
    results.append(result2)
    print()

    print("=" * 80)
    print("METHOD 3: RAKE -> granite4:micro (Hybrid)")
    print("=" * 80)
    result3 = skill_extractor.extract_rake_llm(job_text, runs=4)
    results.append(result3)
    print()

    # Display results
    print("=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    print()

    for result in results:
        print(f"Method: {result['method']}")
        print(f"Average Latency: {result['latency']:.2f}s (excluding 1st run)")
        print(f"Latencies: {[f'{l:.2f}s' for l in result['all_latencies']]}")
        print(f"Skills Count: {len(result['skills'])}")
        print(f"Extracted Skills:")
        for skill in result['skills']:
            print(f"  - {skill}")

        if 'rake_candidates' in result:
            print(f"\nRAKE Candidates (first 15):")
            for candidate in result['rake_candidates']:
                print(f"  - {candidate}")

        print()
        print("-" * 80)
        print()

    # Save results
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(exist_ok=True)

    output_file = output_dir / "job_skill_extraction_comparison.json"

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            'test_date': time.strftime('%Y-%m-%d %H:%M:%S'),
            'test_job': {
                'title': test_job.title,
                'company': test_job.company,
                'description_length': len(test_job.description)
            },
            'results': results
        }, f, indent=2)

    print(f"[SAVED] Results saved to: {output_file}")
    print()

    # Comparison table
    print("=" * 80)
    print("COMPARISON TABLE")
    print("=" * 80)
    print()

    print(f"{'Method':<30} {'Latency':<12} {'Skills':<10} {'Quality'}")
    print("-" * 80)

    for result in results:
        method_name = result['method']
        latency = f"{result['latency']:.2f}s"
        count = f"{len(result['skills'])}"
        print(f"{method_name:<30} {latency:<12} {count:<10} [Review Below]")

    print()
    print("=" * 80)
    print("NEXT STEPS")
    print("=" * 80)
    print()
    print("1. Review the extracted skills for each method")
    print("2. Evaluate quality: Are they technical skills? Any noise?")
    print("3. Consider latency trade-off")
    print("4. Decide together which method to use")
    print()


if __name__ == "__main__":
    try:
        run_test()
    except Exception as e:
        logger.error(f"Test failed: {e}", exc_info=True)
        print(f"\n[ERROR] Test failed: {e}")
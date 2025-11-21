"""
Experiment: RAKE + Ollama Skill Extraction

This experiment tests different Ollama models for extracting atomic skills
from RAKE-extracted keyphrases.

Pipeline: Job Text -> RAKE (keyphrases) -> Ollama (atomic skills)

Models tested:
- granite4:micro (fastest, smallest)
- llama3.2:3b (balanced)
- llama3.1:8b (larger llama model)
- mistral:7b (new model to test)
- gemma3:4b (highest quality from Phase 2)
- gemma3:12b-it-q4_K_M  (larger gemma model)

Metrics:
- Quality: How well does it extract individual, meaningful skills
- Speed: Latency per extraction
- Consistency: Same results across runs

Author: Mert Alp Aydin
Date: 2025-11-21
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import time
import json
import logging
from typing import List, Dict, Any
from datetime import datetime
from rake_nltk import Rake
from langchain_ollama import ChatOllama

from logging_utils import setup_logger

# Setup logging
logger = setup_logger(__name__, level=logging.DEBUG)

# Test job descriptions from different professions
TEST_JOBS = [
    {
        "id": "devops-001",
        "profession": "DevOps Engineer",
        "text": """
        Senior DevOps Engineer - AWS, Docker, Kubernetes
        We are looking for a Senior DevOps Engineer with experience in:
        - AWS, Azure, or GCP cloud platforms
        - Docker and Kubernetes container orchestration
        - CI/CD pipelines (Jenkins, GitLab CI, GitHub Actions)
        - Infrastructure as Code (Terraform, Ansible)
        - Linux system administration
        - Python or Bash scripting
        - Monitoring tools (Prometheus, Grafana, ELK)
        Requirements: 5+ years DevOps experience, strong communication skills
        """
    },
    {
        "id": "nurse-001",
        "profession": "Registered Nurse",
        "text": """
        ICU Registered Nurse - Critical Care
        City Hospital is seeking an experienced ICU Registered Nurse.
        Requirements:
        - Current RN license in state
        - BLS and ACLS certification
        - 2+ years critical care or ICU experience
        - Ventilator management experience
        - Electronic health records (Epic or Cerner)
        - Patient assessment and monitoring
        - Medication administration
        - Strong communication and teamwork
        """
    },
    {
        "id": "marketing-001",
        "profession": "Marketing Manager",
        "text": """
        Digital Marketing Manager
        Lead our digital marketing initiatives across multiple channels.
        Required skills:
        - SEO/SEM and Google Analytics
        - Social media marketing (Facebook, Instagram, LinkedIn)
        - Content marketing and copywriting
        - Email marketing and marketing automation (HubSpot, Marketo)
        - PPC advertising and Google Ads
        - A/B testing and conversion optimization
        - Project management and team leadership
        - Budget management and ROI analysis
        """
    },
    {
        "id": "finance-001",
        "profession": "Financial Analyst",
        "text": """
        Senior Financial Analyst - Corporate Finance
        Join our corporate finance team for financial planning and analysis.
        Requirements:
        - Financial modeling and forecasting
        - Advanced Excel (pivot tables, VLOOKUP, macros)
        - Bloomberg Terminal experience
        - Financial statement analysis
        - Budgeting and variance analysis
        - SQL for data extraction
        - PowerPoint for executive presentations
        - CFA or MBA preferred
        """
    }
]

# Prompt template for skill extraction
SKILL_EXTRACTION_PROMPT = """Extract individual, atomic skills from the following keyphrases.

Rules:
1. Return ONLY a JSON array of skill strings
2. Each skill should be 1-3 words maximum
3. Remove generic words like "experience", "required", "strong"
4. Normalize to lowercase
5. Keep technical terms, tools, certifications, and specific competencies
6. Do NOT include job titles or years of experience

Keyphrases to extract skills from:
{keyphrases}

Return ONLY a valid JSON array like: ["skill1", "skill2", "skill3"]

JSON array:"""


def extract_keyphrases_rake(text: str, top_n: int = 20) -> List[str]:
    """Extract keyphrases using RAKE."""
    logger.debug(f"Extracting keyphrases from text ({len(text)} chars)")

    rake = Rake(max_length=4)
    rake.extract_keywords_from_text(text)
    phrases = rake.get_ranked_phrases()[:top_n]

    logger.debug(f"RAKE extracted {len(phrases)} keyphrases")
    logger.debug(f"Keyphrases: {phrases}")

    return phrases


def extract_skills_ollama(
    keyphrases: List[str],
    model_name: str,
    temperature: float = 0.1
) -> Dict[str, Any]:
    """
    Extract atomic skills from keyphrases using Ollama.

    Returns dict with skills list, latency, and any errors.
    """
    logger.debug(f"Initializing Ollama model: {model_name}")
    llm = ChatOllama(model=model_name, temperature=temperature)

    prompt = SKILL_EXTRACTION_PROMPT.format(
        keyphrases="\n".join(f"- {p}" for p in keyphrases)
    )
    logger.debug(f"Prompt length: {len(prompt)} chars")

    start_time = time.time()
    try:
        logger.debug(f"Invoking {model_name}...")
        response = llm.invoke(prompt)
        latency = time.time() - start_time
        logger.debug(f"Response received in {latency:.2f}s")

        # Parse JSON response
        response_text = response.content.strip()
        logger.debug(f"Raw response: {response_text[:300]}...")

        # Find JSON array in response
        if '[' in response_text and ']' in response_text:
            json_start = response_text.find('[')
            json_end = response_text.rfind(']') + 1
            json_text = response_text[json_start:json_end]

            logger.debug(f"Extracted JSON: {json_text}")
            skills = json.loads(json_text)

            # Ensure all skills are strings and clean
            skills = [str(s).lower().strip() for s in skills if s]
            logger.debug(f"Parsed {len(skills)} skills: {skills}")

            return {
                'success': True,
                'skills': skills,
                'latency': round(latency, 2),
                'raw_response': response_text[:200]
            }
        else:
            logger.warning(f"No JSON array found in response from {model_name}")
            return {
                'success': False,
                'skills': [],
                'latency': round(latency, 2),
                'error': 'No JSON array found in response',
                'raw_response': response_text[:200]
            }

    except json.JSONDecodeError as e:
        logger.error(f"JSON parse error from {model_name}: {e}")
        return {
            'success': False,
            'skills': [],
            'latency': round(time.time() - start_time, 2),
            'error': f'JSON parse error: {e}',
            'raw_response': response_text[:200] if 'response_text' in locals() else ''
        }
    except Exception as e:
        logger.error(f"Error from {model_name}: {e}")
        return {
            'success': False,
            'skills': [],
            'latency': round(time.time() - start_time, 2),
            'error': str(e)
        }


def run_experiment():
    """Run the skill extraction experiment across all models and jobs."""

    models = [
        'granite4:micro', 'llama3.2:3b', 'llama3.1:8b',
        'mistral:7b', 'gemma3:4b', 'gemma3:12b-it-q4_K_M',
    ]

    results = {model: [] for model in models}

    logger.info("=" * 70)
    logger.info("EXPERIMENT: RAKE + Ollama Skill Extraction")
    logger.info(f"Date: {datetime.now().isoformat()}")
    logger.info(f"Models: {models}")
    logger.info(f"Test jobs: {len(TEST_JOBS)}")
    logger.info("=" * 70)

    print("=" * 70)
    print("EXPERIMENT: RAKE + Ollama Skill Extraction")
    print("=" * 70)

    for job in TEST_JOBS:
        logger.info(f"Processing job: {job['profession']} ({job['id']})")
        print(f"\n--- Job: {job['profession']} ({job['id']}) ---")

        # Step 1: Extract keyphrases with RAKE
        keyphrases = extract_keyphrases_rake(job['text'])
        logger.info(f"RAKE extracted {len(keyphrases)} keyphrases")
        print(f"RAKE keyphrases ({len(keyphrases)}): {keyphrases[:5]}...")

        # Step 2: Test each model
        for model in models:
            logger.info(f"Testing model: {model}")
            print(f"Testing {model}...", end=" ", flush=True)

            result = extract_skills_ollama(keyphrases, model)
            result['job_id'] = job['id']
            result['profession'] = job['profession']
            result['keyphrases_count'] = len(keyphrases)
            result['keyphrases'] = keyphrases

            results[model].append(result)

            if result['success']:
                logger.info(f"{model}: SUCCESS - {result['latency']}s, {len(result['skills'])} skills")
                logger.debug(f"{model} skills: {result['skills']}")
                print(f"OK ({result['latency']}s, {len(result['skills'])} skills)")
                print(f"   Skills: {result['skills'][:8]}...")
            else:
                logger.warning(f"{model}: FAILED - {result.get('error', 'Unknown')}")
                print(f"FAILED ({result['latency']}s) - {result.get('error', 'Unknown')}")

    # Summary
    logger.info("=" * 70)
    logger.info("EXPERIMENT SUMMARY")
    logger.info("=" * 70)

    print("\n" + "=" * 70)
    print("EXPERIMENT SUMMARY")
    print("=" * 70)

    summary_data = {}
    for model in models:
        model_results = results[model]
        successes = sum(1 for r in model_results if r['success'])
        avg_latency = sum(r['latency'] for r in model_results) / len(model_results)
        avg_skills = sum(len(r['skills']) for r in model_results if r['success']) / max(successes, 1)

        # Collect all unique skills
        all_skills = set()
        for r in model_results:
            if r['success']:
                all_skills.update(r['skills'])

        summary_data[model] = {
            'success_rate': f"{successes}/{len(model_results)}",
            'avg_latency': round(avg_latency, 2),
            'avg_skills': round(avg_skills, 1),
            'unique_skills': len(all_skills),
            'all_skills': sorted(list(all_skills))
        }

        logger.info(f"{model}: success={successes}/{len(model_results)}, "
                   f"latency={avg_latency:.2f}s, avg_skills={avg_skills:.1f}, "
                   f"unique={len(all_skills)}")
        logger.debug(f"{model} all unique skills: {sorted(list(all_skills))}")

        print(f"\n{model}:")
        print(f"  Success rate: {successes}/{len(model_results)}")
        print(f"  Avg latency: {avg_latency:.2f}s")
        print(f"  Avg skills extracted: {avg_skills:.1f}")
        print(f"  Unique skills across all jobs: {len(all_skills)}")

    # Detailed results
    logger.info("=" * 70)
    logger.info("DETAILED RESULTS BY JOB")
    logger.info("=" * 70)

    print("\n" + "=" * 70)
    print("DETAILED RESULTS BY JOB")
    print("=" * 70)

    for job in TEST_JOBS:
        logger.info(f"Job: {job['profession']}")
        print(f"\n{job['profession']}:")
        for model in models:
            for r in results[model]:
                if r['job_id'] == job['id'] and r['success']:
                    logger.debug(f"  {model}: {r['skills']}")
                    print(f"  {model}: {r['skills']}")
                    break

    # Log final summary to experiment log
    logger.info("=" * 70)
    logger.info("EXPERIMENT COMPLETE")
    logger.info(f"Summary: {json.dumps(summary_data, indent=2)}")
    logger.info("=" * 70)

    return results, summary_data


if __name__ == "__main__":
    results, summary = run_experiment()
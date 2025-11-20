"""
Test and compare local LLM models for resume parsing capability.

This script tests granite4:micro, llama3.2:3b, and gemma3:4b models
to determine which performs best for structured resume extraction.

Author: Mert Alp Aydin
Date: 2025-11-18
Phase: Phase 2 - Step 2.1
"""

import time
import json
import sys
from pathlib import Path
from typing import Dict, Any, List
import ollama

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from logging_utils import setup_logger, log_experiment

# Initialize logger
logger = setup_logger(__name__)


def test_llm_extraction(model_name: str, prompt: str, test_name: str) -> Dict[str, Any]:
    """
    Test a single LLM model with a given prompt.

    Args:
        model_name: Name of the Ollama model
        prompt: Prompt to send to the model
        test_name: Description of the test

    Returns:
        Dict with results including response, latency, and observations
    """
    logger.info(f"Testing {model_name} - {test_name}")

    try:
        start_time = time.time()

        response = ollama.chat(
            model=model_name,
            messages=[
                {
                    'role': 'user',
                    'content': prompt
                }
            ]
        )

        end_time = time.time()
        latency = end_time - start_time

        result = {
            'model': model_name,
            'test': test_name,
            'response': response['message']['content'],
            'latency_seconds': round(latency, 2),
            'success': True,
            'error': None
        }

        logger.info(f"{model_name} completed in {latency:.2f}s")
        return result

    except Exception as e:
        logger.error(f"Error testing {model_name}: {str(e)}")
        return {
            'model': model_name,
            'test': test_name,
            'response': None,
            'latency_seconds': None,
            'success': False,
            'error': str(e)
        }


def test_json_parsing_capability(model_name: str) -> Dict[str, Any]:
    """
    Test model's ability to extract structured JSON from resume text.
    Uses CHALLENGING test cases with edge cases and validation.

    Args:
        model_name: Name of the Ollama model

    Returns:
        Test results including parsed JSON and quality assessment
    """

    # CHALLENGING RESUME - Edge cases, ambiguity, missing data
    sample_resume_text = """
    Maria Elena Rodriguez-Smith
    Contact: maria.rodriguez@company.co.uk (work) | mrodriguez@personal.com (personal)
    Mobile: +44 20 7123 4567 | WhatsApp: +1 (555) 987-6543

    PROFESSIONAL PROFILE
    Bilingual (English/Spanish) Product Manager & Technical Lead with 7+ years building
    enterprise SaaS platforms. Proven track record scaling products from 0-100K users.
    Previously: Software Engineer → Senior Engineer → Tech Lead → Product Manager

    PROFESSIONAL EXPERIENCE

    Senior Product Manager / Technical PM
    TechVision Inc. (acquired by MegaCorp in 2023) | London, UK & Remote
    March 2021 - Present (3.5 years)
    • Led cross-functional team of 12 (eng, design, data) building B2B analytics platform
    • Increased ARR from $2M to $15M (7.5x growth) in 18 months through product-led growth
    • Shipped 40+ features, maintained 99.9% uptime, reduced churn by 35%
    • Technologies: React, Python/Django, AWS, PostgreSQL, Redis, Docker, k8s

    Tech Lead & Senior Software Engineer
    StartupXYZ → acquired by BigTech Corp | San Francisco, CA (Remote from UK)
    Jan 2018 - Feb 2021 (3 years, 2 months)
    Started as Senior Engineer (Jan 2018-June 2019), promoted to Tech Lead (July 2019-Feb 2021)
    • Built real-time collaboration features serving 50K+ concurrent users
    • Designed microservices architecture, reduced latency by 60%
    • Mentored 5 junior engineers, established code review process
    Stack: Node.js, TypeScript, React, MongoDB, WebSockets, AWS Lambda

    Software Engineer (Contract → Full-time)
    DataCorp Solutions | Berlin, Germany
    June 2016 - December 2017 (1.5 years, contract first 6 months)
    • Developed RESTful APIs and data pipelines processing 10M+ records/day
    • Python, Flask, PostgreSQL, Celery, RabbitMQ

    EDUCATION & CERTIFICATIONS

    MSc Computer Science (with Distinction)
    Technical University of Berlin, Germany | 2014-2016
    Focus: Machine Learning, Distributed Systems
    Thesis: "Scalable Real-time Stream Processing" (published in ACM)

    BSc Information Technology
    Universidad Complutense de Madrid, Spain | 2010-2014

    Certifications:
    - AWS Solutions Architect Professional (2022, expires 2025)
    - Certified Scrum Product Owner (CSPO) - 2021
    - Google Cloud Professional Data Engineer - 2020 (expired, not renewed)

    TECHNICAL SKILLS & EXPERTISE
    Languages: Python, JavaScript/TypeScript, SQL, Go (learning)
    Frontend: React, Vue.js, HTML5/CSS3, Next.js
    Backend: Django, Flask, Node.js, Express, FastAPI
    Data & ML: PostgreSQL, MongoDB, Redis, Pandas, scikit-learn, basic TensorFlow
    DevOps/Cloud: AWS (EC2, S3, Lambda, RDS), GCP, Docker, Kubernetes, CI/CD, Terraform
    Tools: Git, Jira, Figma, Analytics (Mixpanel, Amplitude), A/B testing

    LANGUAGES
    Spanish (native), English (fluent), German (intermediate B2), Portuguese (basic)
    """

    prompt = f"""Extract structured information from the following resume and return ONLY valid JSON with no additional text or explanation.

IMPORTANT INSTRUCTIONS:
- Extract ONLY information that is explicitly stated
- DO NOT make up or infer information
- Use null for missing fields
- Parse dates and durations accurately
- Handle multiple email/phone numbers by taking the primary one
- For overlapping roles or promotions, create separate experience entries

Resume:
{sample_resume_text}

Return a JSON object with this exact structure:
{{
    "full_name": "string",
    "email": "string (primary email only)",
    "phone": "string (primary phone only)",
    "summary": "string",
    "experience": [
        {{
            "company": "string",
            "position": "string",
            "duration": "string (exact dates from resume)",
            "description": "string (key achievements)"
        }}
    ],
    "skills": ["string (individual skills, not categories)"],
    "education": [
        {{
            "institution": "string",
            "degree": "string",
            "field": "string"
        }}
    ],
    "certifications": ["string (name only, no dates)"]
}}

Return ONLY the JSON, no other text."""

    result = test_llm_extraction(model_name, prompt, "Challenging JSON Parsing Test")

    if result['success']:
        # Try to parse the JSON response
        try:
            # Extract JSON from response (handle cases where model adds extra text)
            response_text = result['response'].strip()

            # Try to find JSON in the response
            if '{' in response_text and '}' in response_text:
                json_start = response_text.find('{')
                json_end = response_text.rfind('}') + 1
                json_text = response_text[json_start:json_end]
                parsed_json = json.loads(json_text)

                result['parsed_json'] = parsed_json
                result['json_valid'] = True
                result['fields_extracted'] = len(parsed_json.keys())

                # RIGOROUS VALIDATION
                validation_results = validate_extraction_quality(parsed_json)
                result.update(validation_results)

                # Check for key fields
                required_fields = ['full_name', 'email', 'experience', 'skills']
                missing_fields = [f for f in required_fields if f not in parsed_json or not parsed_json[f]]
                result['missing_required_fields'] = missing_fields
                result['quality_score'] = result['validation_score']

            else:
                result['json_valid'] = False
                result['error'] = "No JSON found in response"

        except json.JSONDecodeError as e:
            result['json_valid'] = False
            result['error'] = f"JSON parse error: {str(e)}"

    return result


def validate_extraction_quality(parsed_json: Dict[str, Any]) -> Dict[str, Any]:
    """
    Rigorously validate the quality of extracted data.
    Checks for correctness, not just presence.

    Args:
        parsed_json: Extracted JSON data

    Returns:
        Dict with validation results and scores
    """
    validation = {
        'validation_errors': [],
        'validation_warnings': [],
        'correctness_checks': {}
    }

    # Expected correct values (ground truth)
    expected = {
        'full_name': 'Maria Elena Rodriguez-Smith',
        'email_domain': ['company.co.uk', 'personal.com'],  # Should pick one
        'experience_count_min': 3,  # At least 3 distinct roles
        'skills_count_min': 10,  # At least 10 distinct technical skills
        'education_count': 2,  # MSc and BSc
        'certifications_count_min': 2,  # At least active certifications
    }

    # Check full name
    name = parsed_json.get('full_name', '')
    if 'Maria' in name and 'Rodriguez' in name:
        validation['correctness_checks']['name_correct'] = True
    else:
        validation['correctness_checks']['name_correct'] = False
        validation['validation_errors'].append(f"Name incorrect: got '{name}'")

    # Check email (should be one of the two)
    email = parsed_json.get('email', '')
    if any(domain in email for domain in expected['email_domain']):
        validation['correctness_checks']['email_correct'] = True
    else:
        validation['correctness_checks']['email_correct'] = False
        validation['validation_errors'].append(f"Email incorrect or missing: got '{email}'")

    # Check experience entries
    experience = parsed_json.get('experience', [])
    if len(experience) >= expected['experience_count_min']:
        validation['correctness_checks']['experience_count_ok'] = True
    else:
        validation['correctness_checks']['experience_count_ok'] = False
        validation['validation_errors'].append(f"Too few experience entries: got {len(experience)}, expected >= {expected['experience_count_min']}")

    # Check if experience has proper structure
    for i, exp in enumerate(experience):
        if not all(key in exp for key in ['company', 'position', 'duration']):
            validation['validation_errors'].append(f"Experience entry {i} missing required fields")

    # Check skills
    skills = parsed_json.get('skills', [])
    if isinstance(skills, list):
        unique_skills = len(set(skills))
        if unique_skills >= expected['skills_count_min']:
            validation['correctness_checks']['skills_count_ok'] = True
        else:
            validation['correctness_checks']['skills_count_ok'] = False
            validation['validation_warnings'].append(f"Only {unique_skills} skills extracted, expected >= {expected['skills_count_min']}")
    else:
        validation['validation_errors'].append("Skills should be a list")
        validation['correctness_checks']['skills_count_ok'] = False

    # Check education
    education = parsed_json.get('education', [])
    if len(education) >= expected['education_count']:
        validation['correctness_checks']['education_count_ok'] = True
    else:
        validation['correctness_checks']['education_count_ok'] = False
        validation['validation_warnings'].append(f"Expected {expected['education_count']} education entries, got {len(education)}")

    # Check for hallucinations (data not in original)
    if email and '@' in email:
        # Make sure it's one of the actual emails
        actual_emails = ['maria.rodriguez@company.co.uk', 'mrodriguez@personal.com']
        if email.lower() not in [e.lower() for e in actual_emails]:
            validation['validation_errors'].append(f"Possible hallucination: email '{email}' not in original resume")
            validation['correctness_checks']['no_hallucination_email'] = False
        else:
            validation['correctness_checks']['no_hallucination_email'] = True
    else:
        validation['correctness_checks']['no_hallucination_email'] = True

    # Calculate validation score
    total_checks = len(validation['correctness_checks'])
    passed_checks = sum(1 for v in validation['correctness_checks'].values() if v)
    validation['validation_score'] = passed_checks / total_checks if total_checks > 0 else 0

    return validation


def run_model_comparison() -> List[Dict[str, Any]]:
    """
    Run comprehensive comparison of all three models.
    Runs 4 tests per model, discards first (warm-up), averages remaining 3.

    Returns:
        List of results for each model (averaged)
    """
    logger.info("Starting LLM model comparison experiment...")
    logger.info("Running 4 tests per model (discarding first warm-up test)")

    models = [
        'granite4:micro',
        'llama3.2:3b',
        'gemma3:4b'
    ]

    final_results = []

    for model in models:
        logger.info(f"\n{'='*60}")
        logger.info(f"Testing model: {model}")
        logger.info(f"{'='*60}")

        model_runs = []

        # Run 4 tests
        for run_num in range(1, 5):
            logger.info(f"  Run {run_num}/4...")
            result = test_json_parsing_capability(model)
            model_runs.append(result)

            if result['success'] and result.get('json_valid'):
                logger.info(f"    Latency: {result['latency_seconds']}s, Quality: {result.get('quality_score', 0):.1%}")
            elif result['success']:
                logger.info(f"    Latency: {result['latency_seconds']}s, JSON Invalid")
            else:
                logger.error(f"    Failed: {result.get('error', 'Unknown')}")

            # Small delay between runs
            if run_num < 4:
                time.sleep(1)

        # Discard first run (warm-up), average the rest
        valid_runs = [r for r in model_runs[1:] if r['success'] and r.get('json_valid')]

        if valid_runs:
            # Calculate averages
            avg_result = {
                'model': model,
                'test': 'Averaged Results (3 runs, excluding warm-up)',
                'success': True,
                'json_valid': True,
                'latency_seconds': round(sum(r['latency_seconds'] for r in valid_runs) / len(valid_runs), 2),
                'quality_score': sum(r.get('quality_score', 0) for r in valid_runs) / len(valid_runs),
                'validation_score': sum(r.get('validation_score', 0) for r in valid_runs) / len(valid_runs),
                'fields_extracted': round(sum(r.get('fields_extracted', 0) for r in valid_runs) / len(valid_runs)),
                'missing_required_fields': model_runs[-1].get('missing_required_fields', []),  # Use last run
                'validation_errors': model_runs[-1].get('validation_errors', []),  # Use last run
                'validation_warnings': model_runs[-1].get('validation_warnings', []),  # Use last run
                'correctness_checks': model_runs[-1].get('correctness_checks', {}),  # Use last run
                'total_runs': 4,
                'valid_runs': len(valid_runs),
                'warm_up_discarded': True,
                'all_runs': model_runs
            }
        else:
            # All runs failed or invalid
            avg_result = {
                'model': model,
                'test': 'All runs failed or invalid JSON',
                'success': False,
                'json_valid': False,
                'error': 'No valid runs after warm-up',
                'total_runs': 4,
                'valid_runs': 0,
                'all_runs': model_runs
            }

        final_results.append(avg_result)

        # Log summary
        if avg_result['success']:
            logger.info(f"\n{model} AVERAGE RESULTS (3 runs):")
            logger.info(f"  Avg Latency: {avg_result['latency_seconds']}s")
            logger.info(f"  Avg Quality Score: {avg_result.get('quality_score', 0):.1%}")
            logger.info(f"  Validation Score: {avg_result.get('validation_score', 0):.1%}")
            if avg_result.get('validation_errors'):
                logger.info(f"  Validation Errors: {len(avg_result['validation_errors'])}")
            if avg_result.get('validation_warnings'):
                logger.info(f"  Validation Warnings: {len(avg_result['validation_warnings'])}")
        else:
            logger.error(f"\n{model} FAILED - No valid runs")

        # Delay between models
        if model != models[-1]:
            time.sleep(2)

    return final_results


def log_experiment_results(results: List[Dict[str, Any]]) -> None:
    """
    Log the experiment results to the experiment log.

    Args:
        results: List of test results for each model
    """
    logger.info("Logging experiment results...")

    # Prepare options tested
    options_tested = []
    for result in results:
        model_name = result['model']

        # Build detailed results string
        if result['success']:
            results_str = (f"Avg Latency: {result.get('latency_seconds', 'N/A')}s (over {result.get('valid_runs', 0)} runs), "
                          f"JSON Valid: {result.get('json_valid', False)}, "
                          f"Quality Score: {result.get('quality_score', 0):.1%}, "
                          f"Validation Errors: {len(result.get('validation_errors', []))}")

            observations_parts = []
            if result.get('missing_required_fields'):
                observations_parts.append(f"Missing fields: {', '.join(result['missing_required_fields'])}")
            if result.get('validation_errors'):
                observations_parts.append(f"Errors: {len(result['validation_errors'])}")
            if result.get('validation_warnings'):
                observations_parts.append(f"Warnings: {len(result['validation_warnings'])}")
            observations_str = '; '.join(observations_parts) if observations_parts else 'No issues detected'
        else:
            results_str = f"Failed - {result.get('error', 'Unknown error')}"
            observations_str = "All test runs failed or produced invalid JSON"

        options_tested.append({
            'name': model_name,
            'configuration': 'Default Ollama settings, 4 runs (1st discarded as warm-up), temperature=1.0',
            'results': results_str,
            'observations': observations_str
        })

    # Determine winner based on quality score and latency
    successful_results = [r for r in results if r['success'] and r.get('json_valid', False)]

    if successful_results:
        # Sort by quality score (descending) then latency (ascending)
        winner_result = max(successful_results,
                           key=lambda x: (x.get('quality_score', 0), -x.get('latency_seconds', float('inf'))))
        winner = winner_result['model']

        rationale = (f"{winner} was selected for its "
                    f"quality score of {winner_result.get('quality_score', 0):.1%} "
                    f"and latency of {winner_result.get('latency_seconds', 0):.2f}s")
    else:
        winner = "None - all models failed"
        rationale = "All models failed the JSON extraction test"

    # Log experiment using our logging utilities
    log_experiment(
        experiment_number=1,
        name="LLM Model Comparison for Resume Parsing",
        phase="Phase 2 - Resume Parsing & Structured Extraction",
        objective="Compare local LLM models (granite4:micro, llama3.2:3b, gemma3:4b) "
                  "for structured resume data extraction capability",
        hypothesis="Larger models (gemma3:4b, llama3.2:3b) will have better extraction "
                   "accuracy but slower response times compared to granite4:micro",
        options_tested=options_tested,
        metrics_used={
            'Latency': 'Time in seconds to complete extraction',
            'JSON Validity': 'Whether output is valid, parseable JSON',
            'Quality Score': 'Percentage of required fields correctly extracted',
            'Missing Fields': 'Count of required fields not extracted'
        },
        winner=winner,
        rationale=rationale,
        key_learning="Model selection involves trade-offs between speed and accuracy. "
                     "Prompt engineering is critical for consistent JSON output.",
        impact=f"Will use {winner} as primary model for resume parsing in subsequent phases. "
               "May implement fallback logic or retry mechanisms for improved reliability.",
        next_steps="Test winner model with actual PDF resume samples. "
                   "Implement Pydantic output parsers for better JSON reliability."
    )

    logger.info("Experiment results logged successfully!")


def main():
    """Main execution function."""
    logger.info("="*70)
    logger.info("LLM MODEL COMPARISON - PHASE 2 STEP 2.1")
    logger.info("="*70)

    # Run the comparison
    results = run_model_comparison()

    # Log results to experiment log
    log_experiment_results(results)

    # Print summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    for result in results:
        if result['success'] and result.get('json_valid'):
            print(f"\n{result['model']}:")
            print(f"  [OK] Latency: {result['latency_seconds']}s (avg of {result.get('valid_runs', 0)} runs)")
            print(f"  [OK] Quality Score: {result.get('quality_score', 0):.1%}")
            print(f"  [OK] Validation Errors: {len(result.get('validation_errors', []))}")
            print(f"  [OK] Valid JSON: Yes")
        elif result['success']:
            print(f"\n{result['model']}:")
            print(f"  [OK] Latency: {result['latency_seconds']}s")
            print(f"  [FAIL] Valid JSON: No")
            print(f"  Error: {result.get('error', 'Unknown')}")
        else:
            print(f"\n{result['model']}:")
            print(f"  [FAIL] {result.get('error', 'Unknown error')}")

    print("\n" + "="*70)
    logger.info("LLM model comparison complete!")


if __name__ == "__main__":
    main()
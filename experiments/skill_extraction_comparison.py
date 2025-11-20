"""
Skill Extraction Method Comparison Experiment

This script compares different skill extraction methods and embedding models:
1. KeyBERT with different transformers
2. Statistical methods (YAKE, RAKE)

Metrics evaluated:
- Extraction quality (precision/recall against ground truth)
- Diversity of extracted skills
- Execution speed
- Consistency across runs

Author: AI Resume Analyzer Project
Date: 2025-11-20
"""

import logging
import time
import json
from typing import List, Dict, Tuple, Set
from dataclasses import dataclass, asdict
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from skill_extractor import SkillExtractor, SkillNormalizer
from keybert import KeyBERT
import spacy

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class ExtractionResult:
    """Results from a skill extraction method"""
    method_name: str
    model_name: str
    skills: List[str]
    skill_scores: Dict[str, float]
    execution_time: float
    num_skills: int

    # Quality metrics (if ground truth available)
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0

    # Diversity metrics
    unique_skills: int = 0
    avg_skill_length: float = 0.0


class SkillExtractionExperiment:
    """
    Experimental framework for comparing skill extraction methods.
    """

    def __init__(self, top_n: int = 20, diversity: float = 0.5):
        """
        Initialize the experiment.

        Args:
            top_n: Number of skills to extract per method
            diversity: Diversity parameter for KeyBERT MMR
        """
        self.top_n = top_n
        self.diversity = diversity
        self.normalizer = SkillNormalizer()

        # Load spaCy for statistical methods
        self.nlp = spacy.load("en_core_web_sm")

        logger.info(f"Experiment initialized: top_n={top_n}, diversity={diversity}")

    def extract_with_keybert(
        self,
        text: str,
        model_name: str = "all-MiniLM-L6-v2"
    ) -> ExtractionResult:
        """
        Extract skills using KeyBERT with specified embedding model.

        Args:
            text: Input text
            model_name: SentenceTransformer model name

        Returns:
            ExtractionResult with timing and skills
        """
        logger.info(f"Testing KeyBERT with {model_name}")

        start_time = time.time()

        try:
            # Initialize KeyBERT with specified model
            kw_model = KeyBERT(model=model_name)

            # Extract keywords
            keywords = kw_model.extract_keywords(
                text,
                keyphrase_ngram_range=(1, 3),
                stop_words='english',
                top_n=self.top_n,
                use_mmr=True,
                diversity=self.diversity
            )

            # Normalize skills
            normalized_skills = {}
            for skill, score in keywords:
                norm_skill = self.normalizer.normalize(skill)
                if norm_skill not in normalized_skills or score > normalized_skills[norm_skill]:
                    normalized_skills[norm_skill] = score

            skills = list(normalized_skills.keys())
            execution_time = time.time() - start_time

            result = ExtractionResult(
                method_name="KeyBERT",
                model_name=model_name,
                skills=skills[:self.top_n],
                skill_scores=normalized_skills,
                execution_time=execution_time,
                num_skills=len(skills[:self.top_n]),
                unique_skills=len(set(skills[:self.top_n])),
                avg_skill_length=sum(len(s) for s in skills[:self.top_n]) / len(skills[:self.top_n]) if skills else 0
            )

            logger.info(f"KeyBERT ({model_name}): {len(skills)} skills in {execution_time:.2f}s")
            return result

        except Exception as e:
            logger.error(f"Error with KeyBERT ({model_name}): {e}")
            return ExtractionResult(
                method_name="KeyBERT",
                model_name=model_name,
                skills=[],
                skill_scores={},
                execution_time=time.time() - start_time,
                num_skills=0
            )

    def extract_with_yake(self, text: str) -> ExtractionResult:
        """
        Extract skills using YAKE (statistical method).

        Args:
            text: Input text

        Returns:
            ExtractionResult with timing and skills
        """
        logger.info("Testing YAKE")

        start_time = time.time()

        try:
            import yake

            # Initialize YAKE
            kw_extractor = yake.KeywordExtractor(
                lan="en",
                n=3,  # Max n-gram size
                dedupLim=0.7,  # Deduplication threshold
                top=self.top_n,
                features=None
            )

            # Extract keywords (lower score = better in YAKE)
            keywords = kw_extractor.extract_keywords(text)

            # Normalize and invert scores (YAKE uses lower=better, we want higher=better)
            normalized_skills = {}
            max_score = max([score for _, score in keywords]) if keywords else 1.0

            for skill, score in keywords:
                norm_skill = self.normalizer.normalize(skill)
                # Invert score: lower YAKE score = higher importance
                inverted_score = 1.0 - (score / max_score) if max_score > 0 else 0.0

                if norm_skill not in normalized_skills or inverted_score > normalized_skills[norm_skill]:
                    normalized_skills[norm_skill] = inverted_score

            skills = list(normalized_skills.keys())
            execution_time = time.time() - start_time

            result = ExtractionResult(
                method_name="YAKE",
                model_name="statistical",
                skills=skills[:self.top_n],
                skill_scores=normalized_skills,
                execution_time=execution_time,
                num_skills=len(skills[:self.top_n]),
                unique_skills=len(set(skills[:self.top_n])),
                avg_skill_length=sum(len(s) for s in skills[:self.top_n]) / len(skills[:self.top_n]) if skills else 0
            )

            logger.info(f"YAKE: {len(skills)} skills in {execution_time:.2f}s")
            return result

        except ImportError:
            logger.warning("YAKE not installed. Install with: pip install yake")
            return ExtractionResult(
                method_name="YAKE",
                model_name="statistical",
                skills=[],
                skill_scores={},
                execution_time=0,
                num_skills=0
            )
        except Exception as e:
            logger.error(f"Error with YAKE: {e}")
            return ExtractionResult(
                method_name="YAKE",
                model_name="statistical",
                skills=[],
                skill_scores={},
                execution_time=time.time() - start_time,
                num_skills=0
            )

    def extract_with_rake(self, text: str) -> ExtractionResult:
        """
        Extract skills using RAKE (statistical method).

        Args:
            text: Input text

        Returns:
            ExtractionResult with timing and skills
        """
        logger.info("Testing RAKE")

        start_time = time.time()

        try:
            from rake_nltk import Rake

            # Initialize RAKE
            rake = Rake(max_length=3)  # Max n-gram size

            # Extract keywords
            rake.extract_keywords_from_text(text)
            keywords_with_scores = rake.get_ranked_phrases_with_scores()

            # Normalize skills
            normalized_skills = {}
            for score, skill in keywords_with_scores[:self.top_n * 2]:  # Get extra for normalization
                norm_skill = self.normalizer.normalize(skill)
                if norm_skill not in normalized_skills or score > normalized_skills[norm_skill]:
                    normalized_skills[norm_skill] = score

            # Normalize scores to 0-1 range
            if normalized_skills:
                max_score = max(normalized_skills.values())
                if max_score > 0:
                    normalized_skills = {k: v/max_score for k, v in normalized_skills.items()}

            skills = list(normalized_skills.keys())
            execution_time = time.time() - start_time

            result = ExtractionResult(
                method_name="RAKE",
                model_name="statistical",
                skills=skills[:self.top_n],
                skill_scores=normalized_skills,
                execution_time=execution_time,
                num_skills=len(skills[:self.top_n]),
                unique_skills=len(set(skills[:self.top_n])),
                avg_skill_length=sum(len(s) for s in skills[:self.top_n]) / len(skills[:self.top_n]) if skills else 0
            )

            logger.info(f"RAKE: {len(skills)} skills in {execution_time:.2f}s")
            return result

        except ImportError:
            logger.warning("RAKE not installed. Install with: pip install rake-nltk")
            return ExtractionResult(
                method_name="RAKE",
                model_name="statistical",
                skills=[],
                skill_scores={},
                execution_time=0,
                num_skills=0
            )
        except Exception as e:
            logger.error(f"Error with RAKE: {e}")
            return ExtractionResult(
                method_name="RAKE",
                model_name="statistical",
                skills=[],
                skill_scores={},
                execution_time=time.time() - start_time,
                num_skills=0
            )

    def calculate_quality_metrics(
        self,
        extracted_skills: List[str],
        ground_truth: Set[str]
    ) -> Tuple[float, float, float]:
        """
        Calculate precision, recall, and F1 score.

        Args:
            extracted_skills: Skills extracted by method
            ground_truth: True skills (manually annotated)

        Returns:
            Tuple of (precision, recall, f1_score)
        """
        extracted_set = set(self.normalizer.normalize(s) for s in extracted_skills)
        ground_truth_set = set(self.normalizer.normalize(s) for s in ground_truth)

        if not extracted_set:
            return 0.0, 0.0, 0.0

        true_positives = len(extracted_set & ground_truth_set)

        precision = true_positives / len(extracted_set) if extracted_set else 0.0
        recall = true_positives / len(ground_truth_set) if ground_truth_set else 0.0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        return precision, recall, f1_score

    def run_comparison(
        self,
        text: str,
        ground_truth: Set[str] = None,
        embedding_models: List[str] = None
    ) -> List[ExtractionResult]:
        """
        Run full comparison of all methods.

        Args:
            text: Input text for skill extraction
            ground_truth: Optional set of true skills for quality metrics
            embedding_models: List of embedding models to test with KeyBERT

        Returns:
            List of ExtractionResult objects
        """
        if embedding_models is None:
            embedding_models = [
                "all-MiniLM-L6-v2",  # Current default (384 dims, fast)
                "all-mpnet-base-v2",  # Higher quality (768 dims, slower)
                "all-distilroberta-v1",  # Good balance (768 dims)
            ]

        results = []

        # Test KeyBERT with different embedding models
        for model_name in embedding_models:
            result = self.extract_with_keybert(text, model_name)

            if ground_truth:
                precision, recall, f1 = self.calculate_quality_metrics(
                    result.skills, ground_truth
                )
                result.precision = precision
                result.recall = recall
                result.f1_score = f1

            results.append(result)

        # Test YAKE
        yake_result = self.extract_with_yake(text)
        if ground_truth and yake_result.skills:
            precision, recall, f1 = self.calculate_quality_metrics(
                yake_result.skills, ground_truth
            )
            yake_result.precision = precision
            yake_result.recall = recall
            yake_result.f1_score = f1
        results.append(yake_result)

        # Test RAKE
        rake_result = self.extract_with_rake(text)
        if ground_truth and rake_result.skills:
            precision, recall, f1 = self.calculate_quality_metrics(
                rake_result.skills, ground_truth
            )
            rake_result.precision = precision
            rake_result.recall = recall
            rake_result.f1_score = f1
        results.append(rake_result)

        return results

    def save_results(self, results: List[ExtractionResult], output_file: str):
        """Save experiment results to JSON file."""
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(
                [asdict(r) for r in results],
                f,
                indent=2
            )

        logger.info(f"Results saved to {output_path}")


def print_comparison_table(results: List[ExtractionResult]):
    """Print a formatted comparison table."""
    print("\n" + "="*120)
    print("SKILL EXTRACTION METHOD COMPARISON")
    print("="*120)

    # Header
    print(f"{'Method':<15} {'Model':<25} {'Skills':<8} {'Time (s)':<10} {'Unique':<8} {'Precision':<10} {'Recall':<10} {'F1':<10}")
    print("-"*120)

    # Results
    for result in results:
        if result.num_skills > 0:
            print(f"{result.method_name:<15} {result.model_name:<25} {result.num_skills:<8} "
                  f"{result.execution_time:<10.2f} {result.unique_skills:<8} "
                  f"{result.precision:<10.3f} {result.recall:<10.3f} {result.f1_score:<10.3f}")

    print("="*120)

    # Print top skills from each method
    print("\nTOP 10 SKILLS BY METHOD:")
    print("-"*120)

    for result in results:
        if result.num_skills > 0:
            print(f"\n{result.method_name} ({result.model_name}):")
            for i, skill in enumerate(result.skills[:10], 1):
                score = result.skill_scores.get(skill, 0.0)
                print(f"  {i:2}. {skill:<30} ({score:.3f})")


# Example usage
if __name__ == "__main__":
    # Sample resume text
    resume_text = """
    Senior Software Engineer with 7+ years of experience in Python, JavaScript, and Java.
    Expert in machine learning, deep learning, and natural language processing.
    Proficient in AWS, Docker, Kubernetes, and CI/CD pipelines.
    Strong background in React, Node.js, and MongoDB.
    Experience with TensorFlow, PyTorch, and scikit-learn for ML projects.
    Skilled in PostgreSQL, Redis, and Elasticsearch for data storage.
    Implemented microservices architecture using Spring Boot and FastAPI.
    Led teams in agile development with strong communication and problem-solving skills.
    """

    # Ground truth skills (manually annotated)
    ground_truth = {
        "python", "javascript", "java", "machine learning", "deep learning",
        "nlp", "natural language processing", "aws", "docker", "kubernetes",
        "ci/cd", "react", "node.js", "mongodb", "tensorflow", "pytorch",
        "scikit-learn", "postgresql", "redis", "elasticsearch", "microservices",
        "spring", "fastapi", "agile", "communication", "problem solving"
    }

    # Run experiment
    print("Starting Skill Extraction Comparison Experiment...")
    print(f"Text length: {len(resume_text)} characters")
    print(f"Ground truth skills: {len(ground_truth)}")

    experiment = SkillExtractionExperiment(top_n=20, diversity=0.5)

    # Test different embedding models
    embedding_models = [
        "all-MiniLM-L6-v2",      # Fast, lightweight (384 dims)
        "all-mpnet-base-v2",     # High quality (768 dims)
        # "all-distilroberta-v1",  # Commented out to save time, uncomment to test
    ]

    results = experiment.run_comparison(
        text=resume_text,
        ground_truth=ground_truth,
        embedding_models=embedding_models
    )

    # Display results
    print_comparison_table(results)

    # Save results
    experiment.save_results(
        results,
        "experiments/results/skill_extraction_comparison.json"
    )

    # Print winner
    print("\n" + "="*120)
    print("RECOMMENDATIONS:")
    print("="*120)

    valid_results = [r for r in results if r.num_skills > 0]

    if valid_results:
        # Best F1 score
        best_f1 = max(valid_results, key=lambda x: x.f1_score)
        print(f"\nBest F1 Score: {best_f1.method_name} ({best_f1.model_name})")
        print(f"  F1: {best_f1.f1_score:.3f}, Precision: {best_f1.precision:.3f}, Recall: {best_f1.recall:.3f}")
        print(f"  Time: {best_f1.execution_time:.2f}s, Skills: {best_f1.num_skills}")

        # Fastest method
        fastest = min(valid_results, key=lambda x: x.execution_time)
        print(f"\nFastest Method: {fastest.method_name} ({fastest.model_name})")
        print(f"  Time: {fastest.execution_time:.2f}s")
        print(f"  F1: {fastest.f1_score:.3f}, Skills: {fastest.num_skills}")

        # Best balance (F1 / time)
        balanced = max(valid_results, key=lambda x: x.f1_score / max(x.execution_time, 0.1))
        print(f"\nBest Balance (F1/time): {balanced.method_name} ({balanced.model_name})")
        print(f"  F1/time ratio: {balanced.f1_score / max(balanced.execution_time, 0.1):.3f}")
        print(f"  F1: {balanced.f1_score:.3f}, Time: {balanced.execution_time:.2f}s")
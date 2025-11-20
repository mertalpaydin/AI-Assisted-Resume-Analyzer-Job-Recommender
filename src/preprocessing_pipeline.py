"""
Preprocessing Pipeline Module

This module provides a unified preprocessing pipeline that combines text cleaning,
tokenization, lemmatization, and skill extraction for resume and job posting analysis.

Author: AI Resume Analyzer Project
Date: 2025-01-20
"""

import logging
import json
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path

from text_preprocessor import TextPreprocessor, PreprocessingMode
from skill_extractor import SkillExtractor

# Set up logging
logger = logging.getLogger(__name__)


@dataclass
class PreprocessedDocument:
    """
    Data class representing a preprocessed document with metadata.
    """
    # Original data
    original_text: str
    doc_id: str
    doc_type: str  # 'resume' or 'job_posting'

    # Preprocessed data
    cleaned_text: str
    tokens: List[str]
    lemmas: List[str]

    # Extracted skills
    skills: List[str]
    skill_scores: Dict[str, float]

    # Metadata
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    preprocessing_params: Dict = field(default_factory=dict)

    # Statistics
    stats: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return asdict(self)

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, data: Dict) -> 'PreprocessedDocument':
        """Create from dictionary."""
        return cls(**data)


class PreprocessingPipeline:
    """
    Unified preprocessing pipeline for resume and job posting analysis.

    This class combines text preprocessing and skill extraction into a single
    cohesive pipeline with configurable parameters and metadata tracking.
    """

    def __init__(
        self,
        # Text preprocessing parameters
        remove_stopwords: bool = True,
        lowercase: bool = True,
        remove_punctuation: bool = True,
        preprocessing_mode: PreprocessingMode = PreprocessingMode.FULL,

        # Skill extraction parameters
        skill_extraction_method: str = "rake",  # DEFAULT: RAKE (F1: 0.356, fastest, best precision)
        top_n_skills: int = 20,
        skill_diversity: float = 0.5,
        use_skill_normalization: bool = True,

        # General parameters
        enable_caching: bool = True,
        cache_dir: Optional[str] = None
    ):
        """
        Initialize the preprocessing pipeline.

        Args:
            remove_stopwords: Remove stopwords during text preprocessing
            lowercase: Convert text to lowercase
            remove_punctuation: Remove punctuation marks
            preprocessing_mode: Mode for text preprocessing (MINIMAL, STANDARD, FULL)
            skill_extraction_method: Method for skill extraction ('keybert', 'spacy', 'hybrid')
            top_n_skills: Number of top skills to extract
            skill_diversity: Diversity parameter for skill extraction
            use_skill_normalization: Whether to normalize extracted skills
            enable_caching: Enable caching of preprocessed documents
            cache_dir: Directory for caching (default: ./data/cache)
        """
        logger.info("Initializing PreprocessingPipeline")

        # Initialize text preprocessor
        self.text_preprocessor = TextPreprocessor(
            remove_stopwords=remove_stopwords,
            lowercase=lowercase,
            remove_punctuation=remove_punctuation
        )

        # Initialize skill extractor
        self.skill_extractor = SkillExtractor(
            top_n=top_n_skills,
            diversity=skill_diversity,
            use_normalization=use_skill_normalization
        )

        # Store parameters
        self.preprocessing_mode = preprocessing_mode
        self.skill_extraction_method = skill_extraction_method
        self.top_n_skills = top_n_skills

        # Caching
        self.enable_caching = enable_caching
        if cache_dir:
            self.cache_dir = Path(cache_dir)
        else:
            self.cache_dir = Path("./data/cache")

        if enable_caching:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Caching enabled at: {self.cache_dir}")

        # Store processing parameters for metadata
        self.params = {
            'remove_stopwords': remove_stopwords,
            'lowercase': lowercase,
            'remove_punctuation': remove_punctuation,
            'preprocessing_mode': preprocessing_mode.value,
            'skill_extraction_method': skill_extraction_method,
            'top_n_skills': top_n_skills,
            'skill_diversity': skill_diversity,
            'use_skill_normalization': use_skill_normalization
        }

        logger.info("PreprocessingPipeline initialized successfully")

    def process_document(
        self,
        text: str,
        doc_id: str,
        doc_type: str = "resume",
        extract_skills: bool = True
    ) -> PreprocessedDocument:
        """
        Process a single document through the complete pipeline.

        Args:
            text: Input text to process
            doc_id: Unique identifier for the document
            doc_type: Type of document ('resume' or 'job_posting')
            extract_skills: Whether to extract skills from the text

        Returns:
            PreprocessedDocument object with all processed data and metadata
        """
        logger.info(f"Processing document: {doc_id} (type: {doc_type})")

        # Check cache first
        if self.enable_caching:
            cached_doc = self._load_from_cache(doc_id)
            if cached_doc:
                logger.info(f"Loaded document {doc_id} from cache")
                return cached_doc

        # Step 1: Clean text
        cleaned_text = self.text_preprocessor.clean_text(text)

        # Step 2: Tokenize
        tokens = self.text_preprocessor.preprocess(
            text,
            mode=PreprocessingMode.STANDARD,
            return_as_string=False
        )

        # Step 3: Lemmatize
        lemmas = self.text_preprocessor.preprocess(
            text,
            mode=PreprocessingMode.FULL,
            return_as_string=False
        )

        # Step 4: Extract skills (if enabled)
        skills = []
        skill_scores = {}

        if extract_skills:
            skill_results = self.skill_extractor.extract_skills(
                text,
                method=self.skill_extraction_method,
                top_n=self.top_n_skills,
                return_scores=True
            )

            skills = [skill for skill, _ in skill_results]
            skill_scores = {skill: score for skill, score in skill_results}

            logger.debug(f"Extracted {len(skills)} skills from document {doc_id}")

        # Step 5: Gather statistics
        stats = self.text_preprocessor.get_statistics(text)
        stats['num_skills_extracted'] = len(skills)

        # Create PreprocessedDocument
        doc = PreprocessedDocument(
            original_text=text,
            doc_id=doc_id,
            doc_type=doc_type,
            cleaned_text=cleaned_text,
            tokens=tokens,
            lemmas=lemmas,
            skills=skills,
            skill_scores=skill_scores,
            preprocessing_params=self.params.copy(),
            stats=stats
        )

        # Cache the result
        if self.enable_caching:
            self._save_to_cache(doc)

        logger.info(f"Document {doc_id} processed successfully")
        return doc

    def process_batch(
        self,
        documents: List[Tuple[str, str, str]],
        extract_skills: bool = True
    ) -> List[PreprocessedDocument]:
        """
        Process multiple documents efficiently.

        Args:
            documents: List of (text, doc_id, doc_type) tuples
            extract_skills: Whether to extract skills from texts

        Returns:
            List of PreprocessedDocument objects
        """
        logger.info(f"Processing batch of {len(documents)} documents")

        results = []
        for text, doc_id, doc_type in documents:
            try:
                doc = self.process_document(
                    text=text,
                    doc_id=doc_id,
                    doc_type=doc_type,
                    extract_skills=extract_skills
                )
                results.append(doc)
            except Exception as e:
                logger.error(f"Error processing document {doc_id}: {e}")
                continue

        logger.info(f"Batch processing completed: {len(results)}/{len(documents)} successful")
        return results

    def process_resume(self, text: str, resume_id: str) -> PreprocessedDocument:
        """
        Convenience method to process a resume.

        Args:
            text: Resume text
            resume_id: Unique resume identifier

        Returns:
            PreprocessedDocument
        """
        return self.process_document(text, resume_id, doc_type="resume")

    def process_job_posting(self, text: str, job_id: str) -> PreprocessedDocument:
        """
        Convenience method to process a job posting.

        Args:
            text: Job posting text
            job_id: Unique job posting identifier

        Returns:
            PreprocessedDocument
        """
        return self.process_document(text, job_id, doc_type="job_posting")

    def compare_documents(
        self,
        doc1: PreprocessedDocument,
        doc2: PreprocessedDocument
    ) -> Dict:
        """
        Compare two preprocessed documents (e.g., resume vs job posting).

        Args:
            doc1: First document (typically a resume)
            doc2: Second document (typically a job posting)

        Returns:
            Dictionary with comparison metrics
        """
        logger.info(f"Comparing documents: {doc1.doc_id} vs {doc2.doc_id}")

        # Skill matching
        skill_match = self.skill_extractor.match_skills(
            doc1.skills,
            doc2.skills
        )

        # Token overlap
        tokens1 = set(doc1.tokens)
        tokens2 = set(doc2.tokens)
        token_overlap = len(tokens1 & tokens2)
        token_overlap_pct = (token_overlap / len(tokens2) * 100) if tokens2 else 0

        # Lemma overlap
        lemmas1 = set(doc1.lemmas)
        lemmas2 = set(doc2.lemmas)
        lemma_overlap = len(lemmas1 & lemmas2)
        lemma_overlap_pct = (lemma_overlap / len(lemmas2) * 100) if lemmas2 else 0

        comparison = {
            'doc1_id': doc1.doc_id,
            'doc2_id': doc2.doc_id,
            'skill_match_percentage': skill_match['match_percentage'],
            'matched_skills': skill_match['matched'],
            'missing_skills': skill_match['missing'],
            'extra_skills': skill_match['extra'],
            'token_overlap': token_overlap,
            'token_overlap_percentage': token_overlap_pct,
            'lemma_overlap': lemma_overlap,
            'lemma_overlap_percentage': lemma_overlap_pct,
        }

        logger.info(f"Comparison complete: {skill_match['match_percentage']:.1f}% skill match")
        return comparison

    def _save_to_cache(self, doc: PreprocessedDocument):
        """Save preprocessed document to cache."""
        try:
            cache_file = self.cache_dir / f"{doc.doc_id}.json"
            with open(cache_file, 'w', encoding='utf-8') as f:
                f.write(doc.to_json())
            logger.debug(f"Saved document {doc.doc_id} to cache")
        except Exception as e:
            logger.warning(f"Failed to cache document {doc.doc_id}: {e}")

    def _load_from_cache(self, doc_id: str) -> Optional[PreprocessedDocument]:
        """Load preprocessed document from cache."""
        try:
            cache_file = self.cache_dir / f"{doc_id}.json"
            if cache_file.exists():
                with open(cache_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                return PreprocessedDocument.from_dict(data)
        except Exception as e:
            logger.warning(f"Failed to load document {doc_id} from cache: {e}")
        return None

    def clear_cache(self):
        """Clear all cached documents."""
        if self.cache_dir.exists():
            for cache_file in self.cache_dir.glob("*.json"):
                cache_file.unlink()
            logger.info("Cache cleared")

    def get_pipeline_info(self) -> Dict:
        """Get information about the pipeline configuration."""
        return {
            'parameters': self.params,
            'cache_enabled': self.enable_caching,
            'cache_directory': str(self.cache_dir) if self.enable_caching else None,
            'text_preprocessor_stats': {
                'stopwords_count': len(self.text_preprocessor.stopwords)
            },
            'skill_extractor_stats': {
                'canonical_skills': len(self.skill_extractor.normalizer.get_all_canonical_skills())
            }
        }


# Example usage and testing
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Sample resume
    resume_text = """
    John Doe
    Senior Software Engineer
    Email: john.doe@email.com

    SUMMARY:
    Experienced Software Engineer with 7+ years in full-stack development.
    Expert in Python, JavaScript, and cloud technologies.

    SKILLS:
    - Programming: Python, JavaScript, Java, TypeScript
    - Web: React, Node.js, Django, Flask
    - Cloud: AWS, Docker, Kubernetes
    - ML/AI: TensorFlow, PyTorch, scikit-learn
    - Databases: PostgreSQL, MongoDB, Redis

    EXPERIENCE:
    Senior Software Engineer at Tech Corp (2020-Present)
    - Led development of microservices architecture using Python and Docker
    - Implemented ML models for recommendation system using TensorFlow
    - Managed CI/CD pipelines and automated testing
    """

    # Sample job posting
    job_text = """
    Senior ML Engineer Position

    We are seeking a talented Senior Machine Learning Engineer to join our team.

    Required Skills:
    - Strong programming skills in Python
    - Experience with TensorFlow or PyTorch
    - Cloud deployment experience (AWS preferred)
    - Docker and Kubernetes knowledge
    - Strong problem-solving and communication skills

    Preferred:
    - Experience with React and Node.js
    - Background in NLP or Computer Vision
    - CI/CD pipeline experience
    """

    # Create pipeline
    print("Initializing preprocessing pipeline...")
    pipeline = PreprocessingPipeline(
        preprocessing_mode=PreprocessingMode.FULL,
        skill_extraction_method="hybrid",
        top_n_skills=15,
        enable_caching=False  # Disable caching for demo
    )

    # Process resume
    print("\n" + "="*80)
    print("PROCESSING RESUME:")
    resume_doc = pipeline.process_resume(resume_text, "resume_001")

    print(f"\nDocument ID: {resume_doc.doc_id}")
    print(f"Document Type: {resume_doc.doc_type}")
    print(f"Original Length: {len(resume_doc.original_text)} chars")
    print(f"Cleaned Length: {len(resume_doc.cleaned_text)} chars")
    print(f"Tokens: {len(resume_doc.tokens)}")
    print(f"Lemmas: {len(resume_doc.lemmas)}")
    print(f"\nTop Skills Extracted:")
    for i, skill in enumerate(resume_doc.skills[:10], 1):
        score = resume_doc.skill_scores.get(skill, 0)
        print(f"  {i}. {skill} ({score:.3f})")

    # Process job posting
    print("\n" + "="*80)
    print("PROCESSING JOB POSTING:")
    job_doc = pipeline.process_job_posting(job_text, "job_001")

    print(f"\nDocument ID: {job_doc.doc_id}")
    print(f"Document Type: {job_doc.doc_type}")
    print(f"Tokens: {len(job_doc.tokens)}")
    print(f"Lemmas: {len(job_doc.lemmas)}")
    print(f"\nTop Skills Extracted:")
    for i, skill in enumerate(job_doc.skills[:10], 1):
        score = job_doc.skill_scores.get(skill, 0)
        print(f"  {i}. {skill} ({score:.3f})")

    # Compare documents
    print("\n" + "="*80)
    print("COMPARING RESUME vs JOB POSTING:")
    comparison = pipeline.compare_documents(resume_doc, job_doc)

    print(f"\nSkill Match: {comparison['skill_match_percentage']:.1f}%")
    print(f"Token Overlap: {comparison['token_overlap_percentage']:.1f}%")
    print(f"Lemma Overlap: {comparison['lemma_overlap_percentage']:.1f}%")

    print(f"\nMatched Skills ({len(comparison['matched_skills'])}):")
    for skill in comparison['matched_skills'][:10]:
        print(f"  [MATCH] {skill}")

    print(f"\nMissing Skills ({len(comparison['missing_skills'])}):")
    for skill in comparison['missing_skills'][:10]:
        print(f"  [NEED] {skill}")

    # Pipeline info
    print("\n" + "="*80)
    print("PIPELINE INFORMATION:")
    info = pipeline.get_pipeline_info()
    print(json.dumps(info, indent=2))
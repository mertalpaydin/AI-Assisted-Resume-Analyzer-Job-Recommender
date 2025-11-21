"""
Skill Extraction Module

This module provides skill extraction capabilities using KeyBERT for resume and job posting analysis.
Features include keyword extraction, skill normalization, and custom skill databases.

Author: Mert Alp Aydin
Date: 2025-11-20
"""

import logging
import re
from typing import List, Dict, Set, Tuple, Optional
from collections import defaultdict
import spacy
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer

# Set up logging
logger = logging.getLogger(__name__)


class SkillNormalizer:
    """
    Handles skill name normalization and alias mapping.
    Maps various skill representations to their canonical forms.
    """

    # Comprehensive skill normalization database
    SKILL_ALIASES = {
        # Programming Languages
        'python': ['python', 'py', 'python3', 'python2'],
        'javascript': ['javascript', 'js', 'ecmascript', 'es6', 'es5'],
        'java': ['java', 'java se', 'java ee'],
        'c++': ['c++', 'cpp', 'c plus plus'],
        'c#': ['c#', 'csharp', 'c sharp'],
        'typescript': ['typescript', 'ts'],
        'ruby': ['ruby', 'ruby on rails', 'ror'],
        'php': ['php', 'php7', 'php8'],
        'go': ['go', 'golang'],
        'rust': ['rust', 'rust-lang'],
        'swift': ['swift', 'swift ui'],
        'kotlin': ['kotlin', 'kt'],
        'r': ['r', 'r-lang', 'r programming'],
        'scala': ['scala'],
        'perl': ['perl'],

        # Web Technologies
        'html': ['html', 'html5', 'html 5'],
        'css': ['css', 'css3', 'css 3'],
        'react': ['react', 'reactjs', 'react.js', 'react js'],
        'angular': ['angular', 'angularjs', 'angular.js'],
        'vue': ['vue', 'vuejs', 'vue.js'],
        'node.js': ['node', 'nodejs', 'node.js', 'node js'],
        'express': ['express', 'expressjs', 'express.js'],
        'django': ['django'],
        'flask': ['flask'],
        'fastapi': ['fastapi', 'fast api'],
        'spring': ['spring', 'spring boot', 'spring framework'],
        'asp.net': ['asp.net', 'asp net', 'aspnet'],

        # Databases
        'sql': ['sql', 'structured query language'],
        'mysql': ['mysql', 'my sql'],
        'postgresql': ['postgresql', 'postgres', 'psql'],
        'mongodb': ['mongodb', 'mongo'],
        'redis': ['redis'],
        'elasticsearch': ['elasticsearch', 'elastic search', 'es'],
        'oracle': ['oracle', 'oracle db'],
        'sqlite': ['sqlite', 'sqlite3'],
        'cassandra': ['cassandra'],
        'dynamodb': ['dynamodb', 'dynamo db'],

        # Cloud & DevOps
        'aws': ['aws', 'amazon web services'],
        'azure': ['azure', 'microsoft azure'],
        'gcp': ['gcp', 'google cloud', 'google cloud platform'],
        'docker': ['docker', 'containerization'],
        'kubernetes': ['kubernetes', 'k8s', 'k8'],
        'jenkins': ['jenkins'],
        'terraform': ['terraform'],
        'ansible': ['ansible'],
        'ci/cd': ['ci/cd', 'cicd', 'continuous integration', 'continuous deployment'],
        'git': ['git', 'version control'],
        'github': ['github'],
        'gitlab': ['gitlab'],

        # Machine Learning & AI
        'machine learning': ['machine learning', 'ml'],
        'deep learning': ['deep learning', 'dl'],
        'artificial intelligence': ['artificial intelligence', 'ai'],
        'neural networks': ['neural networks', 'neural network', 'nn'],
        'tensorflow': ['tensorflow', 'tf'],
        'pytorch': ['pytorch', 'torch'],
        'scikit-learn': ['scikit-learn', 'sklearn', 'scikit learn'],
        'keras': ['keras'],
        'nlp': ['nlp', 'natural language processing', 'natural language'],
        'computer vision': ['computer vision', 'cv', 'image processing'],
        'data science': ['data science', 'data scientist'],

        # Data & Analytics
        'pandas': ['pandas', 'pd'],
        'numpy': ['numpy', 'np'],
        'matplotlib': ['matplotlib', 'mpl'],
        'seaborn': ['seaborn'],
        'tableau': ['tableau'],
        'power bi': ['power bi', 'powerbi'],
        'apache spark': ['apache spark', 'spark', 'pyspark'],
        'hadoop': ['hadoop'],
        'kafka': ['kafka', 'apache kafka'],

        # Testing & Quality
        'unit testing': ['unit testing', 'unit tests', 'testing'],
        'pytest': ['pytest', 'py.test'],
        'jest': ['jest'],
        'selenium': ['selenium'],
        'junit': ['junit'],

        # Soft Skills
        'leadership': ['leadership', 'team leadership', 'lead'],
        'communication': ['communication', 'verbal communication', 'written communication'],
        'problem solving': ['problem solving', 'problem-solving'],
        'teamwork': ['teamwork', 'team work', 'collaboration'],
        'agile': ['agile', 'scrum', 'agile methodology'],
        'project management': ['project management', 'pm'],
    }

    # Reverse mapping: alias -> canonical form
    ALIAS_TO_CANONICAL: Dict[str, str] = {}

    def __init__(self, custom_aliases: Optional[Dict[str, List[str]]] = None):
        """
        Initialize the SkillNormalizer.

        Args:
            custom_aliases: Additional skill aliases to add to the database
        """
        # Build reverse mapping
        self._build_reverse_mapping()

        # Add custom aliases if provided
        if custom_aliases:
            for canonical, aliases in custom_aliases.items():
                self.add_skill_alias(canonical, aliases)

        logger.info(f"SkillNormalizer initialized with {len(self.SKILL_ALIASES)} canonical skills")

    def _build_reverse_mapping(self):
        """Build the reverse mapping from aliases to canonical forms."""
        for canonical, aliases in self.SKILL_ALIASES.items():
            for alias in aliases:
                self.ALIAS_TO_CANONICAL[alias.lower()] = canonical

    def normalize(self, skill: str) -> str:
        """
        Normalize a skill name to its canonical form.

        Args:
            skill: Skill name to normalize

        Returns:
            Canonical skill name, or original if not found
        """
        skill_lower = skill.lower().strip()

        # Check if it's already canonical
        if skill_lower in self.SKILL_ALIASES:
            return skill_lower

        # Check aliases
        if skill_lower in self.ALIAS_TO_CANONICAL:
            return self.ALIAS_TO_CANONICAL[skill_lower]

        # Return original if not found
        return skill_lower

    def add_skill_alias(self, canonical: str, aliases: List[str]):
        """
        Add a new skill and its aliases to the database.

        Args:
            canonical: Canonical form of the skill
            aliases: List of aliases for this skill
        """
        canonical_lower = canonical.lower()

        if canonical_lower not in self.SKILL_ALIASES:
            self.SKILL_ALIASES[canonical_lower] = []

        self.SKILL_ALIASES[canonical_lower].extend([a.lower() for a in aliases])

        # Update reverse mapping
        for alias in aliases:
            self.ALIAS_TO_CANONICAL[alias.lower()] = canonical_lower

    def get_all_canonical_skills(self) -> List[str]:
        """Get all canonical skill names."""
        return list(self.SKILL_ALIASES.keys())


class SkillExtractor:
    """
    Extracts skills from text using KeyBERT and spaCy.
    Handles skill normalization and provides various extraction strategies.
    """

    def __init__(
        self,
        embedding_model: str = "all-MiniLM-L6-v2",
        spacy_model: str = "en_core_web_sm",
        top_n: int = 20,
        keyphrase_ngram_range: Tuple[int, int] = (1, 3),
        diversity: float = 0.5,
        use_normalization: bool = True
    ):
        """
        Initialize the SkillExtractor.

        Args:
            embedding_model: SentenceTransformer model for KeyBERT
            spacy_model: spaCy model for NER and POS tagging
            top_n: Number of top skills to extract
            keyphrase_ngram_range: N-gram range for keyphrase extraction
            diversity: Diversity parameter for MMR (0=no diversity, 1=max diversity)
            use_normalization: Whether to normalize extracted skills
        """
        logger.info(f"Initializing SkillExtractor with model: {embedding_model}")

        # Initialize KeyBERT
        try:
            self.kw_model = KeyBERT(model=embedding_model)
            logger.info("KeyBERT model initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize KeyBERT: {e}")
            raise

        # Initialize spaCy
        try:
            self.nlp = spacy.load(spacy_model)
            logger.info(f"spaCy model {spacy_model} loaded successfully")
        except OSError:
            logger.error(f"spaCy model {spacy_model} not found")
            raise

        # Configuration
        self.top_n = top_n
        self.keyphrase_ngram_range = keyphrase_ngram_range
        self.diversity = diversity
        self.use_normalization = use_normalization

        # Initialize normalizer
        self.normalizer = SkillNormalizer()

        logger.info(f"SkillExtractor initialized (top_n={top_n}, diversity={diversity})")

    def extract_skills_keybert(
        self,
        text: str,
        top_n: Optional[int] = None,
        diversity: Optional[float] = None,
        return_scores: bool = False
    ) -> List[str] | List[Tuple[str, float]]:
        """
        Extract skills using KeyBERT keyword extraction.

        Args:
            text: Input text (resume or job description)
            top_n: Number of skills to extract (overrides default)
            diversity: Diversity parameter (overrides default)
            return_scores: If True, return (skill, score) tuples

        Returns:
            List of skills or (skill, score) tuples
        """
        if not text or not isinstance(text, str):
            logger.warning("Invalid input text")
            return []

        n = top_n if top_n is not None else self.top_n
        div = diversity if diversity is not None else self.diversity

        try:
            # Extract keywords using KeyBERT with MMR
            keywords = self.kw_model.extract_keywords(
                text,
                keyphrase_ngram_range=self.keyphrase_ngram_range,
                stop_words='english',
                top_n=n,
                use_mmr=True,
                diversity=div
            )

            logger.debug(f"Extracted {len(keywords)} keywords using KeyBERT")

            # Normalize skills if enabled
            if self.use_normalization:
                normalized = []
                for keyword, score in keywords:
                    norm_skill = self.normalizer.normalize(keyword)
                    normalized.append((norm_skill, score))
                keywords = normalized

            # Deduplicate while keeping highest scores
            skill_dict = {}
            for skill, score in keywords:
                if skill not in skill_dict or score > skill_dict[skill]:
                    skill_dict[skill] = score

            # Sort by score
            sorted_skills = sorted(skill_dict.items(), key=lambda x: x[1], reverse=True)

            if return_scores:
                return sorted_skills[:n]
            else:
                return [skill for skill, _ in sorted_skills[:n]]

        except Exception as e:
            logger.error(f"Error extracting skills with KeyBERT: {e}")
            return []

    def extract_skills_spacy(self, text: str) -> List[str]:
        """
        Extract skills using spaCy NER and POS tagging.
        Focuses on NOUN phrases and technical entities.

        Args:
            text: Input text

        Returns:
            List of extracted skill terms
        """
        doc = self.nlp(text)

        skills = set()

        # Extract noun chunks (potential skills)
        for chunk in doc.noun_chunks:
            # Filter out very short or very long chunks
            if 2 <= len(chunk.text) <= 30:
                skill = chunk.text.lower().strip()
                if self.use_normalization:
                    skill = self.normalizer.normalize(skill)
                skills.add(skill)

        # Extract named entities (ORG, PRODUCT, etc.)
        for ent in doc.ents:
            if ent.label_ in ['ORG', 'PRODUCT', 'GPE']:
                skill = ent.text.lower().strip()
                if self.use_normalization:
                    skill = self.normalizer.normalize(skill)
                skills.add(skill)

        logger.debug(f"Extracted {len(skills)} skills using spaCy")
        return list(skills)

    def extract_skills_hybrid(
        self,
        text: str,
        keybert_weight: float = 0.7,
        top_n: Optional[int] = None
    ) -> List[Tuple[str, float]]:
        """
        Hybrid approach combining KeyBERT and spaCy.

        Args:
            text: Input text
            keybert_weight: Weight for KeyBERT scores (1 - weight for spaCy)
            top_n: Number of skills to return

        Returns:
            List of (skill, combined_score) tuples
        """
        n = top_n if top_n is not None else self.top_n

        # Get KeyBERT skills with scores
        keybert_skills = self.extract_skills_keybert(text, return_scores=True, top_n=n*2)
        keybert_dict = {skill: score for skill, score in keybert_skills}

        # Get spaCy skills
        spacy_skills = self.extract_skills_spacy(text)
        spacy_dict = {skill: 1.0 for skill in spacy_skills}

        # Combine scores
        all_skills = set(keybert_dict.keys()) | set(spacy_dict.keys())
        combined_scores = {}

        for skill in all_skills:
            kb_score = keybert_dict.get(skill, 0.0)
            sp_score = spacy_dict.get(skill, 0.0)

            # Weighted combination
            combined = (keybert_weight * kb_score) + ((1 - keybert_weight) * sp_score)
            combined_scores[skill] = combined

        # Sort and return top N
        sorted_skills = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)

        logger.debug(f"Hybrid extraction: {len(sorted_skills)} unique skills")
        return sorted_skills[:n]

    def extract_skills_rake(
        self,
        text: str,
        top_n: Optional[int] = None,
        return_scores: bool = False
    ) -> List[str] | List[Tuple[str, float]]:
        """
        Extract skills using RAKE (Rapid Automatic Keyword Extraction).
        **DEFAULT METHOD** - Best F1 score (0.356), fastest (0.29s), best precision (0.400).

        Args:
            text: Input text (resume or job description)
            top_n: Number of skills to extract (overrides default)
            return_scores: If True, return (skill, score) tuples

        Returns:
            List of skills or (skill, score) tuples
        """
        if not text or not isinstance(text, str):
            logger.warning("Invalid input text")
            return []

        n = top_n if top_n is not None else self.top_n

        try:
            from rake_nltk import Rake

            # Initialize RAKE
            rake = Rake(max_length=3)  # Max n-gram size
            rake.extract_keywords_from_text(text)
            keywords_with_scores = rake.get_ranked_phrases_with_scores()

            # Normalize skills
            normalized_skills = {}
            for score, skill in keywords_with_scores[:n * 2]:  # Get extra for normalization
                norm_skill = self.normalizer.normalize(skill)
                if norm_skill not in normalized_skills or score > normalized_skills[norm_skill]:
                    normalized_skills[norm_skill] = score

            # Normalize scores to 0-1 range
            if normalized_skills:
                max_score = max(normalized_skills.values())
                if max_score > 0:
                    normalized_skills = {k: v/max_score for k, v in normalized_skills.items()}

            # Sort by score
            sorted_skills = sorted(normalized_skills.items(), key=lambda x: x[1], reverse=True)

            logger.debug(f"Extracted {len(sorted_skills)} skills using RAKE")

            if return_scores:
                return sorted_skills[:n]
            else:
                return [skill for skill, _ in sorted_skills[:n]]

        except ImportError:
            logger.error("RAKE not installed. Install with: pip install rake-nltk")
            logger.warning("Falling back to KeyBERT method")
            return self.extract_skills_keybert(text, top_n=top_n, return_scores=return_scores)
        except Exception as e:
            logger.error(f"Error extracting skills with RAKE: {e}")
            return []

    def extract_skills(
        self,
        text: str,
        method: str = "rake",
        top_n: Optional[int] = None,
        return_scores: bool = False
    ) -> List[str] | List[Tuple[str, float]]:
        """
        Main skill extraction method with multiple strategy options.

        Args:
            text: Input text
            method: Extraction method ('rake' [DEFAULT], 'keybert', 'spacy', or 'hybrid')
            top_n: Number of skills to extract
            return_scores: If True, return (skill, score) tuples

        Returns:
            List of skills or (skill, score) tuples
        """
        if method == "rake":
            return self.extract_skills_rake(text, top_n=top_n, return_scores=return_scores)
        elif method == "keybert":
            return self.extract_skills_keybert(text, top_n=top_n, return_scores=return_scores)
        elif method == "spacy":
            skills = self.extract_skills_spacy(text)
            if return_scores:
                return [(s, 1.0) for s in skills[:top_n if top_n else self.top_n]]
            return skills[:top_n if top_n else self.top_n]
        elif method == "hybrid":
            results = self.extract_skills_hybrid(text, top_n=top_n)
            if return_scores:
                return results
            return [skill for skill, _ in results]
        else:
            logger.error(f"Unknown extraction method: {method}")
            return []

    def match_skills(
        self,
        resume_skills: List[str],
        job_skills: List[str]
    ) -> Dict[str, List[str]]:
        """
        Match resume skills against job requirements.

        Args:
            resume_skills: Skills extracted from resume
            job_skills: Skills extracted from job posting

        Returns:
            Dictionary with matched, missing, and extra skills
        """
        # Normalize all skills
        resume_normalized = {self.normalizer.normalize(s) for s in resume_skills}
        job_normalized = {self.normalizer.normalize(s) for s in job_skills}

        matched = resume_normalized & job_normalized
        missing = job_normalized - resume_normalized
        extra = resume_normalized - job_normalized

        return {
            'matched': sorted(list(matched)),
            'missing': sorted(list(missing)),
            'extra': sorted(list(extra)),
            'match_percentage': (len(matched) / len(job_normalized) * 100) if job_normalized else 0.0
        }


# Example usage and testing
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Sample resume text
    resume_text = """
    Senior Software Engineer with 7 years of experience in Python, JavaScript, and Java.
    Expert in machine learning, deep learning, and natural language processing.
    Proficient in AWS, Docker, Kubernetes, and CI/CD pipelines.
    Strong background in React, Node.js, and MongoDB.
    Experience with TensorFlow, PyTorch, and scikit-learn for ML projects.
    """

    # Sample job description
    job_text = """
    We are looking for a Senior ML Engineer with expertise in Python and TensorFlow.
    Required skills: machine learning, deep learning, NLP, AWS, Docker.
    Experience with React and Node.js is a plus.
    Must have strong problem-solving and leadership skills.
    Kubernetes and CI/CD experience preferred.
    """

    # Create skill extractor
    print("Initializing SkillExtractor...")
    extractor = SkillExtractor(top_n=15, diversity=0.5)

    print("\n" + "="*80)
    print("RESUME SKILLS (KeyBERT):")
    resume_skills = extractor.extract_skills(resume_text, method="keybert", return_scores=True)
    for skill, score in resume_skills:
        print(f"  {skill}: {score:.3f}")

    print("\n" + "="*80)
    print("JOB SKILLS (KeyBERT):")
    job_skills = extractor.extract_skills(job_text, method="keybert", return_scores=True)
    for skill, score in job_skills:
        print(f"  {skill}: {score:.3f}")

    print("\n" + "="*80)
    print("SKILL MATCHING:")
    resume_skill_list = [s for s, _ in resume_skills]
    job_skill_list = [s for s, _ in job_skills]

    match_results = extractor.match_skills(resume_skill_list, job_skill_list)

    print(f"Match Percentage: {match_results['match_percentage']:.1f}%")
    print(f"\nMatched Skills ({len(match_results['matched'])}):")
    for skill in match_results['matched']:
        print(f"  [MATCH] {skill}")

    print(f"\nMissing Skills ({len(match_results['missing'])}):")
    for skill in match_results['missing']:
        print(f"  [MISS] {skill}")

    print(f"\nExtra Skills ({len(match_results['extra'])}):")
    for skill in match_results['extra']:
        print(f"  [EXTRA] {skill}")

    print("\n" + "="*80)
    print("HYBRID EXTRACTION (Resume):")
    hybrid_skills = extractor.extract_skills(resume_text, method="hybrid", return_scores=True, top_n=10)
    for skill, score in hybrid_skills:
        print(f"  {skill}: {score:.3f}")
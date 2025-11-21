"""
Text Preprocessing Module

This module provides comprehensive text preprocessing utilities for resume and job posting text.
Features include cleaning, tokenization, lemmatization, and stopword removal using spaCy.

Author: Mert Alp Aydin
Date: 2025-11-20
"""

import re
import logging
from typing import List, Dict, Optional, Set
from enum import Enum
import spacy
from spacy.language import Language
from spacy.tokens import Doc

# Set up logging
logger = logging.getLogger(__name__)


class PreprocessingMode(Enum):
    """Enumeration for different preprocessing modes"""
    MINIMAL = "minimal"  # Only basic cleaning
    STANDARD = "standard"  # Cleaning + tokenization
    FULL = "full"  # All preprocessing including lemmatization


class TextPreprocessor:
    """
    Comprehensive text preprocessing class for NLP tasks.

    This class provides various text preprocessing methods including:
    - Text cleaning (lowercase, special chars, URLs, emails)
    - Tokenization using spaCy
    - Lemmatization
    - Stopword removal
    - Custom preprocessing pipelines
    """

    def __init__(
        self,
        model_name: str = "en_core_web_sm",
        remove_stopwords: bool = True,
        lowercase: bool = True,
        remove_punctuation: bool = True,
        custom_stopwords: Optional[Set[str]] = None
    ):
        """
        Initialize the TextPreprocessor.

        Args:
            model_name: spaCy model to load (default: en_core_web_sm)
            remove_stopwords: Whether to remove stopwords
            lowercase: Whether to convert text to lowercase
            remove_punctuation: Whether to remove punctuation
            custom_stopwords: Additional custom stopwords to remove
        """
        logger.info(f"Initializing TextPreprocessor with model: {model_name}")

        try:
            self.nlp: Language = spacy.load(model_name)
            logger.info(f"Successfully loaded spaCy model: {model_name}")
        except OSError:
            logger.error(f"Model {model_name} not found. Please download it first.")
            raise

        self.remove_stopwords = remove_stopwords
        self.lowercase = lowercase
        self.remove_punctuation = remove_punctuation

        # Build stopwords set
        self.stopwords: Set[str] = set(self.nlp.Defaults.stop_words)
        if custom_stopwords:
            self.stopwords.update(custom_stopwords)
            logger.info(f"Added {len(custom_stopwords)} custom stopwords")

        logger.info(f"TextPreprocessor initialized with {len(self.stopwords)} stopwords")

    def clean_text(
        self,
        text: str,
        remove_urls: bool = True,
        remove_emails: bool = True,
        remove_extra_whitespace: bool = True,
        remove_special_chars: bool = False
    ) -> str:
        """
        Clean text by removing URLs, emails, and special characters.

        Args:
            text: Input text to clean
            remove_urls: Remove URLs from text
            remove_emails: Remove email addresses from text
            remove_extra_whitespace: Remove extra whitespace
            remove_special_chars: Remove special characters (keep only alphanumeric and spaces)

        Returns:
            Cleaned text string
        """
        if not text or not isinstance(text, str):
            logger.warning("Invalid input text received")
            return ""

        cleaned = text

        # Remove URLs
        if remove_urls:
            url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
            cleaned = re.sub(url_pattern, '', cleaned)

        # Remove email addresses
        if remove_emails:
            email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
            cleaned = re.sub(email_pattern, '', cleaned)

        # Remove special characters (optional)
        if remove_special_chars:
            # Keep only alphanumeric, spaces, and basic punctuation
            cleaned = re.sub(r'[^a-zA-Z0-9\s\.,!?-]', '', cleaned)

        # Lowercase conversion
        if self.lowercase:
            cleaned = cleaned.lower()

        # Remove extra whitespace
        if remove_extra_whitespace:
            cleaned = re.sub(r'\s+', ' ', cleaned)
            cleaned = cleaned.strip()

        return cleaned

    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text using spaCy.

        Args:
            text: Input text to tokenize

        Returns:
            List of token strings
        """
        doc: Doc = self.nlp(text)
        tokens = [token.text for token in doc]
        logger.debug(f"Tokenized text into {len(tokens)} tokens")
        return tokens

    def lemmatize(self, text: str, keep_pos: Optional[List[str]] = None) -> List[str]:
        """
        Lemmatize text using spaCy.

        Args:
            text: Input text to lemmatize
            keep_pos: List of POS tags to keep (e.g., ['NOUN', 'VERB', 'ADJ'])
                     If None, keeps all tokens

        Returns:
            List of lemmatized tokens
        """
        doc: Doc = self.nlp(text)

        lemmas = []
        for token in doc:
            # Skip if filtering by POS tags and token doesn't match
            if keep_pos and token.pos_ not in keep_pos:
                continue

            # Skip punctuation if configured
            if self.remove_punctuation and token.is_punct:
                continue

            # Skip stopwords if configured
            if self.remove_stopwords and token.is_stop:
                continue

            # Skip whitespace tokens
            if token.is_space:
                continue

            lemmas.append(token.lemma_)

        logger.debug(f"Lemmatized text into {len(lemmas)} lemmas")
        return lemmas

    def preprocess(
        self,
        text: str,
        mode: PreprocessingMode = PreprocessingMode.FULL,
        return_as_string: bool = True,
        keep_pos: Optional[List[str]] = None
    ) -> str | List[str]:
        """
        Apply full preprocessing pipeline to text.

        Args:
            text: Input text to preprocess
            mode: Preprocessing mode (MINIMAL, STANDARD, or FULL)
            return_as_string: If True, return joined string; if False, return list of tokens
            keep_pos: Optional POS tags to keep (only for FULL mode)

        Returns:
            Preprocessed text as string or list of tokens
        """
        if not text:
            return "" if return_as_string else []

        logger.debug(f"Preprocessing text with mode: {mode.value}")

        # Step 1: Clean text (always performed)
        cleaned = self.clean_text(text)

        if mode == PreprocessingMode.MINIMAL:
            return cleaned if return_as_string else [cleaned]

        # Step 2: Tokenization for STANDARD mode
        if mode == PreprocessingMode.STANDARD:
            tokens = self.tokenize(cleaned)

            # Remove stopwords and punctuation if configured
            filtered_tokens = []
            doc = self.nlp(cleaned)
            for token in doc:
                if self.remove_punctuation and token.is_punct:
                    continue
                if self.remove_stopwords and token.is_stop:
                    continue
                if token.is_space:
                    continue
                filtered_tokens.append(token.text)

            return ' '.join(filtered_tokens) if return_as_string else filtered_tokens

        # Step 3: Full preprocessing with lemmatization
        if mode == PreprocessingMode.FULL:
            lemmas = self.lemmatize(cleaned, keep_pos=keep_pos)
            return ' '.join(lemmas) if return_as_string else lemmas

        return cleaned

    def batch_preprocess(
        self,
        texts: List[str],
        mode: PreprocessingMode = PreprocessingMode.FULL,
        return_as_string: bool = True,
        batch_size: int = 100
    ) -> List[str | List[str]]:
        """
        Preprocess multiple texts efficiently using spaCy's pipe.

        Args:
            texts: List of input texts
            mode: Preprocessing mode
            return_as_string: If True, return joined strings; if False, return lists of tokens
            batch_size: Number of texts to process at once

        Returns:
            List of preprocessed texts
        """
        logger.info(f"Batch preprocessing {len(texts)} texts with mode: {mode.value}")

        results = []

        # Clean all texts first
        cleaned_texts = [self.clean_text(text) for text in texts]

        if mode == PreprocessingMode.MINIMAL:
            return cleaned_texts if return_as_string else [[t] for t in cleaned_texts]

        # Process with spaCy pipe for efficiency
        for doc in self.nlp.pipe(cleaned_texts, batch_size=batch_size):
            if mode == PreprocessingMode.STANDARD:
                tokens = []
                for token in doc:
                    if self.remove_punctuation and token.is_punct:
                        continue
                    if self.remove_stopwords and token.is_stop:
                        continue
                    if token.is_space:
                        continue
                    tokens.append(token.text)

                results.append(' '.join(tokens) if return_as_string else tokens)

            elif mode == PreprocessingMode.FULL:
                lemmas = []
                for token in doc:
                    if self.remove_punctuation and token.is_punct:
                        continue
                    if self.remove_stopwords and token.is_stop:
                        continue
                    if token.is_space:
                        continue
                    lemmas.append(token.lemma_)

                results.append(' '.join(lemmas) if return_as_string else lemmas)

        logger.info(f"Batch preprocessing completed: {len(results)} texts processed")
        return results

    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """
        Extract named entities from text using spaCy.

        Args:
            text: Input text

        Returns:
            Dictionary mapping entity types to lists of entity texts
        """
        doc: Doc = self.nlp(text)

        entities: Dict[str, List[str]] = {}
        for ent in doc.ents:
            if ent.label_ not in entities:
                entities[ent.label_] = []
            entities[ent.label_].append(ent.text)

        logger.debug(f"Extracted {sum(len(v) for v in entities.values())} entities")
        return entities

    def get_pos_tags(self, text: str) -> List[tuple]:
        """
        Get part-of-speech tags for text.

        Args:
            text: Input text

        Returns:
            List of (token, pos_tag) tuples
        """
        doc: Doc = self.nlp(text)
        pos_tags = [(token.text, token.pos_) for token in doc]
        return pos_tags

    def get_statistics(self, text: str) -> Dict[str, any]:
        """
        Get various statistics about the text.

        Args:
            text: Input text

        Returns:
            Dictionary containing text statistics
        """
        doc: Doc = self.nlp(text)

        stats = {
            'total_tokens': len(doc),
            'total_sentences': len(list(doc.sents)),
            'total_chars': len(text),
            'avg_token_length': sum(len(token.text) for token in doc) / len(doc) if len(doc) > 0 else 0,
            'stopword_count': sum(1 for token in doc if token.is_stop),
            'punctuation_count': sum(1 for token in doc if token.is_punct),
            'unique_tokens': len(set(token.text.lower() for token in doc)),
        }

        return stats


def create_preprocessor(
    remove_stopwords: bool = True,
    lowercase: bool = True,
    remove_punctuation: bool = True,
    custom_stopwords: Optional[Set[str]] = None
) -> TextPreprocessor:
    """
    Factory function to create a TextPreprocessor instance.

    Args:
        remove_stopwords: Whether to remove stopwords
        lowercase: Whether to convert to lowercase
        remove_punctuation: Whether to remove punctuation
        custom_stopwords: Additional stopwords to use

    Returns:
        Configured TextPreprocessor instance
    """
    return TextPreprocessor(
        remove_stopwords=remove_stopwords,
        lowercase=lowercase,
        remove_punctuation=remove_punctuation,
        custom_stopwords=custom_stopwords
    )


# Example usage and testing
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Sample text for testing
    sample_text = """
    Senior Software Engineer with 5+ years of experience in Python, Java, and JavaScript.
    Visit my website at https://example.com or email me at john.doe@email.com.
    Expertise in machine learning, deep learning, and natural language processing.
    Strong background in AWS, Docker, and Kubernetes!!!
    """

    # Create preprocessor
    preprocessor = create_preprocessor()

    # Test different preprocessing modes
    print("=" * 80)
    print("ORIGINAL TEXT:")
    print(sample_text)
    print()

    print("=" * 80)
    print("MINIMAL PREPROCESSING:")
    minimal = preprocessor.preprocess(sample_text, mode=PreprocessingMode.MINIMAL)
    print(minimal)
    print()

    print("=" * 80)
    print("STANDARD PREPROCESSING (tokens):")
    standard = preprocessor.preprocess(sample_text, mode=PreprocessingMode.STANDARD, return_as_string=False)
    print(standard)
    print()

    print("=" * 80)
    print("FULL PREPROCESSING (lemmas):")
    full = preprocessor.preprocess(sample_text, mode=PreprocessingMode.FULL)
    print(full)
    print()

    print("=" * 80)
    print("NAMED ENTITIES:")
    entities = preprocessor.extract_entities(sample_text)
    for entity_type, entity_list in entities.items():
        print(f"{entity_type}: {entity_list}")
    print()

    print("=" * 80)
    print("TEXT STATISTICS:")
    stats = preprocessor.get_statistics(sample_text)
    for key, value in stats.items():
        print(f"{key}: {value}")
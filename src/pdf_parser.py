"""
PDF Resume Parser Module

This module provides functionality to extract text from PDF resumes using
multiple parsing strategies (PDFPlumber, PyPDF) with LangChain integration.

Author: Mert Alp Aydin
Date: 2025-11-18
Phase: Phase 2 - Step 2.2
"""

from pathlib import Path
from typing import Optional, Dict, Any, List
import time

# LangChain imports for document loading
from langchain_community.document_loaders import PyPDFLoader, PDFPlumberLoader

from logging_utils import setup_logger

# Initialize logger
logger = setup_logger(__name__)


class PDFResumeParser:
    """
    Handles PDF resume parsing with multiple strategies for robustness.

    Supports:
    - PDFPlumber (better for complex layouts, tables)
    - PyPDF (faster, simpler)

    Includes metadata tracking and error handling.
    """

    def __init__(self, default_loader: str = "pdfplumber"):
        """
        Initialize the PDF parser.

        Args:
            default_loader: Default loading strategy ('pdfplumber' or 'pypdf')
        """
        self.default_loader = default_loader
        logger.info(f"PDFResumeParser initialized with default loader: {default_loader}")

    def parse_with_pdfplumber(self, pdf_path: Path) -> Dict[str, Any]:
        """
        Parse PDF using PDFPlumber loader.

        Args:
            pdf_path: Path to PDF file

        Returns:
            Dict with 'text', 'metadata', 'pages', 'success', 'error'
        """
        try:
            logger.info(f"Parsing with PDFPlumber: {pdf_path.name}")
            start_time = time.time()

            loader = PDFPlumberLoader(str(pdf_path))
            documents = loader.load()

            end_time = time.time()
            parse_time = end_time - start_time

            # Combine all pages
            full_text = "\n\n".join([doc.page_content for doc in documents])

            result = {
                'text': full_text,
                'num_pages': len(documents),
                'pages': [doc.page_content for doc in documents],
                'metadata': {
                    'source': str(pdf_path),
                    'loader': 'pdfplumber',
                    'parse_time_seconds': round(parse_time, 3),
                    'page_count': len(documents),
                    'char_count': len(full_text),
                    'word_count': len(full_text.split())
                },
                'success': True,
                'error': None
            }

            logger.info(f"PDFPlumber success: {len(documents)} pages, "
                       f"{result['metadata']['word_count']} words, "
                       f"{parse_time:.2f}s")

            return result

        except Exception as e:
            logger.error(f"PDFPlumber parsing failed for {pdf_path.name}: {str(e)}")
            return {
                'text': None,
                'num_pages': 0,
                'pages': [],
                'metadata': {'source': str(pdf_path), 'loader': 'pdfplumber'},
                'success': False,
                'error': str(e)
            }

    def parse_with_pypdf(self, pdf_path: Path) -> Dict[str, Any]:
        """
        Parse PDF using PyPDF loader.

        Args:
            pdf_path: Path to PDF file

        Returns:
            Dict with 'text', 'metadata', 'pages', 'success', 'error'
        """
        try:
            logger.info(f"Parsing with PyPDF: {pdf_path.name}")
            start_time = time.time()

            loader = PyPDFLoader(str(pdf_path))
            documents = loader.load()

            end_time = time.time()
            parse_time = end_time - start_time

            # Combine all pages
            full_text = "\n\n".join([doc.page_content for doc in documents])

            result = {
                'text': full_text,
                'num_pages': len(documents),
                'pages': [doc.page_content for doc in documents],
                'metadata': {
                    'source': str(pdf_path),
                    'loader': 'pypdf',
                    'parse_time_seconds': round(parse_time, 3),
                    'page_count': len(documents),
                    'char_count': len(full_text),
                    'word_count': len(full_text.split())
                },
                'success': True,
                'error': None
            }

            logger.info(f"PyPDF success: {len(documents)} pages, "
                       f"{result['metadata']['word_count']} words, "
                       f"{parse_time:.2f}s")

            return result

        except Exception as e:
            logger.error(f"PyPDF parsing failed for {pdf_path.name}: {str(e)}")
            return {
                'text': None,
                'num_pages': 0,
                'pages': [],
                'metadata': {'source': str(pdf_path), 'loader': 'pypdf'},
                'success': False,
                'error': str(e)
            }

    def parse(self, pdf_path: Path, loader: Optional[str] = None) -> Dict[str, Any]:
        """
        Parse a PDF resume using specified or default loader.

        Args:
            pdf_path: Path to PDF file
            loader: Specific loader to use ('pdfplumber', 'pypdf') or None for default

        Returns:
            Dict with parsing results including text, metadata, and status
        """
        pdf_path = Path(pdf_path)

        if not pdf_path.exists():
            logger.error(f"PDF file not found: {pdf_path}")
            return {
                'text': None,
                'num_pages': 0,
                'pages': [],
                'metadata': {'source': str(pdf_path)},
                'success': False,
                'error': 'File not found'
            }

        # Determine which loader to use
        loader_choice = loader if loader else self.default_loader

        if loader_choice.lower() == 'pdfplumber':
            return self.parse_with_pdfplumber(pdf_path)
        elif loader_choice.lower() == 'pypdf':
            return self.parse_with_pypdf(pdf_path)
        else:
            logger.error(f"Unknown loader: {loader_choice}")
            return {
                'text': None,
                'num_pages': 0,
                'pages': [],
                'metadata': {'source': str(pdf_path)},
                'success': False,
                'error': f'Unknown loader: {loader_choice}'
            }

    def parse_with_fallback(self, pdf_path: Path) -> Dict[str, Any]:
        """
        Parse PDF with automatic fallback strategy.

        Tries PDFPlumber first, falls back to PyPDF if it fails.

        Args:
            pdf_path: Path to PDF file

        Returns:
            Dict with parsing results from successful loader
        """
        logger.info(f"Parsing with fallback strategy: {pdf_path.name}")

        # Try PDFPlumber first
        result = self.parse_with_pdfplumber(pdf_path)

        if result['success']:
            return result

        # Fallback to PyPDF
        logger.warning(f"PDFPlumber failed, falling back to PyPDF for {pdf_path.name}")
        result = self.parse_with_pypdf(pdf_path)

        if result['success']:
            result['metadata']['fallback_used'] = True
            result['metadata']['original_loader_failed'] = 'pdfplumber'

        return result

    def compare_loaders(self, pdf_path: Path) -> Dict[str, Any]:
        """
        Compare both loaders on the same PDF for experimentation.

        Args:
            pdf_path: Path to PDF file

        Returns:
            Dict with results from both loaders for comparison
        """
        logger.info(f"Comparing loaders for: {pdf_path.name}")

        pdfplumber_result = self.parse_with_pdfplumber(pdf_path)
        pypdf_result = self.parse_with_pypdf(pdf_path)

        comparison = {
            'pdf_file': str(pdf_path),
            'pdfplumber': pdfplumber_result,
            'pypdf': pypdf_result,
            'comparison': {
                'both_succeeded': pdfplumber_result['success'] and pypdf_result['success'],
                'pdfplumber_faster': (
                    pdfplumber_result['metadata'].get('parse_time_seconds', float('inf')) <
                    pypdf_result['metadata'].get('parse_time_seconds', float('inf'))
                ) if pdfplumber_result['success'] and pypdf_result['success'] else None,
                'word_count_difference': abs(
                    pdfplumber_result['metadata'].get('word_count', 0) -
                    pypdf_result['metadata'].get('word_count', 0)
                ) if pdfplumber_result['success'] and pypdf_result['success'] else None
            }
        }

        logger.info(f"Comparison complete for {pdf_path.name}")
        return comparison


def test_pdf_parser():
    """Test the PDF parser with sample resumes."""
    logger.info("="*70)
    logger.info("PDF PARSER TEST - PHASE 2 STEP 2.2")
    logger.info("="*70)

    # Initialize parser
    parser = PDFResumeParser(default_loader="pdfplumber")

    # Get sample PDFs
    samples_dir = Path(__file__).parent.parent / "data" / "cv_samples" / "ai generated"
    pdf_files = list(samples_dir.glob("*.pdf"))

    if not pdf_files:
        logger.error("No sample PDF files found!")
        return

    logger.info(f"Found {len(pdf_files)} sample PDFs")

    # Test each PDF with fallback strategy
    results = []
    for pdf_file in pdf_files[:3]:  # Test first 3 samples
        logger.info(f"\n{'='*60}")
        logger.info(f"Testing: {pdf_file.name}")
        logger.info(f"{'='*60}")

        result = parser.parse_with_fallback(pdf_file)
        results.append(result)

        if result['success']:
            print(f"\n[OK] {pdf_file.name}")
            print(f"  Pages: {result['num_pages']}")
            print(f"  Words: {result['metadata']['word_count']}")
            print(f"  Loader: {result['metadata']['loader']}")
            print(f"  Time: {result['metadata']['parse_time_seconds']}s")
            print(f"  Preview: {result['text'][:200]}...")
        else:
            print(f"\n[FAIL] {pdf_file.name}")
            print(f"  Error: {result['error']}")

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    successful = sum(1 for r in results if r['success'])
    print(f"Successfully parsed: {successful}/{len(results)}")
    avg_time = sum(r['metadata'].get('parse_time_seconds', 0) for r in results if r['success']) / max(successful, 1)
    print(f"Average parse time: {avg_time:.2f}s")

    logger.info("PDF parser test complete!")
    return results


if __name__ == "__main__":
    test_pdf_parser()
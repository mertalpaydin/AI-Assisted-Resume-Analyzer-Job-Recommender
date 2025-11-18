"""
PDF Parser Quality Assessment with User Validation

This script performs quality checks on PDF parsing and allows for
user-rated validation of extraction quality.

Author: Mert Alp Aydin
Date: 2025-11-18
Phase: Phase 2 - Step 2.2 (Quality Assessment)
"""

from pathlib import Path
from typing import Dict, Any, List
import re
import json
from pdf_parser import PDFResumeParser
from logging_utils import setup_logger, log_experiment

logger = setup_logger(__name__)


class PDFQualityValidator:
    """Validates the quality of PDF text extraction."""

    def __init__(self):
        self.parser = PDFResumeParser(default_loader="pdfplumber")

    def check_key_fields_presence(self, text: str) -> Dict[str, Any]:
        """
        Check if resume contains expected key fields.

        Args:
            text: Extracted resume text

        Returns:
            Dict with presence indicators for key fields
        """
        text_lower = text.lower()

        checks = {
            'has_email': bool(re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text)),
            'has_phone': bool(re.search(r'[\+\(]?[0-9][0-9 .\-\(\)]{8,}[0-9]', text)),
            'has_experience_section': any(keyword in text_lower for keyword in
                                         ['experience', 'employment', 'work history', 'professional experience']),
            'has_education_section': any(keyword in text_lower for keyword in
                                        ['education', 'academic', 'degree', 'university', 'college']),
            'has_skills_section': any(keyword in text_lower for keyword in
                                     ['skills', 'technical skills', 'competencies', 'technologies']),
            'has_name_candidate': len(text.split('\n')[0].split()) <= 5,  # First line is usually name
            'has_reasonable_length': 100 < len(text.split()) < 5000,  # Between 100-5000 words
        }

        checks['quality_score'] = sum(checks.values()) / len(checks)

        return checks

    def check_text_quality(self, text: str) -> Dict[str, Any]:
        """
        Check for common text extraction issues.

        Args:
            text: Extracted text

        Returns:
            Dict with quality indicators
        """
        checks = {
            'has_garbled_chars': bool(re.search(r'[^\x00-\x7F]{10,}', text)),  # Long sequences of non-ASCII
            'has_excessive_whitespace': bool(re.search(r'\s{10,}', text)),
            'has_repeated_chars': bool(re.search(r'(.)\1{10,}', text)),
            'char_count': len(text),
            'word_count': len(text.split()),
            'line_count': len(text.split('\n')),
            'avg_words_per_line': len(text.split()) / max(len(text.split('\n')), 1),
        }

        # Overall text quality (lower is better for issues)
        issue_count = sum([
            checks['has_garbled_chars'],
            checks['has_excessive_whitespace'],
            checks['has_repeated_chars']
        ])

        checks['has_quality_issues'] = issue_count > 0
        checks['issue_count'] = issue_count

        return checks

    def compare_loaders(self, pdf_path: Path) -> Dict[str, Any]:
        """
        Compare both loaders and analyze quality differences.

        Args:
            pdf_path: Path to PDF file

        Returns:
            Detailed comparison with quality metrics
        """
        logger.info(f"Quality assessment for: {pdf_path.name}")

        comparison = self.parser.compare_loaders(pdf_path)

        # Add quality checks for both loaders
        if comparison['pdfplumber']['success']:
            pdfplumber_text = comparison['pdfplumber']['text']
            comparison['pdfplumber']['quality'] = {
                'field_presence': self.check_key_fields_presence(pdfplumber_text),
                'text_quality': self.check_text_quality(pdfplumber_text)
            }

        if comparison['pypdf']['success']:
            pypdf_text = comparison['pypdf']['text']
            comparison['pypdf']['quality'] = {
                'field_presence': self.check_key_fields_presence(pypdf_text),
                'text_quality': self.check_text_quality(pypdf_text)
            }

        return comparison

    def user_rate_extraction(self, pdf_path: Path, extracted_text: str) -> Dict[str, Any]:
        """
        Display extracted text and prompt for user rating.

        Args:
            pdf_path: Path to PDF file
            extracted_text: Extracted text to review

        Returns:
            Dict with user rating and feedback
        """
        print("\n" + "="*80)
        print(f"PDF: {pdf_path.name}")
        print("="*80)
        print("\nEXTRACTED TEXT:")
        print("-"*80)
        print(extracted_text)


        print("\nPlease rate the extraction quality (1-5):")
        print("1 - Very Poor (mostly garbled or missing)")
        print("2 - Poor (significant issues)")
        print("3 - Fair (some issues but usable)")
        print("4 - Good (minor issues)")
        print("5 - Excellent (perfect or near-perfect)")

        while True:
            try:
                rating = int(input("\nYour rating: "))
                if 1 <= rating <= 5:
                    break
                print("Please enter a number between 1 and 5")
            except ValueError:
                print("Please enter a valid number")
            except (EOFError, KeyboardInterrupt):
                print("\nUser rating cancelled")
                return {'rating': None, 'feedback': 'Cancelled'}

        feedback = input("Optional feedback (press Enter to skip): ").strip()

        result = {
            'pdf_file': str(pdf_path),
            'rating': rating,
            'feedback': feedback if feedback else "No feedback provided",
            'text_length': len(extracted_text),
            'word_count': len(extracted_text.split())
        }

        return result


def run_quality_assessment(with_user_rating: bool = False):
    """
    Run comprehensive quality assessment on ALL sample PDFs with BOTH parsers.

    Args:
        with_user_rating: If True, prompt for manual user ratings
    """
    logger.info("="*70)
    logger.info("PDF PARSER QUALITY ASSESSMENT - PHASE 2 STEP 2.2")
    logger.info("="*70)

    validator = PDFQualityValidator()

    # Get ALL sample PDFs
    samples_dir = Path(__file__).parent.parent / "data" / "cv_samples" / "ai generated"
    pdf_files = list(samples_dir.glob("*.pdf"))  # ALL PDFs

    logger.info(f"Testing {len(pdf_files)} PDFs with BOTH parsers (PDFPlumber & PyPDF)")

    all_results = []
    user_ratings = []
    csv_rows = []

    # CSV header
    csv_rows.append("PDF_File,Parser,Success,Parse_Time_s,Word_Count,Has_Email,Has_Phone,Has_Experience,Has_Education,Has_Skills,Field_Score,Quality_Issues,User_Rating,User_Feedback")

    for pdf_file in pdf_files:
        print(f"\n{'='*80}")
        print(f"Assessing: {pdf_file.name}")
        print("="*80)

        # Test BOTH parsers
        for parser_name in ['pdfplumber', 'pypdf']:
            print(f"\n  Testing with: {parser_name.upper()}")

            # Parse with specific loader
            parse_result = validator.parser.parse(pdf_file, loader=parser_name)

            if not parse_result['success']:
                print(f"    [FAIL] Could not parse with {parser_name}")
                csv_rows.append(f"{pdf_file.name},{parser_name},False,0,0,False,False,False,False,False,0.0,N/A,N/A,Parse failed")
                continue

            # Automated quality checks
            field_presence = validator.check_key_fields_presence(parse_result['text'])
            text_quality = validator.check_text_quality(parse_result['text'])

            result = {
                'pdf_file': str(pdf_file),
                'parser': parser_name,
                'parse_metadata': parse_result['metadata'],
                'field_presence': field_presence,
                'text_quality': text_quality
            }

            all_results.append(result)

            # Display automated quality results
            print(f"    [AUTOMATED QUALITY CHECK]")
            print(f"      Field Presence Score: {field_presence['quality_score']:.1%}")
            print(f"        - Email: {field_presence['has_email']}")
            print(f"        - Phone: {field_presence['has_phone']}")
            print(f"        - Experience: {field_presence['has_experience_section']}")
            print(f"        - Education: {field_presence['has_education_section']}")
            print(f"        - Skills: {field_presence['has_skills_section']}")
            print(f"      Text Quality:")
            print(f"        - Issues: {text_quality['issue_count']}")
            print(f"        - Words: {text_quality['word_count']}")
            print(f"        - Parse Time: {parse_result['metadata']['parse_time_seconds']}s")

            # Add to CSV (without user rating for now)
            csv_row = f"{pdf_file.name},{parser_name},True,{parse_result['metadata']['parse_time_seconds']},{text_quality['word_count']},{field_presence['has_email']},{field_presence['has_phone']},{field_presence['has_experience_section']},{field_presence['has_education_section']},{field_presence['has_skills_section']},{field_presence['quality_score']:.3f},{text_quality['issue_count']},N/A,N/A"
            csv_rows.append(csv_row)

            # User rating if requested (only for pdfplumber to avoid redundancy)
            if with_user_rating and parser_name == 'pdfplumber':
                user_rating = validator.user_rate_extraction(pdf_file, parse_result['text'])
                user_ratings.append(user_rating)
                result['user_rating'] = user_rating

                # Update CSV row with user rating
                csv_rows[-1] = csv_row.replace(',N/A,N/A', f",{user_rating['rating']},{user_rating['feedback']}")

    # Summary
    print("\n" + "="*80)
    print("QUALITY ASSESSMENT SUMMARY")
    print("="*80)

    avg_field_score = sum(r['field_presence']['quality_score'] for r in all_results) / len(all_results)
    avg_issues = sum(r['text_quality']['issue_count'] for r in all_results) / len(all_results)
    avg_parse_time = sum(r['parse_metadata']['parse_time_seconds'] for r in all_results) / len(all_results)

    print(f"\nAutomated Metrics (across {len(all_results)} tests):")
    print(f"  Average field presence score: {avg_field_score:.1%}")
    print(f"  Average quality issues per document: {avg_issues:.1f}")
    print(f"  Average parse time: {avg_parse_time:.3f}s")

    # Compare parsers
    pdfplumber_results = [r for r in all_results if r['parser'] == 'pdfplumber']
    pypdf_results = [r for r in all_results if r['parser'] == 'pypdf']

    if pdfplumber_results and pypdf_results:
        print(f"\nParser Comparison:")
        print(f"  PDFPlumber avg score: {sum(r['field_presence']['quality_score'] for r in pdfplumber_results) / len(pdfplumber_results):.1%}")
        print(f"  PyPDF avg score: {sum(r['field_presence']['quality_score'] for r in pypdf_results) / len(pypdf_results):.1%}")
        print(f"  PDFPlumber avg time: {sum(r['parse_metadata']['parse_time_seconds'] for r in pdfplumber_results) / len(pdfplumber_results):.3f}s")
        print(f"  PyPDF avg time: {sum(r['parse_metadata']['parse_time_seconds'] for r in pypdf_results) / len(pypdf_results):.3f}s")

    if user_ratings:
        valid_ratings = [r for r in user_ratings if r['rating']]
        if valid_ratings:
            avg_rating = sum(r['rating'] for r in valid_ratings) / len(valid_ratings)
            print(f"\nUser Ratings:")
            print(f"  Average rating: {avg_rating:.1f}/5.0")
            for rating in user_ratings:
                if rating['rating']:
                    print(f"    {Path(rating['pdf_file']).name}: {rating['rating']}/5")
                    if rating['feedback'] != "No feedback provided":
                        print(f"      Feedback: {rating['feedback']}")

    # Save results as JSON, CSV, and TXT
    logs_dir = Path(__file__).parent.parent / "logs"

    # JSON
    json_file = logs_dir / "pdf_quality_assessment.json"
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump({
            'automated_results': all_results,
            'user_ratings': user_ratings if user_ratings else None,
            'summary': {
                'avg_field_presence_score': avg_field_score,
                'avg_quality_issues': avg_issues,
                'avg_parse_time': avg_parse_time,
                'avg_user_rating': avg_rating if user_ratings and valid_ratings else None
            }
        }, f, indent=2)

    # CSV
    csv_file = logs_dir / "pdf_quality_assessment.csv"
    with open(csv_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(csv_rows))

    # TXT (human readable)
    txt_file = logs_dir / "pdf_quality_assessment.txt"
    with open(txt_file, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("PDF PARSER QUALITY ASSESSMENT RESULTS\n")
        f.write("="*80 + "\n\n")

        for result in all_results:
            f.write(f"\nPDF: {Path(result['pdf_file']).name}\n")
            f.write(f"Parser: {result['parser'].upper()}\n")
            f.write(f"Parse Time: {result['parse_metadata']['parse_time_seconds']}s\n")
            f.write(f"Word Count: {result['text_quality']['word_count']}\n")
            f.write(f"\nField Presence (Score: {result['field_presence']['quality_score']:.1%}):\n")
            f.write(f"  Email: {result['field_presence']['has_email']}\n")
            f.write(f"  Phone: {result['field_presence']['has_phone']}\n")
            f.write(f"  Experience: {result['field_presence']['has_experience_section']}\n")
            f.write(f"  Education: {result['field_presence']['has_education_section']}\n")
            f.write(f"  Skills: {result['field_presence']['has_skills_section']}\n")
            f.write(f"\nText Quality:\n")
            f.write(f"  Issues Found: {result['text_quality']['issue_count']}\n")
            f.write(f"  Has Garbled Chars: {result['text_quality']['has_garbled_chars']}\n")
            f.write(f"  Has Excessive Whitespace: {result['text_quality']['has_excessive_whitespace']}\n")
            f.write("-"*80 + "\n")

        f.write(f"\n{'='*80}\n")
        f.write("SUMMARY\n")
        f.write(f"{'='*80}\n")
        f.write(f"Total tests: {len(all_results)}\n")
        f.write(f"Average field presence score: {avg_field_score:.1%}\n")
        f.write(f"Average quality issues: {avg_issues:.1f}\n")
        f.write(f"Average parse time: {avg_parse_time:.3f}s\n")

        if user_ratings and valid_ratings:
            f.write(f"\nUser Ratings (Average: {avg_rating:.1f}/5.0):\n")
            for rating in user_ratings:
                if rating['rating']:
                    f.write(f"  {Path(rating['pdf_file']).name}: {rating['rating']}/5\n")
                    if rating['feedback'] != "No feedback provided":
                        f.write(f"    Feedback: {rating['feedback']}\n")

    print(f"\n{'='*80}")
    print("RESULTS SAVED:")
    print(f"  JSON: {json_file}")
    print(f"  CSV:  {csv_file}")
    print(f"  TXT:  {txt_file}")
    print("="*80)

    logger.info(f"Quality assessment complete. Results saved in 3 formats.")

    return all_results, user_ratings


if __name__ == "__main__":
    print("\n" + "="*80)
    print("PDF PARSER QUALITY ASSESSMENT WITH USER RATING")
    print("="*80)
    print("You will be asked to manually rate each PDF extraction (5 CVs).")
    print("This helps validate the quality of our parsing.")
    print("\nFor each CV, you'll see the extracted text and rate it 1-5:")
    print("  5 = Excellent (perfect extraction)")
    print("  4 = Good (minor issues)")
    print("  3 = Fair (usable but some problems)")
    print("  2 = Poor (significant issues)")
    print("  1 = Very Poor (mostly broken)")

    response = input("\nReady to start? (y/n): ").strip().lower()

    if response == 'y' or response == 'yes' or response == '':
        run_quality_assessment(with_user_rating=True)
    else:
        print("Running automated assessment only (no user ratings)...")
        run_quality_assessment(with_user_rating=False)
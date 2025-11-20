"""
Resume Extraction Validator & Error Handler

Provides validation, error handling, and retry logic for resume extraction.
Ensures robust extraction with comprehensive error reporting.

Author: Mert Alp Aydin
Date: 2025-11-18
Phase: Phase 2 - Step 2.5
"""

from typing import Dict, Any, Optional, List
from pathlib import Path
import re

from resume_schema import Resume
from logging_utils import setup_logger

logger = setup_logger(__name__)


class ExtractionValidator:
    """
    Validates and provides detailed feedback on resume extraction quality.
    """

    @staticmethod
    def validate_contact_info(resume: Resume) -> Dict[str, Any]:
        """
        Validate contact information fields.

        Args:
            resume: Resume object to validate

        Returns:
            Dict with validation results
        """
        issues = []
        warnings = []

        # Email validation
        if resume.email:
            email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
            if not re.match(email_pattern, resume.email):
                warnings.append(f"Email format may be invalid: {resume.email}")
        else:
            warnings.append("No email address found")

        # Phone validation
        if resume.phone:
            # Should have at least 7 digits
            digits = re.findall(r'\d', resume.phone)
            if len(digits) < 7:
                warnings.append(f"Phone number seems incomplete: {resume.phone}")
        else:
            warnings.append("No phone number found")

        # At least one contact method
        if not resume.email and not resume.phone:
            issues.append("No contact information found (no email or phone)")

        return {
            'valid': len(issues) == 0,
            'issues': issues,
            'warnings': warnings
        }

    @staticmethod
    def validate_experience(resume: Resume) -> Dict[str, Any]:
        """
        Validate experience section.

        Args:
            resume: Resume object to validate

        Returns:
            Dict with validation results
        """
        issues = []
        warnings = []

        if not resume.experience or len(resume.experience) == 0:
            warnings.append("No work experience found")
        else:
            for i, exp in enumerate(resume.experience):
                # Check required fields
                if not exp.company or not exp.company.strip():
                    issues.append(f"Experience entry {i+1}: Missing company name")

                if not exp.position or not exp.position.strip():
                    issues.append(f"Experience entry {i+1}: Missing position/title")

                if not exp.duration or not exp.duration.strip():
                    warnings.append(f"Experience entry {i+1}: Missing duration")

                if not exp.description or not exp.description.strip():
                    warnings.append(f"Experience entry {i+1}: Missing description")

        return {
            'valid': len(issues) == 0,
            'issues': issues,
            'warnings': warnings,
            'count': len(resume.experience)
        }

    @staticmethod
    def validate_education(resume: Resume) -> Dict[str, Any]:
        """
        Validate education section.

        Args:
            resume: Resume object to validate

        Returns:
            Dict with validation results
        """
        issues = []
        warnings = []

        if not resume.education or len(resume.education) == 0:
            warnings.append("No education found")
        else:
            for i, edu in enumerate(resume.education):
                if not edu.institution or not edu.institution.strip():
                    issues.append(f"Education entry {i+1}: Missing institution name")

                if not edu.degree or not edu.degree.strip():
                    issues.append(f"Education entry {i+1}: Missing degree")

        return {
            'valid': len(issues) == 0,
            'issues': issues,
            'warnings': warnings,
            'count': len(resume.education)
        }

    @staticmethod
    def validate_skills(resume: Resume) -> Dict[str, Any]:
        """
        Validate skills section.

        Args:
            resume: Resume object to validate

        Returns:
            Dict with validation results
        """
        issues = []
        warnings = []

        if not resume.skills or len(resume.skills) == 0:
            warnings.append("No skills found")
        elif len(resume.skills) < 3:
            warnings.append(f"Very few skills extracted ({len(resume.skills)}) - may be incomplete")

        # Check for suspiciously generic skills
        generic_skills = {'communication', 'teamwork', 'leadership', 'problem solving'}
        technical_skills = [s for s in resume.skills if s.lower() not in generic_skills]

        if len(technical_skills) < 2:
            warnings.append("Very few technical skills found")

        return {
            'valid': len(issues) == 0,
            'issues': issues,
            'warnings': warnings,
            'count': len(resume.skills),
            'technical_count': len(technical_skills)
        }

    @classmethod
    def validate_resume(cls, resume: Resume) -> Dict[str, Any]:
        """
        Comprehensive resume validation.

        Args:
            resume: Resume object to validate

        Returns:
            Dict with complete validation results
        """
        logger.info("Running comprehensive resume validation...")

        # Validate each section
        contact_validation = cls.validate_contact_info(resume)
        experience_validation = cls.validate_experience(resume)
        education_validation = cls.validate_education(resume)
        skills_validation = cls.validate_skills(resume)

        # Aggregate results
        all_issues = (
            contact_validation['issues'] +
            experience_validation['issues'] +
            education_validation['issues'] +
            skills_validation['issues']
        )

        all_warnings = (
            contact_validation['warnings'] +
            experience_validation['warnings'] +
            education_validation['warnings'] +
            skills_validation['warnings']
        )

        # Overall validity
        is_valid = len(all_issues) == 0

        # Quality score (0-1)
        quality_score = 1.0
        quality_score -= len(all_issues) * 0.15  # Each issue reduces score by 15%
        quality_score -= len(all_warnings) * 0.05  # Each warning reduces by 5%
        quality_score = max(0.0, quality_score)  # Don't go below 0

        validation_result = {
            'valid': is_valid,
            'quality_score': round(quality_score, 2),
            'issues': all_issues,
            'warnings': all_warnings,
            'sections': {
                'contact': contact_validation,
                'experience': experience_validation,
                'education': education_validation,
                'skills': skills_validation
            },
            'summary': {
                'total_issues': len(all_issues),
                'total_warnings': len(all_warnings),
                'has_contact': bool(resume.email or resume.phone),
                'experience_count': len(resume.experience),
                'education_count': len(resume.education),
                'skills_count': len(resume.skills)
            }
        }

        if is_valid:
            logger.info(f"Resume validation passed with quality score: {quality_score:.2f}")
        else:
            logger.warning(f"Resume validation failed with {len(all_issues)} issues")

        return validation_result


class ExtractionErrorHandler:
    """
    Handles errors and implements retry logic for resume extraction.
    """

    def __init__(self, max_retries: int = 2):
        """
        Initialize error handler.

        Args:
            max_retries: Maximum number of retry attempts
        """
        self.max_retries = max_retries
        logger.info(f"ExtractionErrorHandler initialized (max_retries={max_retries})")

    def extract_with_retry(
        self,
        extractor,
        pdf_path: Path,
        use_fallback: bool = True
    ) -> Dict[str, Any]:
        """
        Extract resume with automatic retry on failure.

        Args:
            extractor: ResumeExtractor instance
            pdf_path: Path to PDF file
            use_fallback: Whether to use PDF parser fallback

        Returns:
            Extraction result with retry metadata
        """
        logger.info(f"Extracting with retry (max {self.max_retries} attempts): {pdf_path.name}")

        attempts = []

        for attempt in range(1, self.max_retries + 1):
            logger.info(f"Attempt {attempt}/{self.max_retries}...")

            result = extractor.extract_from_pdf(pdf_path, use_fallback=use_fallback)
            attempts.append(result)

            if result['success']:
                # Validate the extracted resume
                validation = ExtractionValidator.validate_resume(result['resume'])

                result['validation'] = validation
                result['attempts'] = attempt

                # Only accept if quality is reasonable
                if validation['quality_score'] >= 0.5:
                    logger.info(f"Extraction successful on attempt {attempt}")
                    return result
                else:
                    logger.warning(f"Low quality score ({validation['quality_score']}) on attempt {attempt}, retrying...")
            else:
                logger.warning(f"Attempt {attempt} failed: {result['error']}")

            # Don't retry on last attempt
            if attempt < self.max_retries:
                logger.info(f"Retrying...")

        # All attempts failed
        logger.error(f"All {self.max_retries} attempts failed")

        # Return last attempt result
        final_result = attempts[-1]
        final_result['attempts'] = self.max_retries
        final_result['all_attempts'] = attempts

        return final_result

    @staticmethod
    def format_error_report(result: Dict[str, Any]) -> str:
        """
        Format a human-readable error report.

        Args:
            result: Extraction result dict

        Returns:
            Formatted error report string
        """
        lines = []
        lines.append("="*70)
        lines.append("RESUME EXTRACTION REPORT")
        lines.append("="*70)

        if result['success']:
            lines.append("\nStatus: SUCCESS")

            validation = result.get('validation', {})
            quality_score = validation.get('quality_score', 0)
            lines.append(f"Quality Score: {quality_score:.1%}")

            if validation.get('issues'):
                lines.append(f"\nIssues ({len(validation['issues'])}):")
                for issue in validation['issues']:
                    lines.append(f"  - {issue}")

            if validation.get('warnings'):
                lines.append(f"\nWarnings ({len(validation['warnings'])}):")
                for warning in validation['warnings']:
                    lines.append(f"  - {warning}")

            summary = validation.get('summary', {})
            lines.append(f"\nExtracted Data:")
            lines.append(f"  Contact Info: {'Yes' if summary.get('has_contact') else 'No'}")
            lines.append(f"  Experience Entries: {summary.get('experience_count', 0)}")
            lines.append(f"  Education Entries: {summary.get('education_count', 0)}")
            lines.append(f"  Skills: {summary.get('skills_count', 0)}")

        else:
            lines.append("\nStatus: FAILED")
            lines.append(f"Error: {result.get('error', 'Unknown error')}")
            lines.append(f"Attempts: {result.get('attempts', 1)}")

        lines.append("="*70)

        return '\n'.join(lines)


def test_validation():
    """Test validation with example resumes."""
    from resume_schema import Resume, ExperienceEntry, EducationEntry

    print("="*70)
    print("TESTING EXTRACTION VALIDATOR - PHASE 2 STEP 2.5")
    print("="*70)

    # Test Case 1: Complete resume
    print("\nTest Case 1: Complete Resume")
    complete_resume = Resume(
        full_name="John Doe",
        email="john.doe@email.com",
        phone="+1 555-123-4567",
        summary="Software Engineer",
        experience=[
            ExperienceEntry(
                company="Tech Corp",
                position="Senior Engineer",
                duration="2020-Present",
                description="Led development team"
            )
        ],
        skills=["Python", "JavaScript", "AWS", "Docker", "React"],
        education=[
            EducationEntry(
                institution="University of Tech",
                degree="BS Computer Science",
                field="Computer Science"
            )
        ]
    )

    validation = ExtractionValidator.validate_resume(complete_resume)
    print(f"Valid: {validation['valid']}")
    print(f"Quality Score: {validation['quality_score']:.1%}")
    print(f"Issues: {len(validation['issues'])}")
    print(f"Warnings: {len(validation['warnings'])}")

    # Test Case 2: Incomplete resume
    print("\nTest Case 2: Incomplete Resume (no contact, few skills)")
    incomplete_resume = Resume(
        full_name="Jane Smith",
        skills=["Communication"],  # Only 1 generic skill
        education=[]
    )

    validation = ExtractionValidator.validate_resume(incomplete_resume)
    print(f"Valid: {validation['valid']}")
    print(f"Quality Score: {validation['quality_score']:.1%}")
    print(f"Issues: {len(validation['issues'])}")
    for issue in validation['issues']:
        print(f"  - {issue}")
    print(f"Warnings: {len(validation['warnings'])}")
    for warning in validation['warnings'][:3]:  # Show first 3
        print(f"  - {warning}")

    print("\n" + "="*70)
    print("Validation test complete!")


if __name__ == "__main__":
    test_validation()
"""
Resume Data Schema with Pydantic Models

Defines structured data models for resume information extraction.
Ensures type safety and validation for parsed resume data.

Author: Mert Alp Aydin
Date: 2025-11-18
Phase: Phase 2 - Step 2.3
"""

from typing import List, Optional
from pydantic import BaseModel, Field, field_validator, ConfigDict
import re


class ExperienceEntry(BaseModel):
    """
    Model for a single work experience entry.
    """
    company: str = Field(..., description="Company or organization name")
    position: str = Field(..., description="Job title or role")
    duration: str = Field(..., description="Time period (e.g., 'Jan 2020 - Present', '2018-2020')")
    description: str = Field(..., description="Key responsibilities and achievements")
    location: Optional[str] = Field(None, description="Job location (city, country)")

    @field_validator('company', 'position')
    @classmethod
    def not_empty(cls, v: str) -> str:
        """Ensure critical fields are not empty."""
        if not v or not v.strip():
            raise ValueError("Field cannot be empty")
        return v.strip()


class EducationEntry(BaseModel):
    """
    Model for a single education entry.
    """
    institution: str = Field(..., description="School, university, or institution name")
    degree: str = Field(..., description="Degree type (e.g., 'Bachelor of Science', 'MSc')")
    field: Optional[str] = Field(None, description="Field of study or major")
    graduation_year: Optional[str] = Field(None, description="Year of graduation or date range")
    location: Optional[str] = Field(None, description="Institution location")
    gpa: Optional[str] = Field(None, description="GPA or grade (if mentioned)")

    @field_validator('institution', 'degree')
    @classmethod
    def not_empty(cls, v: str) -> str:
        """Ensure critical fields are not empty."""
        if not v or not v.strip():
            raise ValueError("Field cannot be empty")
        return v.strip()


class Resume(BaseModel):
    """
    Complete resume data model.

    This is the main model representing all extracted information from a resume.
    """
    full_name: str = Field(..., description="Candidate's full name")
    email: Optional[str] = Field(None, description="Primary email address")
    phone: Optional[str] = Field(None, description="Primary phone number")
    location: Optional[str] = Field(None, description="Current location (city, state/country)")
    linkedin: Optional[str] = Field(None, description="LinkedIn profile URL")
    github: Optional[str] = Field(None, description="GitHub profile URL")
    website: Optional[str] = Field(None, description="Personal website or portfolio URL")

    summary: Optional[str] = Field(
        None,
        description="Professional summary or objective statement"
    )

    experience: List[ExperienceEntry] = Field(
        default_factory=list,
        description="List of work experience entries"
    )

    skills: List[str] = Field(
        default_factory=list,
        description="List of skills and technologies"
    )

    education: List[EducationEntry] = Field(
        default_factory=list,
        description="List of education entries"
    )

    certifications: List[str] = Field(
        default_factory=list,
        description="List of certifications and professional qualifications"
    )

    languages: List[str] = Field(
        default_factory=list,
        description="List of spoken/written languages with proficiency levels"
    )

    projects: List[str] = Field(
        default_factory=list,
        description="List of notable projects or publications"
    )

    additional_sections: dict = Field(
        default_factory=dict,
        description="Additional sections as key-value pairs for content not fitting standard categories (e.g., {'Awards': ['Best Paper 2023'], 'Volunteer': ['Red Cross 2020-2022'], 'Publications': [...]})"
    )

    @field_validator('full_name')
    @classmethod
    def validate_name(cls, v: str) -> str:
        """Validate and clean the full name."""
        if not v or not v.strip():
            raise ValueError("Full name is required")

        # Clean up extra whitespace
        name = ' '.join(v.split())

        # Basic validation - should have at least 2 parts (first and last name)
        if len(name.split()) < 2:
            # Single name is acceptable in some cultures, just warn
            pass

        return name

    @field_validator('email')
    @classmethod
    def validate_email(cls, v: Optional[str]) -> Optional[str]:
        """Validate email format."""
        if v is None:
            return None

        # Basic email validation
        email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        if not re.match(email_pattern, v):
            # Don't fail, just return as-is (LLM might extract partial emails)
            pass

        return v.strip()

    @field_validator('phone')
    @classmethod
    def validate_phone(cls, v: Optional[str]) -> Optional[str]:
        """Validate and clean phone number."""
        if v is None:
            return None

        # Remove common formatting
        cleaned = re.sub(r'[^\d\+\(\)\-\s]', '', v)
        return cleaned.strip() if cleaned else None

    @field_validator('skills')
    @classmethod
    def validate_skills(cls, v: List[str]) -> List[str]:
        """Clean and deduplicate skills list."""
        if not v:
            return []

        # Remove empty strings, strip whitespace, deduplicate
        skills = [s.strip() for s in v if s and s.strip()]
        # Remove duplicates while preserving order
        seen = set()
        unique_skills = []
        for skill in skills:
            skill_lower = skill.lower()
            if skill_lower not in seen:
                seen.add(skill_lower)
                unique_skills.append(skill)

        return unique_skills

    @field_validator('certifications', 'languages', 'projects')
    @classmethod
    def validate_list_fields(cls, v: List[str]) -> List[str]:
        """Clean list fields."""
        if not v:
            return []

        # Remove empty strings and strip whitespace
        return [item.strip() for item in v if item and item.strip()]

    def model_post_init(self, __context) -> None:
        """Post-initialization validation and cleanup."""
        # Ensure at least name and one contact method
        if not self.email and not self.phone:
            # This is a warning, not an error - some resumes may not have contact info
            pass

    # Pydantic V2 configuration
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "full_name": "Maria Elena Rodriguez-Smith",
                "email": "maria.rodriguez@company.co.uk",
                "phone": "+44 20 7123 4567",
                "location": "London, UK",
                "linkedin": "linkedin.com/in/mariarodriguez",
                "summary": "Senior Product Manager with 7+ years building enterprise SaaS platforms",
                "experience": [
                    {
                        "company": "TechVision Inc.",
                        "position": "Senior Product Manager",
                        "duration": "March 2021 - Present",
                        "description": "Led cross-functional team building B2B analytics platform",
                        "location": "London, UK"
                    }
                ],
                "skills": ["Python", "JavaScript", "React", "AWS", "Product Management"],
                "education": [
                    {
                        "institution": "Technical University of Berlin",
                        "degree": "MSc Computer Science",
                        "field": "Machine Learning",
                        "graduation_year": "2016"
                    }
                ],
                "certifications": ["AWS Solutions Architect Professional"],
                "languages": ["Spanish (native)", "English (fluent)", "German (intermediate)"]
            }
        }
    )


def validate_resume(resume_data: dict) -> Resume:
    """
    Validate and parse resume data into Resume model.

    Args:
        resume_data: Dictionary containing resume information

    Returns:
        Validated Resume model instance

    Raises:
        ValidationError: If data doesn't match schema or validation fails
    """
    return Resume(**resume_data)


# For LangChain compatibility
def get_resume_schema() -> dict:
    """
    Get the JSON schema for Resume model.

    Returns:
        JSON schema dictionary for use with LangChain output parsers
    """
    return Resume.model_json_schema()


def get_resume_schema_description() -> str:
    """
    Get a formatted description of the resume schema for LLM prompts.

    Returns:
        Formatted string describing the expected resume structure
    """
    return """
{
    "full_name": "string (required - candidate's full name)",
    "email": "string (optional - primary email address)",
    "phone": "string (optional - primary phone number)",
    "location": "string (optional - current location)",
    "linkedin": "string (optional - LinkedIn profile URL)",
    "github": "string (optional - GitHub profile URL)",
    "website": "string (optional - personal website URL)",
    "summary": "string (optional - professional summary)",
    "experience": [
        {
            "company": "string (required)",
            "position": "string (required)",
            "duration": "string (required)",
            "description": "string (required)",
            "location": "string (optional)"
        }
    ],
    "skills": ["string (list of skills and technologies)"],
    "education": [
        {
            "institution": "string (required)",
            "degree": "string (required)",
            "field": "string (optional)",
            "graduation_year": "string (optional)",
            "location": "string (optional)",
            "gpa": "string (optional)"
        }
    ],
    "certifications": ["string (list of certifications)"],
    "languages": ["string (list of languages with proficiency)"],
    "projects": ["string (list of notable projects)"],
    "additional_sections": {
        "Awards": ["string (awards and honors)"],
        "Volunteer": ["string (volunteer work)"],
        "Publications": ["string (publications)"],
        "Hobbies": ["string (hobbies and interests)"],
        "(any other section name)": ["string (content)"]
    }
}
"""


if __name__ == "__main__":
    # Test the schema with example data
    example_resume = {
        "full_name": "John Doe",
        "email": "john.doe@email.com",
        "phone": "(555) 123-4567",
        "summary": "Senior Software Engineer with 5 years of experience",
        "experience": [
            {
                "company": "Tech Corp",
                "position": "Senior Software Engineer",
                "duration": "2020-Present",
                "description": "Led development of microservices architecture"
            }
        ],
        "skills": ["Python", "JavaScript", "AWS", "Docker"],
        "education": [
            {
                "institution": "University of Technology",
                "degree": "Bachelor of Science",
                "field": "Computer Science"
            }
        ],
        "certifications": ["AWS Certified Solutions Architect"]
    }

    try:
        resume = Resume(**example_resume)
        print("[OK] Schema validation successful!")
        print(f"\nParsed Resume:")
        print(f"  Name: {resume.full_name}")
        print(f"  Email: {resume.email}")
        print(f"  Skills: {', '.join(resume.skills)}")
        print(f"  Experience entries: {len(resume.experience)}")
        print(f"  Education entries: {len(resume.education)}")

        # Test JSON serialization
        print(f"\n[OK] JSON serialization works:")
        print(resume.model_dump_json(indent=2)[:] + "...")

    except Exception as e:
        print(f"[FAIL] Validation failed: {e}")
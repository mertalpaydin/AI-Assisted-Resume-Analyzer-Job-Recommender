"""
LLM-Powered Resume Extractor

Extracts structured information from resume PDFs using gemma3:4b via Ollama.
Integrates PDF parsing, LLM processing, and Pydantic validation.

Author: Mert Alp Aydin
Date: 2025-11-18
Phase: Phase 2 - Step 2.4
"""

from pathlib import Path
from typing import Dict, Any, Optional
import json
import time
import re

# LangChain imports
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

# Local imports
from pdf_parser import PDFResumeParser
from resume_schema import Resume, get_resume_schema_description
from logging_utils import setup_logger

# Initialize logger
logger = setup_logger(__name__)


def repair_json(json_text: str) -> str:
    """
    Attempt to repair common JSON formatting issues from LLM responses.

    Args:
        json_text: Potentially malformed JSON string

    Returns:
        Repaired JSON string
    """
    # Remove trailing commas before closing braces/brackets
    json_text = re.sub(r',(\s*[}\]])', r'\1', json_text)

    # Fix missing commas between array elements (simple heuristic)
    json_text = re.sub(r'"\s*\n\s*"', '",\n"', json_text)

    # Fix missing commas between object properties
    json_text = re.sub(r'"\s*\n\s*"([^"]+)":', '",\n"\1":', json_text)

    return json_text


class ResumeExtractor:
    """
    Extracts structured resume data from PDFs using gemma3:4b LLM.

    Combines PDF parsing, LLM-based extraction, and Pydantic validation
    into a single end-to-end pipeline.
    """

    def __init__(
        self,
        model_name: str = "gemma3:4b",
        temperature: float = 0.1,  # Low temperature for consistent extraction
        pdf_loader: str = "pdfplumber"
    ):
        """
        Initialize the resume extractor.

        Args:
            model_name: Ollama model to use (default: gemma3:4b - best performer)
            temperature: LLM temperature (lower = more consistent)
            pdf_loader: PDF parsing library to use
        """
        self.model_name = model_name
        self.temperature = temperature

        logger.info(f"Initializing ResumeExtractor with model: {model_name}")

        # Initialize PDF parser
        self.pdf_parser = PDFResumeParser(default_loader=pdf_loader)

        # Initialize LLM
        self.llm = ChatOllama(
            model=model_name,
            temperature=temperature
        )

        # Initialize Pydantic output parser
        self.output_parser = PydanticOutputParser(pydantic_object=Resume)

        # Create prompt template
        self.prompt_template = self._create_prompt_template()

        logger.info("ResumeExtractor initialized successfully")

    def _clean_resume_dict(self, resume_dict: dict) -> dict:
        """
        Clean resume dictionary by filtering out None values from list fields.

        Args:
            resume_dict: Raw resume dictionary from LLM

        Returns:
            Cleaned resume dictionary
        """
        # List fields that should not contain None values
        list_fields = ['skills', 'certifications', 'languages', 'projects', 'awards', 'publications']

        for field in list_fields:
            if field in resume_dict and isinstance(resume_dict[field], list):
                # Filter out None values and empty strings
                resume_dict[field] = [
                    item for item in resume_dict[field]
                    if item is not None and (not isinstance(item, str) or item.strip())
                ]

        return resume_dict

    def _create_prompt_template(self) -> PromptTemplate:
        """
        Create the prompt template for resume extraction.

        Returns:
            PromptTemplate configured for structured resume extraction
        """
        template = """You are an expert at extracting structured information from resumes.

Extract the following information from the resume text below and return it as valid JSON.

IMPORTANT INSTRUCTIONS:
- Extract ONLY information explicitly stated in the resume
- DO NOT make up or infer information
- Use null for missing fields
- For multiple emails/phones, choose the primary/first one
- List ALL skills mentioned (technical skills, tools, frameworks, languages)
- For experience: create separate entries for each role, even if same company
- Be precise with dates and durations

EXPECTED JSON STRUCTURE:
{schema_description}

RESUME TEXT:
{resume_text}

Return ONLY valid JSON, no other text or explanation.

JSON OUTPUT:
"""

        return PromptTemplate(
            template=template,
            input_variables=["resume_text", "schema_description"]
        )

    def _create_refinement_prompt_template(self) -> PromptTemplate:
        """
        Create the prompt template for refining resume extraction with second parser.

        Returns:
            PromptTemplate configured for refinement pass
        """
        template = """You are an expert at extracting structured information from resumes.

You previously extracted resume data from one PDF parser. Now you have text from a SECOND parser that may contain additional or better-formatted information.

Your task: REFINE and IMPROVE the previous extraction by:
1. Filling in any missing fields from the new text
2. Correcting any errors if the new text shows better information
3. Adding any additional information found in the new text
4. Keeping all good data from the previous extraction
5. Preferring more complete and coherent information

CRITICAL SCHEMA REQUIREMENTS - YOU MUST FOLLOW THESE EXACTLY:
- Use "full_name" NOT "name"
- Use "position" NOT "title" or "role" in experience entries
- Use "duration" NOT "dates" or "period" in experience entries
- Use "description" NOT "responsibilities" in experience entries
- Use "institution" NOT "school" or "university" in education entries
- Use "skills" as a flat list of strings, NOT objects with names
- Use "projects" as a flat list of strings, NOT objects
- Use "certifications" as a flat list of strings, NOT objects
- ALL field names must EXACTLY match the schema below

IMPORTANT INSTRUCTIONS:
- Extract ONLY information explicitly stated in the resume
- DO NOT remove correctly extracted information unless the new text clearly contradicts it
- If both sources have the same field, prefer the more complete/detailed version
- List ALL skills from both sources (deduplicate)
- Combine experience descriptions if both parsers captured different details
- Maintain the EXACT schema structure from the previous extraction
- DO NOT change any field names - they must match the schema exactly

EXPECTED JSON STRUCTURE:
{schema_description}

PREVIOUS EXTRACTION (from first parser):
{previous_json}

NEW RESUME TEXT (from second parser):
{resume_text}

Return the REFINED JSON output with EXACT field names matching the schema, incorporating the best information from both sources.

JSON OUTPUT:
"""

        return PromptTemplate(
            template=template,
            input_variables=["resume_text", "schema_description", "previous_json"]
        )

    def extract_from_pdf(
        self,
        pdf_path: Path,
        use_dual_parser: bool = True
    ) -> Dict[str, Any]:
        """
        Extract structured resume data from a PDF file using dual-parser pipeline.

        Args:
            pdf_path: Path to PDF resume file
            use_dual_parser: Whether to use dual-parser sequential pipeline (default: True)
                            If False, only uses PDFPlumber

        Returns:
            Dict containing:
                - 'resume': Resume object (Pydantic model)
                - 'raw_text': Original extracted text (from both parsers if dual mode)
                - 'parse_metadata': PDF parsing metadata
                - 'extraction_metadata': LLM extraction metadata
                - 'success': Whether extraction succeeded
                - 'error': Error message if failed
        """
        logger.info(f"Extracting resume from PDF: {pdf_path.name}")
        logger.info(f"Dual-parser mode: {'ENABLED' if use_dual_parser else 'DISABLED'}")
        start_time = time.time()

        result = {
            'resume': None,
            'raw_text': None,
            'raw_text_pypdf': None,
            'parse_metadata': None,
            'extraction_metadata': None,
            'success': False,
            'error': None
        }

        try:
            # Step 1: Parse PDF with PDFPlumber
            logger.info("Step 1: Parsing PDF with PDFPlumber...")
            pdfplumber_result = self.pdf_parser.parse(pdf_path, loader='pdfplumber')

            if not pdfplumber_result['success']:
                result['error'] = f"PDFPlumber parsing failed: {pdfplumber_result['error']}"
                logger.error(result['error'])
                return result

            pdfplumber_text = pdfplumber_result['text']
            result['raw_text'] = pdfplumber_text
            result['parse_metadata'] = {
                'pdfplumber': pdfplumber_result['metadata']
            }

            logger.info(f"PDFPlumber parsed: {len(pdfplumber_text)} characters")

            # Step 2: Extract structured data using LLM (first pass)
            logger.info("Step 2: Extracting structured data with LLM (first pass - PDFPlumber)...")
            first_extraction = self.extract_from_text(pdfplumber_text)

            if not first_extraction['success']:
                result['error'] = first_extraction['error']
                logger.error(result['error'])
                return result

            first_resume = first_extraction['resume']
            logger.info(f"First pass extraction successful")

            # If dual-parser mode is disabled, return after first pass
            if not use_dual_parser:
                result['resume'] = first_resume
                result['extraction_metadata'] = first_extraction['metadata']
                result['success'] = True

                end_time = time.time()
                total_time = end_time - start_time
                logger.info(f"Resume extraction successful in {total_time:.2f}s (single-parser mode)")
                result['extraction_metadata']['total_time_seconds'] = round(total_time, 2)
                result['extraction_metadata']['pipeline_mode'] = 'single-parser'
                return result

            # Step 3: Parse PDF with PyPDF (second parser)
            logger.info("Step 3: Parsing PDF with PyPDF (second parser)...")
            pypdf_result = self.pdf_parser.parse(pdf_path, loader='pypdf')

            if not pypdf_result['success']:
                logger.warning(f"PyPDF parsing failed: {pypdf_result['error']}")
                logger.warning("Falling back to single-parser result")
                # Fallback to first pass result
                result['resume'] = first_resume
                result['extraction_metadata'] = first_extraction['metadata']
                result['extraction_metadata']['pipeline_mode'] = 'single-parser (pypdf failed)'
                result['success'] = True

                end_time = time.time()
                total_time = end_time - start_time
                logger.info(f"Resume extraction successful in {total_time:.2f}s (fallback to single-parser)")
                result['extraction_metadata']['total_time_seconds'] = round(total_time, 2)
                return result

            pypdf_text = pypdf_result['text']
            result['raw_text_pypdf'] = pypdf_text
            result['parse_metadata']['pypdf'] = pypdf_result['metadata']

            logger.info(f"PyPDF parsed: {len(pypdf_text)} characters")

            # Step 4: Refine extraction with second parser (second pass)
            logger.info("Step 4: Refining extraction with LLM (second pass - PyPDF)...")
            refinement_result = self.refine_extraction(
                new_text=pypdf_text,
                previous_resume=first_resume
            )

            if not refinement_result['success']:
                logger.warning(f"Refinement failed: {refinement_result['error']}")
                logger.warning("Falling back to first pass result")
                # Fallback to first pass result
                result['resume'] = first_resume
                result['extraction_metadata'] = first_extraction['metadata']
                result['extraction_metadata']['pipeline_mode'] = 'single-parser (refinement failed)'
                result['success'] = True

                end_time = time.time()
                total_time = end_time - start_time
                logger.info(f"Resume extraction successful in {total_time:.2f}s (fallback to single-parser)")
                result['extraction_metadata']['total_time_seconds'] = round(total_time, 2)
                return result

            # Success with dual-parser pipeline
            refined_resume = refinement_result['resume']
            result['resume'] = refined_resume
            result['extraction_metadata'] = {
                'first_pass': first_extraction['metadata'],
                'second_pass': refinement_result['metadata'],
                'pipeline_mode': 'dual-parser'
            }
            result['success'] = True

            end_time = time.time()
            total_time = end_time - start_time

            logger.info(f"Resume extraction successful in {total_time:.2f}s (dual-parser mode)")
            result['extraction_metadata']['total_time_seconds'] = round(total_time, 2)

            return result

        except Exception as e:
            logger.error(f"Resume extraction failed: {str(e)}")
            result['error'] = str(e)
            return result

    def refine_extraction(
        self,
        new_text: str,
        previous_resume: Resume
    ) -> Dict[str, Any]:
        """
        Refine resume extraction using second parser's text and previous extraction.

        Args:
            new_text: Text from second PDF parser
            previous_resume: Resume object from first extraction pass

        Returns:
            Dict containing refined resume and metadata
        """
        start_time = time.time()

        try:
            # Convert previous resume to JSON for prompt
            previous_json = previous_resume.model_dump_json(indent=2)

            # Prepare refinement prompt
            schema_description = get_resume_schema_description()
            refinement_template = self._create_refinement_prompt_template()

            prompt = refinement_template.format(
                resume_text=new_text,
                schema_description=schema_description,
                previous_json=previous_json
            )

            logger.info("Sending refinement request to LLM...")

            # Call LLM
            response = self.llm.invoke(prompt)

            end_time = time.time()
            llm_time = end_time - start_time

            logger.info(f"LLM refinement response received in {llm_time:.2f}s")

            # Parse JSON response
            try:
                # Extract JSON from response
                response_text = response.content.strip()

                if '{' in response_text and '}' in response_text:
                    json_start = response_text.find('{')
                    json_end = response_text.rfind('}') + 1
                    json_text = response_text[json_start:json_end]

                    # Try to repair common JSON issues
                    json_text = repair_json(json_text)

                    # Parse JSON
                    resume_dict = json.loads(json_text)

                    # Clean None values from list fields
                    resume_dict = self._clean_resume_dict(resume_dict)

                    # Validate with Pydantic
                    refined_resume = Resume(**resume_dict)

                    logger.info("Refined resume successfully parsed and validated")

                    return {
                        'resume': refined_resume,
                        'success': True,
                        'error': None,
                        'metadata': {
                            'model': self.model_name,
                            'llm_time_seconds': round(llm_time, 2),
                            'response_length': len(response.content),
                            'skills_extracted': len(refined_resume.skills),
                            'experience_entries': len(refined_resume.experience),
                            'education_entries': len(refined_resume.education)
                        }
                    }
                else:
                    error_msg = "No JSON found in refinement LLM response"
                    logger.error(error_msg)
                    return {
                        'resume': None,
                        'success': False,
                        'error': error_msg,
                        'metadata': {'llm_response': response.content[:200]}
                    }

            except json.JSONDecodeError as e:
                error_msg = f"Failed to parse refinement JSON: {str(e)}"
                logger.error(error_msg)
                return {
                    'resume': None,
                    'success': False,
                    'error': error_msg,
                    'metadata': {'llm_response': response.content[:200]}
                }

            except Exception as e:
                error_msg = f"Pydantic validation failed for refinement: {str(e)}"
                logger.error(error_msg)
                return {
                    'resume': None,
                    'success': False,
                    'error': error_msg,
                    'metadata': {'parsed_json': str(resume_dict)[:200]}
                }

        except Exception as e:
            logger.error(f"LLM refinement failed: {str(e)}")
            return {
                'resume': None,
                'success': False,
                'error': str(e),
                'metadata': {}
            }

    def extract_from_text(self, resume_text: str) -> Dict[str, Any]:
        """
        Extract structured resume data from raw text using LLM.

        Args:
            resume_text: Raw resume text

        Returns:
            Dict containing extracted resume and metadata
        """
        start_time = time.time()

        try:
            # Prepare prompt
            schema_description = get_resume_schema_description()
            prompt = self.prompt_template.format(
                resume_text=resume_text,
                schema_description=schema_description
            )

            logger.info("Sending to LLM...")

            # Call LLM
            response = self.llm.invoke(prompt)

            end_time = time.time()
            llm_time = end_time - start_time

            logger.info(f"LLM response received in {llm_time:.2f}s")

            # Parse JSON response
            try:
                # Extract JSON from response (handle cases where LLM adds extra text)
                response_text = response.content.strip()

                if '{' in response_text and '}' in response_text:
                    json_start = response_text.find('{')
                    json_end = response_text.rfind('}') + 1
                    json_text = response_text[json_start:json_end]

                    # Try to repair common JSON issues
                    json_text = repair_json(json_text)

                    # Parse JSON
                    resume_dict = json.loads(json_text)

                    # Clean None values from list fields
                    resume_dict = self._clean_resume_dict(resume_dict)

                    # Validate with Pydantic
                    resume = Resume(**resume_dict)

                    logger.info("Resume successfully parsed and validated")

                    return {
                        'resume': resume,
                        'success': True,
                        'error': None,
                        'metadata': {
                            'model': self.model_name,
                            'llm_time_seconds': round(llm_time, 2),
                            'response_length': len(response.content),
                            'skills_extracted': len(resume.skills),
                            'experience_entries': len(resume.experience),
                            'education_entries': len(resume.education)
                        }
                    }
                else:
                    error_msg = "No JSON found in LLM response"
                    logger.error(error_msg)
                    return {
                        'resume': None,
                        'success': False,
                        'error': error_msg,
                        'metadata': {'llm_response': response.content[:200]}
                    }

            except json.JSONDecodeError as e:
                error_msg = f"Failed to parse JSON: {str(e)}"
                logger.error(error_msg)
                return {
                    'resume': None,
                    'success': False,
                    'error': error_msg,
                    'metadata': {'llm_response': response.content[:200]}
                }

            except Exception as e:
                error_msg = f"Pydantic validation failed: {str(e)}"
                logger.error(error_msg)
                return {
                    'resume': None,
                    'success': False,
                    'error': error_msg,
                    'metadata': {'parsed_json': str(resume_dict)[:200]}
                }

        except Exception as e:
            logger.error(f"LLM extraction failed: {str(e)}")
            return {
                'resume': None,
                'success': False,
                'error': str(e),
                'metadata': {}
            }

    def extract_batch(
        self,
        pdf_paths: list[Path],
        use_dual_parser: bool = True
    ) -> list[Dict[str, Any]]:
        """
        Extract structured data from multiple PDF resumes.

        Args:
            pdf_paths: List of PDF file paths
            use_dual_parser: Whether to use dual-parser sequential pipeline

        Returns:
            List of extraction results
        """
        logger.info(f"Starting batch extraction of {len(pdf_paths)} resumes")
        logger.info(f"Dual-parser mode: {'ENABLED' if use_dual_parser else 'DISABLED'}")

        results = []
        for i, pdf_path in enumerate(pdf_paths, 1):
            logger.info(f"Processing resume {i}/{len(pdf_paths)}: {pdf_path.name}")

            result = self.extract_from_pdf(pdf_path, use_dual_parser=use_dual_parser)
            results.append(result)

            # Add small delay between requests to avoid overwhelming Ollama
            if i < len(pdf_paths):
                time.sleep(0.5)

        successful = sum(1 for r in results if r['success'])
        logger.info(f"Batch extraction complete: {successful}/{len(pdf_paths)} successful")

        return results


def test_resume_extractor():
    """Test the resume extractor with sample PDFs."""
    logger.info("="*70)
    logger.info("RESUME EXTRACTOR TEST - PHASE 2 STEP 2.4")
    logger.info("="*70)

    # Initialize extractor with gemma3:4b (our selected model)
    extractor = ResumeExtractor(model_name="gemma3:4b", temperature=0.1)

    # Get sample PDFs
    samples_dir = Path(__file__).parent.parent / "data" / "cv_samples"
    pdf_files = list(samples_dir.glob("*.pdf"))

    if not pdf_files:
        logger.error("No sample PDF files found!")
        print("[FAIL] No sample PDFs found in data/cv_samples/ai generated/")
        return

    logger.info(f"Found {len(pdf_files)} sample PDFs")
    print(f"\nTesting with ALL {len(pdf_files)} sample resumes...")

    # Test ALL resumes
    for pdf_file in pdf_files:
        print(f"\n{'='*70}")
        print(f"Extracting: {pdf_file.name}")
        print("="*70)

        result = extractor.extract_from_pdf(pdf_file)

        if result['success']:
            resume = result['resume']
            metadata = result['extraction_metadata']

            print(f"\n[OK] Extraction successful!")
            print(f"  Pipeline Mode: {metadata.get('pipeline_mode', 'N/A')}")
            print(f"\n  Name: {resume.full_name}")
            print(f"  Email: {resume.email}")
            print(f"  Phone: {resume.phone}")
            print(f"  Skills ({len(resume.skills)}): {', '.join(resume.skills[:5])}{'...' if len(resume.skills) > 5 else ''}")
            print(f"  Experience: {len(resume.experience)} entries")
            print(f"  Education: {len(resume.education)} entries")

            if resume.certifications:
                print(f"  Certifications: {len(resume.certifications)}")

            print(f"\n  Total Extraction Time: {metadata['total_time_seconds']}s")

            # Handle different metadata structures (single vs dual parser)
            if 'pipeline_mode' in metadata and metadata['pipeline_mode'] == 'dual-parser':
                first_pass_time = metadata['first_pass']['llm_time_seconds']
                second_pass_time = metadata['second_pass']['llm_time_seconds']
                print(f"  First Pass LLM Time: {first_pass_time}s")
                print(f"  Second Pass LLM Time: {second_pass_time}s")
                print(f"  Combined LLM Time: {first_pass_time + second_pass_time}s")
            else:
                print(f"  LLM Time: {metadata.get('llm_time_seconds', 'N/A')}s")

            print()
            for key, value in resume:
                print(f"{key}: {value}")
        else:
            print(f"\n[FAIL] Extraction failed")
            print(f"  Error: {result['error']}")


    print("\n" + "="*70)
    logger.info("Resume extractor test complete!")


if __name__ == "__main__":
    test_resume_extractor()
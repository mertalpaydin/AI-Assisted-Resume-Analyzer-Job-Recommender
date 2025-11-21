"""
Job Chunker Module - Phase 4.1

This module provides intelligent chunking of job postings for better embedding
and retrieval. Jobs are split by semantic sections (title, description, requirements).

Author: Mert Alp Aydin
Date: 2025-11-21
"""

import logging
import json
import re
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field, asdict
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class JobChunk:
    """Represents a single chunk from a job posting."""
    chunk_id: str
    job_id: str
    chunk_num: int
    section: str  # 'title', 'description', 'requirements', 'full'
    text: str
    importance: float  # Higher = more important for matching
    metadata: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class JobPosting:
    """Complete job posting data."""
    job_id: str
    title: str
    company: str
    description: str
    location: Optional[str] = None
    url: Optional[str] = None
    date_posted: Optional[str] = None
    raw_data: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_raw(cls, raw: Dict) -> 'JobPosting':
        """Create JobPosting from raw data."""
        job_id = raw.get('_id', {}).get('$oid', '') if isinstance(raw.get('_id'), dict) else str(raw.get('_id', ''))
        position = raw.get('position', {})

        # Extract company name from nested structure
        org_company = raw.get('orgCompany', {})
        if isinstance(org_company, dict):
            company_name = org_company.get('name', '') or org_company.get('nameOrg', '')
        else:
            company_name = str(org_company) if org_company else ''

        if not company_name:
            company_name = raw.get('name', '')

        return cls(
            job_id=job_id,
            title=position.get('name', '') if isinstance(position, dict) else str(position),
            company=company_name,
            description=raw.get('text', ''),
            location=raw.get('orgAddress', ''),
            url=raw.get('url', ''),
            date_posted=raw.get('dateCreated', ''),
            raw_data=raw
        )


class JobChunker:
    """
    Intelligent job chunking for better embedding and retrieval.

    Splits job postings into semantic sections with importance weights.
    """

    # Section importance weights (title > requirements > description)
    SECTION_WEIGHTS = {
        'title': 1.0,
        'requirements': 0.9,
        'skills': 0.85,
        'qualifications': 0.8,
        'responsibilities': 0.7,
        'description': 0.6,
        'other': 0.5,  # Unmapped content
        'full': 0.5,
        'about': 0.4,
        'benefits': 0.3
    }

    # Patterns to identify sections
    SECTION_PATTERNS = {
        'requirements': r'(?:requirements?|qualifications?|what you.?ll need|must have|required)',
        'skills': r'(?:skills?|technical skills?|expertise)',
        'responsibilities': r'(?:responsibilities?|duties|what you.?ll do|role)',
        'qualifications': r'(?:qualifications?|education|experience required)',
        'benefits': r'(?:benefits?|perks|we offer|compensation)',
        'about': r'(?:about us|company|who we are|our team)'
    }

    def __init__(
        self,
        max_chunk_size: int = 1000,
        min_chunk_size: int = 100,
        overlap: int = 50
    ):
        """
        Initialize the chunker.

        Args:
            max_chunk_size: Maximum characters per chunk
            min_chunk_size: Minimum characters per chunk
            overlap: Character overlap between chunks for context
        """
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size
        self.overlap = overlap
        logger.info(f"JobChunker initialized: max={max_chunk_size}, min={min_chunk_size}, overlap={overlap}")

    def chunk_job(self, job: JobPosting) -> List[JobChunk]:
        """
        Chunk a single job posting into semantic sections.

        Args:
            job: JobPosting object

        Returns:
            List of JobChunk objects
        """
        chunks = []
        chunk_num = 0

        # Chunk 1: Title (always separate, high importance)
        if job.title:
            title_text = f"{job.title}"
            if job.company:
                title_text += f" at {job.company}"
            if job.location:
                title_text += f" - {job.location}"

            chunks.append(JobChunk(
                chunk_id=f"{job.job_id}_c{chunk_num}",
                job_id=job.job_id,
                chunk_num=chunk_num,
                section='title',
                text=title_text,
                importance=self.SECTION_WEIGHTS['title'],
                metadata={'company': job.company, 'location': job.location}
            ))
            chunk_num += 1

        # Parse description into sections
        if job.description:
            sections = self._extract_sections(job.description)

            for section_name, section_text in sections.items():
                if len(section_text) < self.min_chunk_size:
                    continue

                # Split large sections into smaller chunks
                section_chunks = self._split_text(section_text)
                importance = self.SECTION_WEIGHTS.get(section_name, 0.5)

                for i, chunk_text in enumerate(section_chunks):
                    chunks.append(JobChunk(
                        chunk_id=f"{job.job_id}_c{chunk_num}",
                        job_id=job.job_id,
                        chunk_num=chunk_num,
                        section=section_name if len(section_chunks) == 1 else f"{section_name}_{i}",
                        text=chunk_text,
                        importance=importance,
                        metadata={'section_index': i, 'total_sections': len(section_chunks)}
                    ))
                    chunk_num += 1

        # If no chunks were created from sections, create a full chunk
        if len(chunks) <= 1 and job.description:
            full_chunks = self._split_text(job.description)
            for i, chunk_text in enumerate(full_chunks):
                chunks.append(JobChunk(
                    chunk_id=f"{job.job_id}_c{chunk_num}",
                    job_id=job.job_id,
                    chunk_num=chunk_num,
                    section='full',
                    text=chunk_text,
                    importance=self.SECTION_WEIGHTS['full'],
                    metadata={'chunk_index': i}
                ))
                chunk_num += 1

        logger.debug(f"Job {job.job_id}: created {len(chunks)} chunks")
        return chunks

    def _extract_sections(self, text: str) -> Dict[str, str]:
        """
        Extract semantic sections from job description.

        Args:
            text: Full job description text

        Returns:
            Dict mapping section names to their content
        """
        sections = {}
        text_lower = text.lower()

        # Find all section boundaries
        boundaries = []
        for section_name, pattern in self.SECTION_PATTERNS.items():
            for match in re.finditer(pattern, text_lower):
                boundaries.append((match.start(), section_name))

        # Sort by position
        boundaries.sort(key=lambda x: x[0])

        if not boundaries:
            # No sections found, return as description
            return {'description': text}

        # Add start of text as description if first section doesn't start at beginning
        if boundaries[0][0] > self.min_chunk_size:
            sections['description'] = text[:boundaries[0][0]].strip()

        # Extract each section
        for i, (start, section_name) in enumerate(boundaries):
            if i < len(boundaries) - 1:
                end = boundaries[i + 1][0]
            else:
                end = len(text)

            section_text = text[start:end].strip()

            # Remove section header from text
            lines = section_text.split('\n', 1)
            if len(lines) > 1:
                section_text = lines[1].strip()

            if section_text and len(section_text) >= self.min_chunk_size:
                # Merge with existing section if same name
                if section_name in sections:
                    sections[section_name] += '\n' + section_text
                else:
                    sections[section_name] = section_text

        return sections

    def _split_text(self, text: str) -> List[str]:
        """
        Split text into chunks respecting sentence boundaries.

        Args:
            text: Text to split

        Returns:
            List of text chunks
        """
        if len(text) <= self.max_chunk_size:
            return [text]

        chunks = []
        sentences = re.split(r'(?<=[.!?])\s+', text)
        current_chunk = ""

        for sentence in sentences:
            if len(current_chunk) + len(sentence) <= self.max_chunk_size:
                current_chunk += (" " if current_chunk else "") + sentence
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())

                # Handle very long sentences
                if len(sentence) > self.max_chunk_size:
                    words = sentence.split()
                    current_chunk = ""
                    for word in words:
                        if len(current_chunk) + len(word) + 1 <= self.max_chunk_size:
                            current_chunk += (" " if current_chunk else "") + word
                        else:
                            chunks.append(current_chunk.strip())
                            current_chunk = word
                else:
                    current_chunk = sentence

        if current_chunk:
            chunks.append(current_chunk.strip())

        # Add overlap
        if self.overlap > 0 and len(chunks) > 1:
            overlapped_chunks = []
            for i, chunk in enumerate(chunks):
                if i > 0:
                    # Add end of previous chunk as context
                    prev_overlap = chunks[i-1][-self.overlap:]
                    chunk = prev_overlap + " " + chunk
                overlapped_chunks.append(chunk)
            chunks = overlapped_chunks

        return chunks

    def chunk_jobs_batch(self, jobs: List[JobPosting]) -> Tuple[List[JobChunk], Dict]:
        """
        Chunk multiple job postings.

        Args:
            jobs: List of JobPosting objects

        Returns:
            Tuple of (all chunks, statistics dict)
        """
        all_chunks = []
        stats = {
            'total_jobs': len(jobs),
            'total_chunks': 0,
            'avg_chunks_per_job': 0,
            'chunks_by_section': {}
        }

        for job in jobs:
            chunks = self.chunk_job(job)
            all_chunks.extend(chunks)

            for chunk in chunks:
                section = chunk.section.split('_')[0]  # Remove index suffix
                stats['chunks_by_section'][section] = stats['chunks_by_section'].get(section, 0) + 1

        stats['total_chunks'] = len(all_chunks)
        stats['avg_chunks_per_job'] = len(all_chunks) / len(jobs) if jobs else 0

        logger.info(f"Chunked {len(jobs)} jobs into {len(all_chunks)} chunks "
                   f"(avg {stats['avg_chunks_per_job']:.1f} chunks/job)")

        return all_chunks, stats


def load_jobs_from_jsonl(file_path: str, limit: Optional[int] = None) -> List[JobPosting]:
    """
    Load jobs from JSONL file.

    Args:
        file_path: Path to JSONL file
        limit: Maximum number of jobs to load (None for all)

    Returns:
        List of JobPosting objects
    """
    jobs = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            try:
                raw = json.loads(line.strip())
                jobs.append(JobPosting.from_raw(raw))
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse line {i}: {e}")
                continue

    logger.info(f"Loaded {len(jobs)} jobs from {file_path}")
    return jobs


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Test with sample jobs
    data_path = Path(__file__).parent.parent / "data" / "techmap-jobs_us_2023-05-05.json"

    if data_path.exists():
        jobs = load_jobs_from_jsonl(str(data_path), limit=5)
        chunker = JobChunker()

        for job in jobs[:2]:
            print(f"\n{'='*60}")
            print(f"Job: {job.title} at {job.company}")
            chunks = chunker.chunk_job(job)
            print(f"Chunks: {len(chunks)}")
            for chunk in chunks:
                print(f"  [{chunk.section}] ({chunk.importance:.1f}): {chunk.text[:100]}...")
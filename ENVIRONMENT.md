# Environment Setup Documentation

**Project:** AI-Assisted Resume Analyzer & Job Recommender
**Date:** 2025-11-13
**Python Version:** 3.13.7

---

## Virtual Environment

**Tool:** uv (version 0.9.8)
**Environment Location:** `.venv/`
**Python Interpreter:** CPython 3.13.7

---

## Installed Packages

**Total Packages:** 215

### Core Dependencies

**NLP & ML:**
- langchain==1.0.5
- langchain-classic==1.0.0
- langchain-community==0.4.1
- langchain-core==1.0.4
- langchain-text-splitters==1.0.0
- sentence-transformers==5.1.2
- transformers==4.57.1
- torch==2.9.1
- ollama==0.6.0

**PDF & Text Processing:**
- pdfplumber==0.11.8
- pypdf==6.2.0
- spacy==3.8.9
- en-core-web-sm==3.8.0 (spaCy English model)
- keybert==0.9.0

**Vector Database & Similarity:**
- **faiss-cpu==1.12.0** (FAISS CPU version for vector similarity search)
- scikit-learn==1.7.2
- scipy==1.16.3

**Data & Computation:**
- pandas==2.3.3
- numpy==2.3.4

**Web UI (Optional):**
- streamlit==1.51.0

**Utilities:**
- python-dotenv==1.2.1
- pydantic==2.12.4
- pydantic-core==2.41.5
- pydantic-settings==2.12.0
- python-dateutil==2.9.0.post0
- requests==2.32.5

**Logging & Debugging:**
- loguru==0.7.3

**Testing:**
- pytest==9.0.1
- pytest-cov==7.0.0
- pytest-asyncio==1.3.0

**Development:**
- black==25.11.0
- flake8==7.3.0
- isort==7.0.0
- mypy==1.18.2

**Jupyter (Optional):**
- jupyter==1.1.1
- notebook==7.4.7
- ipykernel==7.1.0
- jupyterlab==4.4.10

---

## Ollama Models

**Ollama Version:** Installed and running

**Available Models for Resume Parsing:**

| Model | ID | Size | Purpose |
|-------|-----|------|---------|
| **granite4:micro** | 4235724a127c | 2.1 GB | Fastest baseline for quick iteration |
| **llama3.2:3b** | a80c4f17acd5 | 2.0 GB | Balanced speed/quality |
| **gemma3:4b** | a2af6cc3eb7f | 3.3 GB | Highest quality extraction |

**Additional Models Available:**
- granite4:small-h (19 GB)
- qwen3-coder:30b-a3b-q4_K_M (18 GB)
- qwen3:30b-a3b-instruct-2507-q4_K_M (18 GB)
- qwen3:30b-a3b-thinking-2507-q4_K_M (18 GB)
- mistral:7b (4.4 GB)
- llama3.1:8b (4.9 GB)
- gemma3:12b-it-q4_K_M (8.1 GB)
- gemma3n:e2b (5.6 GB)

**Test Status:** ✅ Verified - granite4:micro responds correctly

---

## Dataset

**File:** `data/techmap-jobs_us_2023-05-05.json`
**Size:** 4.2 GB
**Format:** JSON (one object per line - JSONL format)
**Date:** 2023-05-05
**Estimated Job Postings:** ~200,000+

### JSON Schema (Key Fields)

Each job posting contains:

```json
{
  "_id": {"$oid": "string"},
  "sourceCC": "us",
  "source": "string",
  "idInSource": "string",
  "locationID": {"$oid": "string"},
  "companyID": {"$oid": "string"},
  "text": "string (plain text job description)",
  "html": "string (HTML formatted description)",
  "json": {
    "schemaOrg": {
      "@context": "http://schema.org",
      "@type": "JobPosting",
      "title": "string (job title)",
      "description": "string (full description)",
      "employmentType": "string",
      "datePosted": "ISO date string",
      "hiringOrganization": {
        "@type": "Organization",
        "name": "string (company name)",
        "logo": "string (URL)"
      },
      "jobLocation": {
        "@type": "Place",
        "address": {
          "@type": "PostalAddress",
          "addressLocality": "string (city)",
          "addressRegion": "string (state)",
          "addressCountry": "string (country code)"
        }
      }
    }
  },
  "locale": "string",
  "position": {
    "name": "string (job title)",
    "workType": "string"
  },
  "orgAddress": {
    "companyName": "string",
    "addressLine": "string",
    "country": "string",
    "state": "string",
    "city": "string"
  },
  "orgCompany": {
    "nameOrg": "string (company name)",
    "name": "string"
  },
  "name": "string (job title)",
  "url": "string (original job posting URL)",
  "dateScraped": {"$date": "ISO datetime"},
  "dateMerged": {"$date": "ISO datetime"},
  "dateUploaded": {"$date": "ISO datetime"},
  "dateCreated": {"$date": "ISO datetime"},
  "orgTags": {
    "CATEGORIES": ["array of job categories"]
  }
}
```

### Useful Fields for Matching:
- **Job Title:** `name`, `position.name`, `json.schemaOrg.title`
- **Description:** `text` (plain text - best for NLP), `html`, `json.schemaOrg.description`
- **Company:** `orgCompany.name`, `json.schemaOrg.hiringOrganization.name`
- **Location:** `orgAddress` (city, state, country)
- **Date:** `dateCreated`
- **Categories:** `orgTags.CATEGORIES`

---

## System Configuration

**Operating System:** Windows (win32)
**Git Repository:** Yes (initialized)
**Branch:** main

---

## Installation Commands Used

```bash
# Create virtual environment with Python 3.13
uv venv --python 3.13

# Install all dependencies
uv pip install -r requirements.txt

# Install spaCy English model
uv pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl
```

---

## Key Decisions

### 1. Python Version
- **Chosen:** Python 3.13.7
- **Rationale:** Latest stable version with all required package support (after updating scikit-learn)

### 2. FAISS Version
- **Chosen:** faiss-cpu (not faiss-gpu)
- **Rationale:** faiss-gpu not available for Windows; faiss-cpu sufficient for dataset size (~1.2M vectors)

### 3. Package Manager
- **Chosen:** uv
- **Rationale:** Project plan specifies uv; faster than pip, modern dependency resolution

### 4. Virtual Environment Tool
- **Chosen:** uv venv
- **Rationale:** Integrated with uv package manager, minimal environment setup

---

## Verification Status

- ✅ Python 3.13.7 installed and working
- ✅ Virtual environment created (.venv/)
- ✅ 215 packages installed successfully
- ✅ faiss-cpu 1.12.0 installed
- ✅ spaCy English model (en_core_web_sm-3.8.0) installed
- ✅ Dataset verified (4.2GB JSON file accessible)
- ✅ Ollama running with 3 required models (granite4:micro, llama3.2:3b, gemma3:4b)
- ✅ Ollama test successful (granite4:micro responds)

---

## Next Steps

**Phase 2:** Resume Parsing & Structured Extraction
- Test all 3 LLM models for resume parsing
- Build PDF parser with LangChain
- Design Resume JSON schema with Pydantic
- Implement LLM-powered resume extractor

**Phase 3:** Text Preprocessing & Skill Extraction
- Set up text preprocessing pipeline with spaCy
- Implement skill extraction with KeyBERT
- Build unified preprocessing pipeline

**Phase 4:** Embedding Generation & Vector Storage
- Implement chunking strategy for job postings
- Build job ID mapping system
- Generate embeddings and build FAISS index
- Implement MMR for diverse results

---

**Last Updated:** 2025-11-13
**Status:** Environment setup complete, ready for Phase 2
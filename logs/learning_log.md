# AI-Assisted Resume Analyzer & Job Recommender
## Learning Log

**Project Start Date:** 2025-11-13
**Author:** Student Learning Documentation
**Project Goal:** Build an end-to-end NLP pipeline that matches resumes to relevant job postings, identifies skill gaps, and provides recommendations.

---

## Learning Objectives

1. Master LangChain for document processing and LLM orchestration
2. Implement semantic search using embeddings and FAISS
3. Build production-grade NLP pipelines with proper logging and experimentation
4. Compare and evaluate local LLM models (Ollama: granite4:micro, llama3.2:3b, gemma3:4b)
5. Implement chunking strategies for improved matching accuracy
6. Build comprehensive skill extraction and gap analysis systems
7. Create an end-to-end system from PDF parsing to job recommendations

---

## Phase 0: Documentation & Logging Infrastructure

**Start Date:** 2025-11-13
**End Date:** 2025-11-13
**Status:** Completed

---

## Phase 1: Project Setup & Data Preparation

**Start Date:** 2025-11-13
**End Date:** 2025-11-13
**Status:** Completed


---

## Phase 2: Resume Parsing & Structured Extraction

**Start Date:** 2025-11-18
**End Date:** 2025-11-20
**Status:** Completed

### Step 2.1: Verify & Test Local LLM Models

**Status:** Completed

#### Step 2.1 - Test Complex Resume, 4 runs (1 warm-up) per model - Final Checkpoint
**Timestamp:** 2025-11-18 21:48:31

Implemented rigorous testing with:
- **Complex resume** with edge cases (multiple emails, promotions, acquisitions, 40+ skills)
- **Ground truth validation** (checks correctness, not just presence)
- **4 test runs per model** (1st discarded as warm-up, remaining 3 averaged)
- **Hallucination detection** and strict field validation

**Results (averaged over 3 valid runs):**

1. **gemma3:4b** - WINNER
   - Avg Latency: 14.61s (slowest, but acceptable)
   - Quality Score: **100.0%** (PERFECT - all validation checks passed)
   - Validation Errors: **0**
   - Reliability: 4/4 runs succeeded, all with perfect scores
   - Extracted ALL fields correctly including complex cases

2. **granite4:micro** - Runner-up
   - Avg Latency: 7.92s (good middle ground)
   - Quality Score: **88.9%** (good, but not perfect)
   - Validation Errors: **1** (inconsistent across runs)
   - Reliability: 4/4 runs succeeded
   - Struggled with some edge cases

3. **llama3.2:3b** - Failed harder test
   - Avg Latency: 3.02s (fastest, but unreliable)
   - Quality Score: **50.0%** (POOR - failed multiple validation checks)
   - Validation Errors: 1, Warnings: 2
   - Reliability: **1 run produced invalid JSON** (only 2/3 valid runs after warm-up)
   - Extracted too few skills, missed fields, inconsistent output

**Final Decision:** **Use gemma3:4b as primary model** for resume parsing. The 14.61s latency is acceptable given:
- Perfect extraction accuracy (100%)
- Zero validation errors
- Perfect consistency across all runs
- Handles complex edge cases flawlessly
- Production reliability is worth the extra ~7 seconds

**Metrics:**
- gemma3:4b: 14.61s, 100.0% quality, 0 errors, 4/4 runs perfect
- granite4:micro: 7.92s, 88.9% quality, 1 error, 4/4 runs succeeded
- llama3.2:3b: 3.02s, 50.0% quality, 1 error + 2 warnings, 3/4 runs succeeded

**Next:** Build PDF parser with LangChain and test with actual PDF samples.

### Step 2.2: Build PDF Parser with LangChain

**Status:** Completed
**Timestamp:** 2025-11-20

Implemented `PDFResumeParser` with dual library support:
- **PDFPlumber**: Primary parser, table-aware, better field detection (90.5% avg)
- **PyPDF**: Fallback parser, faster (2.3x), better format handling (76.2% avg)

**Key Findings from Quality Assessment (3 challenging fake resumes):**
- Both parsers averaged 3.0/5 user rating - neither definitively superior
- PDFPlumber: Better at field detection but struggles with complex tables
- PyPDF: Faster but inconsistent field detection, better text ordering
- Different parsers excel at different resume formats (complementary strengths)

**Decision:** Implement dual-parser sequential pipeline to leverage both strengths

### Step 2.3: Define Pydantic Resume Schema

**Status:** Completed
**Timestamp:** 2025-11-18

Created comprehensive Pydantic V2 schema (`resume_schema.py`):
- Main model: `Resume` with nested `ExperienceEntry` and `EducationEntry`
- Field validators for email, phone, skills (with deduplication)
- Support for: contact info, experience, skills, education, certifications, languages, projects
- Added `additional_sections` dict for uncategorized content (Awards, Volunteer, Publications, etc.)
- JSON schema generation for LLM prompts

### Step 2.4 & 2.5: Dual-Parser Resume Extractor & Validation

**Status:** Completed
**Timestamp:** 2025-11-20

Implemented `ResumeExtractor` with dual-parser sequential pipeline:

**Architecture:**
1. **First Pass**: PDFPlumber → LLM extraction → Pydantic validation
2. **Second Pass**: PyPDF → LLM refinement (with previous JSON as context) → Pydantic validation
3. **Fallback**: If second pass fails, returns first pass result (ensures 100% success rate)

**Test Results (3 fake resumes, gemma3:4b @ temp=0.1):**

| Resume | Pipeline Mode | Result | Time | Skills | Experience | Education | Certs |
|--------|---------------|--------|------|--------|------------|-----------|-------|
| Harper Russo | Single-parser (refinement failed) | ✓ SUCCESS | 30.64s | 34 | 2 | 1 | 3 |
| Henry Wotton | Dual-parser ✓ | ✓ SUCCESS | 20.25s | 10 | 2 | 1 | 2 |
| Isabella Bella Ruiz | Dual-parser ✓ | ✓ SUCCESS | 35.55s | 19 | 2 | 1 | 3 |

**Success Rate:**
- Overall: 100% (3/3 extractions succeeded)
- Dual-parser success: 67% (2/3 succeeded with refinement)
- Fallback effectiveness: 100% (Harper Russo fell back gracefully)

**Key Findings:**
1. **Fallback mechanism is critical** - Ensures 100% extraction success rate even when refinement fails
2. **Refinement prompt engineering matters** - Fixed field name errors with explicit schema requirements
3. **Dual-parser adds value when successful** - Henry Wotton completed in 20.25s (faster than single-parser)
4. **JSON repair function helps** - Fixes common LLM formatting issues (trailing commas, missing commas)
5. **Remaining issues**: Occasional JSON syntax errors from LLM on complex resumes (Harper Russo)
6. **Name extraction quirk**: Some resumes have spaced names ("H A R P E R R U S S O") from PDF formatting

**Validation Framework:**
- Comprehensive quality scoring (0-100%)
- Section-level validation (contact, experience, education, skills)
- Retry logic with quality threshold (>= 50%)
- Detailed error reporting with issue/warning classification

**Code Organization:**
- Moved test scripts to `tests/` folder for better structure
- Updated imports in test scripts to reference `src/` modules
- Core modules in `src/`: resume_extractor, pdf_parser, resume_schema, extraction_validator, logging_utils
- Test modules in `tests/`: test_llm_models, test_pdf_quality

**Next:** Test with more diverse resume formats, move to Phase 3 (Text Preprocessing & Skill Extraction)

---

## Phase 3: Text Preprocessing & Skill Extraction

**Status:** Not Started

*(To be updated as phase progresses)*

---

## Phase 4: Embedding Generation & Vector Storage

**Status:** Not Started

*(To be updated as phase progresses)*

---

## Phase 5: Resume-to-Job Matching & Ranking

**Status:** Not Started

*(To be updated as phase progresses)*

---

## Phase 6: CV Improvement Suggestions (Stretch Goal)

**Status:** Not Started

*(To be updated as phase progresses)*

---

## Phase 7: Streamlit UI (Stretch Goal)

**Status:** Not Started

*(To be updated as phase progresses)*

---

## Phase 8: Testing & Documentation

**Status:** Not Started

*(To be updated as phase progresses)*

---

## Phase 9: Learning Documentation & Presentation

**Status:** Not Started

*(To be updated as phase progresses)*

---

## Overall Reflections

*(To be updated throughout the project)*

### Key Learnings
- TBD

### Challenges Overcome
- TBD

### Skills Developed
- TBD

### What I Would Do Differently
- TBD
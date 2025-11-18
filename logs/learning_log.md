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
**Status:** In Progress

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
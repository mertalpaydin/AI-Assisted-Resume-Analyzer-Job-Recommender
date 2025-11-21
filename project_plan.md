# AI-Assisted Resume Analyzer & Job Recommender
## Claude Code Step-by-Step Implementation Plan

**Project Goal:** Build an end-to-end NLP pipeline that matches resumes to relevant job postings, identifies skill gaps, and provides recommendations.

**Learning Log:** Document findings, decisions, and experiments at each stage as a summary. Needs to be updated after completed of each Phase.
**Experiment Log:** Document experiments at each stage. Needs to be updated after completion of each experiment.
**Decision Log:** Document decisions such as with Ollama model to use at each stage. Needs to be updated after each decision.

---

## Phase 0: Documentation & Logging Infrastructure (PRIORITY)

### Step 0.1: Initialize Documentation Structure
- [x] Create `/logs/learning_log.md` - main learning documentation file
  - [x] Add header with project name, start date, learning objectives
  - [x] Create sections for each phase (to be updated continuously)
  - [x] Use consistent markdown formatting

- [x] Create `/logs/experiment_log.md` - detailed experiment tracking
  - [x] Initialize experiment counter
  - [x] Create template section for standardized logging

- [x] Create `/logs/decisions.md` - architectural and technical decisions
  - [x] Track rationale for each major choice
  - [x] Document trade-offs considered

- [x] Create `/logs/challenges.md` - problems encountered and solutions
  - [x] Document issues as they arise
  - [x] Note resolution approaches

- [x] Set up logging in code:
  - [x] Configure Python logging module in all scripts
  - [x] Log to both console and `/logs/debug.log`
  - [x] Include timestamps, module names, and function information

### Step 0.2: Create Logging Helper Module
- [x] Create `src/logging_utils.py`:
  - [x] Centralized logging configuration
  - [x] Functions to log experiments, decisions, and findings
  - [x] Easy-to-use wrappers for documentation capture

- [x] Create documentation checkpoint function:
  - [x] After each major step, log key findings
  - [x] Record metrics and observations
  - [x] Timestamp entries

**Critical Principle:** From this point forward, **after every completed step or experiment, immediately update the learning logs, decisions and experiment log. Do not defer documentation to the end.**

---

## Phase 1: Project Setup & Data Preparation

### Step 1.1: Environment Setup & Dependencies
- [x] Create project directory structure
  - [x] `/data` - for datasets (already downloaded)
  - [x] `/src` - for main pipeline code
  - [x] `/notebooks` - for exploration and analysis
  - [x] `/logs` - for learning documentation (initialized in Phase 0)
  - [x] `/tests` - for unit tests

- [x] Set up Python virtual environment
- [x] Install core dependencies via uv using requirements.txt
- [x] **Logging:** Document Python version and package versions installed

### Step 1.2: Verify Data & Ollama Setup
- [x] Verify dataset location and load:
  - [x] Confirm US Job Postings dataset is accessible
  - [x] Load and examine JSON schema
  - [x] Identify key fields (job title, description, required skills, etc.)
  - [x] Check data quality and missing values
  - [x] Sample 5-10 job postings for testing

- [x] Verify Ollama installation and downloaded models:
  - [x] List available models: `ollama list`
  - [x] Test each model:
    - [x] granite4:micro - verify functionality
    - [x] llama3.2:3b - verify functionality
    - [x] gemma3:4b - verify functionality
  - [x] Document model characteristics: size, latency, response quality

- [x] Prepare sample resumes:
  - [x] Create 3-5 realistic test resumes in PDF format
  - [x] Ensure they cover different experience levels a**Experiment Log:** Try resumes with varying formatting (headers, bullets, paragraphs)

- [x] **Learning Log:** Document initial observations about data structure, models, and potential challenges

---

## Phase 2: Resume Parsing & Structured Extraction

### Step 2.1: Verify & Test Local LLM Models ✓ COMPLETED
- [x] Confirm Ollama is running and models are available:
  - [x] `ollama list` to verify downloaded models
  - [x] Ensure models are accessible via Ollama API

- [x] Test each available model with sample prompts:
  - [x] **granite4:micro** - test parsing capability, document latency
  - [x] **llama3.2:3b** - test parsing capability, document latency
  - [x] **gemma3:4b** - test parsing capability, document latency

- [x] **Experiment Log:** Create systematic comparison:
  - [x] Response quality for resume extraction (rigorous validation with ground truth)
  - [x] Speed and latency measurements (4 runs per model, 1st discarded)
  - [x] Memory/resource usage
  - [x] Reliability (consistency across runs)
  - [x] Document findings in `/logs/experiment_log.md`
  - [x] Make preliminary recommendation for Phase 2.4

**RESULT:** gemma3:4b selected (100% quality, 14.61s latency, perfect consistency)

### Step 2.2: Build PDF Parser with LangChain ✓ COMPLETED
- [x] Create `pdf_parser.py` module:
  - [x] Use LangChain's document loaders (PDFPlumber or PyPDF)
  - [x] Extract raw text from PDFs
  - [x] Handle multi-page resumes
  - [x] Test on sample resumes

- [x] **Experiment Log:** Try multiple PDF loading strategies:
  - [x] Different page extraction methods
  - [x] Handling of special formatting (tables, columns)
  - [x] Performance metrics for each approach

### Step 2.3: Design Resume JSON Schema ✓ COMPLETED
- [x] Define structured output schema with fields:
  - [x] Full resume schema with contact info, experience, education, skills, certifications, languages, projects
  - [x] Nested models for ExperienceEntry and EducationEntry
  - [x] Optional fields for LinkedIn, GitHub, website, location

- [x] Create Pydantic models for type validation
  - [x] Implemented with Pydantic V2 (ConfigDict)
  - [x] Field validators for name, email, phone, skills
  - [x] Automatic deduplication for skills list
  - [x] Comprehensive validation and cleaning

**RESULT:** Complete Pydantic schema in `src/resume_schema.py` with validation, tested and working

### Step 2.4: Build LLM-Powered Resume Extractor ✓ COMPLETED
- [x] Create `resume_extractor.py` using LangChain:
  - [x] Craft prompt template for structured extraction (with clear instructions to avoid hallucination)
  - [x] Use Pydantic output parser for validation
  - [x] Chain together: PDF text → gemma3:4b LLM → Structured JSON → Pydantic validation
  - [x] Integrated with PDFResumeParser (with fallback support)

- [x] Implementation features:
  - [x] End-to-end extraction pipeline (PDF → Resume object)
  - [x] Batch processing support
  - [x] Comprehensive metadata tracking (timing, counts, model info)
  - [x] Low temperature (0.1) for consistent extraction
  - [x] JSON extraction with error handling

- [x] Test on sample resumes:
  - [x] Verify JSON schema compliance
  - [x] Check extraction accuracy for each field
  - [x] Measure latency and token usage

**RESULT:** Complete extraction pipeline in `src/resume_extractor.py`. Testing done, and logged. Going with dual parser approach.

### Step 2.5: Build Error Handling & Validation ✓ COMPLETED
- [x] Add validation for:
  - [x] Missing critical fields (contact info, experience, education, skills)
  - [x] Email/phone format validation
  - [x] Skill name normalization and deduplication
  - [x] Section-by-section quality checks

- [x] Implement retry logic for LLM failures
  - [x] Automatic retry up to 2 attempts
  - [x] Quality score threshold (minimum 50%)
  - [x] Metadata tracking for all attempts

- [x] Create logging for extraction confidence scores
  - [x] Quality scoring system (0-100%)
  - [x] Issue and warning tracking
  - [x] Detailed validation reports

**RESULT:** Complete validation framework in `src/extraction_validator.py` with retry logic and quality scoring

**Phase 2 Summary:** All implementation and testing completed!

---

## Phase 3: Text Preprocessing & Skill Extraction ✓ COMPLETED

### Step 3.1: Set Up Text Preprocessing Pipeline ✓ COMPLETED
- [x] Create `text_preprocessor.py`:
  - [x] Download spaCy English model (`en_core_web_sm`)
  - [x] Implement text cleaning:
    - [x] Lowercase conversion
    - [x] Remove special characters and extra whitespace
    - [x] Remove stopwords
    - [x] Remove URLs and email addresses

  - [x] Implement tokenization using spaCy
  - [x] Implement lemmatization
  - [x] Test on sample texts

- [x] **Experiment Log:** Compare preprocessing approaches:
  - [x] Implemented three modes (MINIMAL, STANDARD, FULL)
  - [x] Tested with and without stopword removal
  - [x] Lemmatization implemented (no stemming comparison needed)
  - [x] Ready for downstream matching

### Step 3.2: Implement Skill Extraction with RAKE (Decision #3)
- [x] Create `skill_extractor.py`:
  - [x] Initialize RAKE, YAKE, and KeyBERT models
  - [x] Extract skills from resume text and job descriptions
  - [x] Normalize skill names (e.g., "Python" → "python", "C++" → "cpp")

- [x] Create skill normalization lookup:
  - [x] Common aliases (e.g., "Deep Learning" → "deep learning", "DL")
  - [x] Technology variations (e.g., "JS", "Javascript", "JavaScript")
  - [x] Built comprehensive skill database (80 canonical skills)

- [x] **Experiment Log:** Test different extraction methods (Experiment #3):
  - [x] KeyBERT with all-MiniLM-L6-v2 (F1: 0.133)
  - [x] KeyBERT with all-mpnet-base-v2 (F1: 0.133, 12x slower)
  - [x] YAKE statistical method (F1: 0.273)
  - [x] RAKE statistical method (F1: 0.356 **WINNER**)
  - [x] Evaluated extraction quality and performance
  - [x] **Decision:** Use RAKE as primary method (2.7x better quality, 13x faster)

### Step 3.3: Build Unified Preprocessing Pipeline ✓ COMPLETED
- [x] Create `preprocessing_pipeline.py`:
  - [x] Combine cleaning, tokenization, and skill extraction
  - [x] Apply to both resume and job posting texts
  - [x] Store preprocessed data with metadata (timestamps, confidence scores)
  - [x] Implement caching system for performance
  - [x] Document comparison functionality

- [x] **Learning Log:** Documented preprocessing decisions and Phase 3 completion

---

## Phase 4: Embedding Generation & Vector Storage (Chunking Strategy) ✓ Mostly COMPLETED

**Status:** COMPLETED - All implementation and experiments done.

### Step 4.1: Intelligent Job Chunking ✓ COMPLETED
- [x] Create `JobChunker` class:
  - [x] Semantic chunking (split by sections: title, description, requirements)
  - [x] Mark each chunk with job_id and chunk_num for easier reconstruction
  - [x] Track chunk importance (title > requirements > description)
  - [x] Test on sample jobs (500 jobs → 3627 chunks, avg 7.3 chunks/job)

- [x] **Learning Log:** Document chunking strategy:
  - [x] Chunks per job: ~7.3 average
  - [x] Chunk size effectiveness: optimal for embedding quality

### Step 4.2: Job ID Mapping System ✓ COMPLETED
- [x] Create `JobIDMapper` class:
  - [x] Bidirectional mapping: chunk_id ↔ job_id
  - [x] Store complete job postings
  - [x] Implement deduplication logic (multiple chunks → one job)
  - [x] Implement `get_job_by_chunk_id()` - retrieve full job
  - [x] Implement `get_unique_jobs()` - deduplicate results

- [x] **Critical:** Automatic deduplication
  - [x] Multiple chunks from same job → returned once
  - [x] Preserves best matching chunk info
  - [x] Maintains link to complete job posting

### Step 4.3: Chunk Embedding Generation ✓ COMPLETED
- [x] Create `ChunkEmbeddingGenerator` class:
  - [x] Efficient batch embedding for all chunks
  - [x] Track: chunk_id, job_id, section, importance
  - [x] Cache embeddings to disk (numpy format)
  - [x] Support for asymmetric encoding (query vs document)

- [x] **Experiment Log:** Compare embedding models (Experiment #4):
  - [x] all-MiniLM-L6-v2: Precision 85.0%, Speed 1489/s
  - [x] all-mpnet-base-v2: Precision 86.0%, Speed 242/s
  - [x] **google/embeddinggemma-300m: Precision 98.0%, Speed 147/s (WINNER)**
  - [x] EmbeddingGemma selected for 12-13% precision improvement

### Step 4.4: Chunk Vector Store with Job ID Deduplication ✓ COMPLETED
- [x] Create `ChunkVectorStore` class:
  - [x] Build FAISS index from chunk embeddings (IndexFlatIP)
  - [x] Store chunk metadata (chunk_id, job_id, section)
  - [x] Implement `search_chunks()` - returns top-k chunks
  - [x] Implement `search_with_job_dedup()` - deduplicates by job_id
  - [x] Implement `save()` / `load()` - persist index + metadata

- [x] **Key feature:** Automatic deduplication
  - [x] Search returns chunks, deduplicates to unique jobs
  - [x] Preserves best matching chunk score
  - [x] Returns complete job postings

### Step 4.5: Maximum Marginal Relevance (MMR) - Chunk-based ✓ COMPLETED
- [x] Implement `search_mmr()` in vector store:
  - [x] Work with chunks → then deduplicate to jobs
  - [x] Balance relevance (chunk similarity) with diversity
  - [x] Returns **unique complete job postings**

- [x] **Experiment Log:** MMR lambda parameter tuning:
  - [x] cosine: Precision 0.850, Companies 9.5/10
  - [x] MMR λ=0.3: Precision 0.850, Companies 9.9/10
  - [x] **MMR λ=0.5: Precision 0.880, Companies 9.9/10 (WINNER)**
  - [x] MMR λ=0.7: Precision 0.840, Companies 9.9/10
  - [x] MMR λ=0.9: Precision 0.850, Companies 9.5/10

**Phase 4 Summary:**
- **Embedding Model:** google/embeddinggemma-300m (98% precision)
- **Retrieval:** MMR with λ=0.5 (88% precision, 9.9/10 diversity)
- **Vector Store:** FAISS IndexFlatIP, 768-dim embeddings

todo: generate embeddings for the job ad data with the winner embedding model
---

## Phase 5: Resume-to-Job Matching & Ranking

### Step 5.1: Build Matching Engine
- [ ] Create `matching_engine.py`:
  - [ ] For uploaded resume:
    1. Parse and extract structured data
    2. Generate embeddings
    3. Query FAISS for similar jobs
  
  - [ ] Implement ranking strategies:
    - [ ] Weighted similarity scores (job description + title + skills)
    - [ ] Custom scoring based on experience level
    - [ ] Recency boost for recent postings

- [ ] **Experiment Log:** Test different ranking strategies:
  - [ ] Equal weighting
  - [ ] Emphasis on skill matches
  - [ ] Machine learning-based scoring (optional)

### Step 5.2: Implement Skill Gap Analysis
- [ ] Create `skill_gap_analyzer.py`:
  - [ ] Extract required skills from job posting
  - [ ] Extract candidate skills from resume
  - [ ] Identify:
    - [ ] Matched skills (highlight overlaps)
    - [ ] Missing skills (gaps to address)
    - [ ] Extra skills (candidate advantages)
  
  - [ ] Generate skill match percentage and breakdown

- [ ] **Experiment Log:** Refine skill matching logic:
  - [ ] Exact vs. fuzzy matching for skill names
  - [ ] Skill level consideration (junior vs. senior)
  - [ ] Skill category grouping (frontend, backend, data, etc.)

### Step 5.3: Build Recommendation Result Object
- [ ] Create data structure for recommendation results:
  ```json
  {
    "job_id": "string",
    "job_title": "string",
    "company": "string",
    "similarity_score": "float",
    "skill_match_percentage": "float",
    "matched_skills": ["string"],
    "missing_skills": ["string"],
    "candidate_extra_skills": ["string"],
    "reasoning": "string"
  }
  ```

- [ ] Return top K matches (configurable, default 10)
- [ ] Include reasoning/explanation for each match

### Step 5.4: Create Comprehensive Output Report
- [ ] Generate detailed matching report:
  - [ ] Resume summary
  - [ ] Top job matches ranked by relevance
  - [ ] Overall skill profile summary
  - [ ] Recommendations for skill development

- [ ] Export formats:
  - [ ] JSON (programmatic use)
  - [ ] Markdown (readability)
  - [ ] HTML (nice visualization)

- [ ] **Learning Log:** Document recommendations quality and user feedback

---

## Phase 6: Stretch Goal – CV Improvement Suggestions

### Step 6.1: Build LLM-Powered CV Improvement Module
- [ ] Create `cv_improvemer.py`:
  - [ ] Analyze resume against top matching job postings
  - [ ] Generate suggestions for:
    - [ ] Missing skills to develop
    - [ ] How to better highlight existing skills
    - [ ] Experience descriptions to improve
    - [ ] Keywords to include for ATS optimization

- [ ] **Experiment Log:** Test different suggestion generation strategies:
  - [ ] Different LLMs (Ollama models)
  - [ ] Try Gemini API calls, compare accuracy and latency vs Ollama models
  - [ ] Prompt engineering variations
  - [ ] Few-shot examples
  - [ ] Chain-of-thought reasoning

### Step 6.2: Create Actionable Recommendations
- [ ] Structure suggestions as:
  - [ ] High-priority (critical skill gaps)
  - [ ] Medium-priority (nice-to-have skills)
  - [ ] Low-priority (optional improvements)
  - [ ] Concrete action items with resources/links

- [ ] **Learning Log:** Document quality and practicality of suggestions

---

## Phase 7: Stretch Goal – Streamlit UI

### Step 7.1: Design Streamlit Application Layout
- [ ] Create `app.py`:
  - [ ] File upload component for PDF resumes
  - [ ] Processing status indicator
  - [ ] Results display sections:
    - [ ] Resume summary overview
    - [ ] Top job matches table/cards
    - [ ] Skill match breakdown (bar chart)
    - [ ] Skill gap visualization
    - [ ] CV improvement suggestions

### Step 7.2: Implement Resume Upload & Processing
- [ ] Handle file uploads:
  - [ ] Validate PDF format and size
  - [ ] Show processing progress
  - [ ] Handle errors gracefully
  
- [ ] Display parsing results:
  - [ ] Show extracted resume data
  - [ ] Allow manual corrections if needed
  - [ ] Confirm before proceeding to matching

### Step 7.3: Implement Interactive Results Display
- [ ] Create tabs/sections for:
  - [ ] Overview dashboard
  - [ ] Detailed job matches with filtering/sorting
  - [ ] Skill analysis and gap visualization
  - [ ] CV improvement suggestions
  - [ ] Export options

- [ ] Add interactivity:
  - [ ] Filter by job category, location, salary
  - [ ] Click through for detailed job posting view
  - [ ] Copy suggestions to clipboard
  - [ ] Bookmark/save favorites

### Step 7.4: Deploy & Optimize
- [ ] Optimize for performance:
  - [ ] Cache embeddings and FAISS index
  - [ ] Lazy load heavy models
  - [ ] Session state management

- [ ] Add configuration panel:
  - [ ] Top K results
  - [ ] Similarity threshold
  - [ ] Model selection (LLM, embedding model)

- [ ] Deploy options:
  - [ ] Local Streamlit server
  - [ ] Streamlit Cloud deployment
  - [ ] Docker containerization

---

## Phase 8: Testing & Documentation

### Step 8.1: Unit Testing
- [ ] Create tests for:
  - [ ] PDF parsing with various formats
  - [ ] JSON extraction accuracy
  - [ ] Text preprocessing functions
  - [ ] Skill extraction consistency
  - [ ] Embedding generation
  - [ ] Similarity computations
  - [ ] Ranking logic

- [ ] **Learning Log:** Document test coverage and edge cases discovered

### Step 8.2: Integration Testing
- [ ] End-to-end pipeline tests:
  - [ ] Resume upload → parsing → matching → recommendations
  - [ ] Performance benchmarks (latency, memory)
  - [ ] Accuracy validation on diverse resumes

### Step 8.3: Create Comprehensive Documentation
- [ ] README with:
  - [ ] Project overview and goals
  - [ ] Setup instructions
  - [ ] Usage examples
  - [ ] Model selection rationale
  - [ ] Performance metrics

- [ ] Code documentation:
  - [ ] Docstrings for all functions
  - [ ] Inline comments for complex logic
  - [ ] Architecture diagrams

- [ ] Learning outcomes document:
  - [ ] Skills developed
  - [ ] Trade-offs analyzed
  - [ ] Experiments conducted and results
  - [ ] Future improvements

---

## Phase 9: Learning Documentation & Presentation

### Step 9.1: Create Learning Log
- [ ] Maintain structured log throughout project:
  - [ ] Each phase summary
  - [ ] Key decisions and rationale
  - [ ] Experiments conducted (models, libraries, approaches)
  - [ ] Results and comparisons
  - [ ] Challenges and solutions
  - [ ] Time spent on each task

- [ ] Document file: `/logs/learning_log.md`

### Step 9.2: Prepare Instructor Discussion Talking Points
- [ ] Organize key discussion points for instructor conversation:
  
  **Project Scope & Motivation:**
  - [ ] Why this project was chosen and what problem it solves
  - [ ] How it connects to real-world applications
  
  **Architecture & Design Decisions:**
  - [ ] High-level architecture overview (data flow diagram in mind)
  - [ ] Major technical choices and rationale:
    - [ ] Why LangChain for PDF parsing
    - [ ] Why FAISS + MMR for similarity search
    - [ ] Model selection decisions (Ollama vs. alternatives)
  - [ ] Trade-offs considered and why they were made
  
  **Workflow & Implementation:**
  - [ ] End-to-end pipeline explanation
  - [ ] Key challenges encountered and solutions
  - [ ] How logging and experimentation were integrated throughout
  
  **Experiments Conducted:**
  - [ ] LLM comparison (granite4:micro, llama3.2:3b, gemma3:4b)
  - [ ] Embedding model variations (EmbeddingGemma vs. alternatives)
  - [ ] Similarity search methods (cosine, FAISS, MMR - what worked best)
  - [ ] PDF parsing strategies and their impact
  - [ ] Skill extraction approaches and accuracy
  
  **Results & Performance:**
  - [ ] Key metrics achieved (parsing accuracy, latency, etc.)
  - [ ] Quality of recommendations on test cases
  - [ ] What surprised you (positive or negative)
  - [ ] Skill gap analysis effectiveness
  
  **Lessons Learned:**
  - [ ] Most valuable technical skill gained
  - [ ] Trade-offs between model complexity and performance
  - [ ] How you'd approach similar problems in the future
  - [ ] Importance of continuous logging/documentation during development
  
  **Limitations & Future Improvements:**
  - [ ] Current system limitations (single language, PDF-only, etc.)
  - [ ] Potential next steps if you had more time
  - [ ] How would you scale this to production
  - [ ] Feedback from instructor's perspective

### Step 9.3: Final Repository Organization
- [ ] Organize GitHub repo with:
  - [ ] Clean folder structure
  - [ ] Comprehensive README
  - [ ] requirements.txt / environment.yml
  - [ ] Sample data and test cases
  - [ ] Presentation files
  - [ ] Learning documentation
  - [ ] Architecture diagrams
  - [ ] Deployment instructions

---

## Experiment Tracking Template

For each experiment, document:
```markdown
### Experiment: [Name]
**Date:** 
**Objective:** 

**Options Tested:**
1. [Option A] - Results: 
2. [Option B] - Results: 

**Winner:** [Option] because [reason]
**Learning:** 
**Impact on Project:** 
```

---

## Key Experiment Opportunities (Go Beyond Requirements)

1. **PDF Parsing Strategies**
   - Compare: pdfplumber, pypdf, PyMuPDF, pdfminer
   - Test on: different layouts, scanned PDFs, multi-column layouts

2. **Embedding Model Comparison**
   - Baseline: EmbeddingGemma
   - Alternatives: All-MiniLM, All-MPNet, domain-specific fine-tuning
   - Metrics: Similarity quality, speed, memory

2. **LLM Model Selection**
   - Granite 4 Micro (baseline - fastest)
   - Llama 3.2 3B (balanced)
   - Gemma 3 4B (potentially highest quality)
   - Evaluate: accuracy, speed, resource usage, reliability
   - Track: which model works best for different resume types

4. **Prompt Engineering**
   - Few-shot vs. zero-shot
   - Chain-of-thought reasoning
   - Output format variations
   - Custom system prompts

5. **Similarity Metrics & Search Strategies**
   - Cosine similarity (baseline)
   - Euclidean distance via FAISS
   - Approximate nearest neighbors (FAISS IndexIVFFlat)
   - **Maximum Marginal Relevance (MMR)** - balance relevance with diversity
   - Hybrid scoring combinations
   - Metrics: accuracy, diversity, speed, scalability

6. **Skill Matching Logic**
   - Exact matching
   - Fuzzy matching (Levenshtein distance)
   - Semantic similarity for skill names
   - Skill hierarchy/taxonomy

7. **Ranking Strategies**
   - Single metric vs. multi-metric ranking
   - Learned scoring (if time permits)
   - User feedback incorporation

8. **Feature Engineering**
   - Job description parsing variants
   - Skill level inference
   - Experience-based matching
   - Salary/location considerations

---

## Success Metrics

- [ ] Resume parsing accuracy: >90% field extraction
- [ ] Skill extraction: >85% precision and recall
- [ ] Recommendation relevance: Manual validation of top matches
- [ ] Latency: <5 seconds for full pipeline per resume
- [ ] Code quality: Full test coverage, documented, clean architecture
- [ ] Portfolio readiness: Production-grade code quality

---

## Timeline Estimate

- Phase 0: 1-2 hours (logging infrastructure - do this first!)
- Phase 1: 1-2 hours (setup/verification - data and Ollama already ready)
- Phase 2: 4-6 hours (resume parsing & extraction)
- Phase 3: 2-3 hours (preprocessing & skills)
- Phase 4: 3-4 hours (embeddings, FAISS, MMR implementation)
- Phase 5: 3-4 hours (matching & ranking)
- Phase 6: 2-3 hours (optional CV suggestions)
- Phase 7: 3-4 hours (optional Streamlit UI)
- Phase 8: 2-3 hours (testing & docs)
- Phase 9: 1-2 hours (finalize learning log & discussion prep)

**Total: 23-33 hours** (flexible based on depth of experimentation)

---

## Notes

- **Version Control:** Commit after each completed phase
- **Logging:** Use Python logging module throughout for debugging
- **Reproducibility:** Document all hyperparameters and random seeds
- **Experimentation:** Don't stick rigidly to original tech choices—try alternatives
- **Documentation:** Write as you go, don't leave it for the end

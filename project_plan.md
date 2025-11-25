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

### Step 4.6: Embedding Generation Script ✓ COMPLETED
- [x] Create `src/generate_job_embeddings.py`:
  - [x] Load jobs from JSONL data file
  - [x] Chunk jobs using JobChunker
  - [x] Generate embeddings with EmbeddingGemma
  - [x] Build FAISS vector store
  - [x] Save all artifacts to `data/embeddings/`
  - [x] Command line interface with `--limit` option

**Phase 4 Summary:**
- **Embedding Model:** google/embeddinggemma-300m (98% precision)
- **Retrieval:** MMR with λ=0.5 (88% precision, 9.9/10 diversity)
- **Vector Store:** FAISS IndexFlatIP, 768-dim embeddings
- **Script:** `python src/generate_job_embeddings.py --limit N`
---

## Phase 5: Resume-to-Job Matching & Ranking ✓ COMPLETED

### Step 5.1: Build Matching Engine ✓ COMPLETED
- [x] Create `matching_engine.py`:
  - [x] For uploaded resume:
    1. Parse and extract structured data (gemma3:4b)
    2. Generate embeddings (EmbeddingGemma)
    3. Query FAISS for similar jobs (MMR λ=0.5)

  - [x] Implement ranking strategies:
    - [x] MMR with λ=0.5 (balanced relevance/diversity) - **WINNER**
    - [x] Pure similarity (baseline)
    - [x] Skill-weighted reranking (tested but counterproductive)

- [x] **Experiment Log:** Test different ranking strategies (Experiment #6):
  - [x] Pure Similarity: 0.660 sim, 7.4% skill, 9/10 diversity
  - [x] MMR λ=0.5: 0.656 sim, 4.2% skill, **10/10 diversity** - **WINNER**
  - [x] Skill reranking: Hurt both similarity AND skill match

### Step 5.2: Implement Skill Gap Analysis ✓ COMPLETED
- [x] Create `skill_gap_analyzer.py`:
  - [x] Extract required skills from job posting (RAKE + granite4:micro)
  - [x] Extract candidate skills from resume (LLM extraction)
  - [x] Identify:
    - [x] Matched skills (highlight overlaps)
    - [x] Missing skills (gaps to address)
    - [x] Extra skills (candidate advantages)

  - [x] Generate skill match percentage and breakdown

- [x] **Experiment Log:** RAKE + Ollama skill extraction (Experiment #5):
  - [x] Tested 6 Ollama models for atomic skill extraction
  - [x] **Winner:** granite4:micro (3.75s, most atomic output)
  - [x] Key finding: Smaller models produce cleaner atomic skills

### Step 5.3: Build Recommendation Result Object ✓ COMPLETED
- [x] Create data structure for recommendation results:
  - [x] RecommendationResult dataclass with all required fields
  - [x] MatchingReport with skill profile summary
  - [x] Development recommendations

- [x] Return top K matches (configurable, default 10)
- [x] Include reasoning/explanation for each match

### Step 5.4: Create Comprehensive Output Report ✓ COMPLETED
- [x] Generate detailed matching report:
  - [x] Resume summary
  - [x] Top job matches ranked by relevance
  - [x] Overall skill profile summary
  - [x] Recommendations for skill development

- [x] Export formats:
  - [x] JSON (programmatic use)
  - [x] Markdown (readability)
  - [x] HTML (nice visualization)

- [x] **Learning Log:** Documented Phase 5 completion

**Phase 5 Summary:**
- **Matching Engine:** MatchingEngine class with MMR λ=0.5
- **Skill Extraction:** RAKE + granite4:micro for atomic skills
- **Report Generator:** JSON, MD, HTML formats
- **Full Pipeline Test:** 45s for complete matching (Harper Russo resume)

---

## Phase 6: Streamlit UI

### Step 6.1: Design Streamlit Application Layout ✓ COMPLETED
- [x] Create `app.py`:
  - [x] File upload component for PDF resumes
  - [x] Processing status indicator
  - [x] Results display sections:
    - [x] Resume summary overview
    - [x] Top job matches table/cards
    - [x] Skill match breakdown (bar chart)
    - [x] Skill gap visualization
    - [ ] AI-generated match insights (placeholder for Phase 7)

### Step 6.2: Implement Resume Upload & Processing ✓ COMPLETED
- [x] Handle file uploads:
  - [x] Validate PDF format and size
  - [x] Show processing progress
  - [x] Handle errors gracefully

- [x] Display parsing results:
  - [x] Show extracted resume data
  - [x] Confirm before proceeding to matching

### Step 6.3: Implement Interactive Results Display ✓ COMPLETED
- [x] Create tabs/sections for:
  - [x] Overview dashboard
  - [x] Detailed job matches with filtering/sorting
  - [x] Skill analysis and gap visualization
  - [ ] AI insights section (placeholder)
  - [x] Get Detailed Insights section (placeholder)
  - [x] Export options

- [x] Add interactivity:
  - [x] Filter/search by job title, company name etc.
  - [x] Click through for detailed job posting view
  - [x] Copy suggestions to clipboard
  - [x] Similarity threshold slider (default 60%)

### Step 6.4: Deploy & Optimize ✓ COMPLETED
- [x] Optimize for performance:
  - [x] Cache embeddings and FAISS index
  - [x] Lazy load heavy models
  - [x] Session state management

- [x] Add configuration panel:
  - [x] Top K results
  - [x] Similarity threshold (in Job Matches tab)

- [x] Deploy options:
  - [x] Local Streamlit server

### Step 6.5: Fix Issues & Improve UX
**Status:** In Progress

**Configuration Improvements:**
- [ ] Move similarity threshold from Job Matches tab to sidebar configuration
  - Currently: Filter slider in Job Matches tab
  - Target: Main sidebar with other config options
  - Default: x% where x is the 10th match

**Content Display Fixes:**
- [ ] Fix company/location information display
  - Issue: Showing raw dict `{'addressLine': 'Remote', 'city': '', ...}`
  - Target: Clean formatted string "Remote" or "Pittsburgh, PA"
  - Implementation: Parse location dict and extract meaningful fields

- [ ] Show full job description in Job Matches tab
  - Issue: Description truncated to 500 characters
  - Target: Full description in expandable section
  - Keep truncated preview in card, full text in expander

**Visualization Improvements:**
- [ ] Replace grouped bar chart in Overview tab
  - Issue: Not intuitive for comparing skill match vs similarity
  - Options to consider (apply all and ask user to select):
    - Scatter plot (x=similarity, y=skill match, color by company)
    - Table with sortable columns
    - Card grid with key metrics
    - Radar/spider chart for multi-dimensional view
  - Decision: TBD (discuss with user)

**AI Insights Placeholders (Phase 7 Preparation):**
- [ ] Add placeholder sections for AI-generated insights
  - Location 1: Job Matches tab - "AI Match Insights" expander (disabled/grayed out)
  - Location 2: Overview tab - "AI Recommendations" section (coming soon message)
  - Include informative message: "AI insights coming in Phase 7"
  - Prepare structure for future Ollama/Gemini integration

**Critical Fix: Skill Matching Algorithm:**
- [ ] Investigate current skill matching implementation
  - Problem: 0% skill matches for most resumes
  - Example mismatches:
    - Resume: "SQL database management" vs Job: "sql queries" (should match)
    - Resume: "Machine learning frameworks" vs Job: "machine learning" (should match)

- [ ] Discuss and implement improved matching strategy:
  - **Current:** Exact string matching (case-insensitive)
  - **Options: (Test all and document under experiment and decision log)**
    1. Fuzzy matching (Levenshtein distance, threshold ~80%)
    2. Semantic similarity (using sentence-transformers EmbeddingGemma)
    3. Keyword/n-gram overlap
    4. Hybrid approach (combine multiple methods)

- [ ] Test improved matching on sample resumes
  - Validate match quality improvement
  - Measure performance impact
  - Update skill extraction if needed

### Step 6.6: STRETCH GOAL – Async Streaming Results
- [ ] Implement async job matching:
  - [ ] Jobs appear on UI as they're identified (above threshold)
  - [ ] Results sorted dynamically as matches stream in
  - [ ] Progress indicator showing jobs processed / total

- [ ] Parallel AI suggestion generation:
  - [ ] Start Ollama suggestions in background for each identified job
  - [ ] Update UI as suggestions become available
  - [ ] Non-blocking UI during generation

- [ ] Technical implementation:
  - [ ] Use `asyncio` for concurrent processing
  - [ ] Streamlit callback/placeholder updates
  - [ ] Queue-based job processing

---

## Phase 7: AI-Generated Match Insights

### Step 7.1: Design AI Insights Module
- [ ] Create `ai_insights.py`:
  - [ ] Analyze resume against each matched job posting
  - [ ] Generate insights covering:
    - [ ] Overall match quality assessment
    - [ ] Key strengths for this specific role
    - [ ] Potential concerns or gaps
    - [ ] Why this candidate might be a good fit
    - [ ] Suggested talking points for interview

### Step 7.2: Implement Dual-Source AI Generation

#### Source 1: Fast Ollama Insights (Always Generated)
- [ ] Use fast local model (granite4:micro or similar)
- [ ] Auto-generate for all matches above threshold
- [ ] Focus on quick, factual observations:
  - [ ] Skill overlap summary
  - [ ] Experience relevance
  - [ ] Key matching points
- [ ] Target latency: <5 seconds per job

#### Source 2: Gemini API Deep Insights (On-Demand)
- [ ] Integrate Google Gemini API
- [ ] Generate only when user clicks "Get Detailed Insights" button
- [ ] Provide more comprehensive analysis:
  - [ ] Detailed match reasoning
  - [ ] Industry-specific advice
  - [ ] Career trajectory insights
  - [ ] Interview preparation tips
  - [ ] Suggested resume improvements for this role
- [ ] Handle API errors gracefully with fallback message

### Step 7.3: Integrate with Streamlit UI
- [ ] Display Ollama insights automatically:
  - [ ] Show in expandable section per job match
  - [ ] Loading indicator while generating

- [ ] Add "Get Detailed Insights" button:
  - [ ] Calls Gemini API on click
  - [ ] Shows loading spinner
  - [ ] Displays rich formatted response
  - [ ] Cache results to avoid re-generation

### Step 7.4: Experiment & Optimize
- [ ] **Experiment Log:** Compare insight generation:
  - [ ] Ollama models: granite4:micro vs other ollama models
  - [ ] Prompt engineering for best insights

- [ ] **Learning Log:** Document:
  - [ ] Quality of generated insights
  - [ ] User experience with dual-source approach

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

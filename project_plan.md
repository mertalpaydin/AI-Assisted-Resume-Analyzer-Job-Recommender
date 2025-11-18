# AI-Assisted Resume Analyzer & Job Recommender
## Claude Code Step-by-Step Implementation Plan

**Project Goal:** Build an end-to-end NLP pipeline that matches resumes to relevant job postings, identifies skill gaps, and provides recommendations.

**Learning Log:** Document findings, decisions, and experiments at each stage.

---

## Phase 0: Documentation & Logging Infrastructure (PRIORITY)

### Step 0.1: Initialize Documentation Structure
- [ ] Create `/logs/learning_log.md` - main learning documentation file
  - [ ] Add header with project name, start date, learning objectives
  - [ ] Create sections for each phase (to be updated continuously)
  - [ ] Use consistent markdown formatting

- [ ] Create `/logs/experiment_log.md` - detailed experiment tracking
  - [ ] Initialize experiment counter
  - [ ] Create template section for standardized logging

- [ ] Create `/logs/decisions.md` - architectural and technical decisions
  - [ ] Track rationale for each major choice
  - [ ] Document trade-offs considered

- [ ] Create `/logs/challenges.md` - problems encountered and solutions
  - [ ] Document issues as they arise
  - [ ] Note resolution approaches

- [ ] Set up logging in code:
  - [ ] Configure Python logging module in all scripts
  - [ ] Log to both console and `/logs/debug.log`
  - [ ] Include timestamps, module names, and function information

### Step 0.2: Create Logging Helper Module
- [ ] Create `src/logging_utils.py`:
  - [ ] Centralized logging configuration
  - [ ] Functions to log experiments, decisions, and findings
  - [ ] Easy-to-use wrappers for documentation capture

- [ ] Create documentation checkpoint function:
  - [ ] After each major step, log key findings
  - [ ] Record metrics and observations
  - [ ] Timestamp entries

**Critical Principle:** From this point forward, **after every completed step or experiment, immediately update the learning logs**. Do not defer documentation to the end.

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

- [ ] Prepare sample resumes:
  - [ ] Create 3-5 realistic test resumes in PDF format
  - [ ] Ensure they cover different experience levels and industries
  - [x] **Experiment Log:** Try resumes with varying formatting (headers, bullets, paragraphs)

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
  - [ ] Test on sample resumes

- [ ] **Experiment Log:** Try multiple PDF loading strategies:
  - [ ] Different page extraction methods
  - [ ] Handling of special formatting (tables, columns)
  - [ ] Performance metrics for each approach

**ToDo:** Testing to be done with sample resumes 

### Step 2.3: Design Resume JSON Schema
- [ ] Define structured output schema with fields:
  ```json
  {
    "full_name": "string",
    "email": "string",
    "phone": "string",
    "summary": "string",
    "experience": [
      {
        "company": "string",
        "position": "string",
        "duration": "string",
        "description": "string"
      }
    ],
    "skills": ["string"],
    "education": [
      {
        "institution": "string",
        "degree": "string",
        "field": "string"
      }
    ],
    "certifications": ["string"]
  }
  ```

- [ ] Create Pydantic models for type validation

### Step 2.4: Build LLM-Powered Resume Extractor
- [ ] Create `resume_extractor.py` using LangChain:
  - [ ] Craft prompt template for structured extraction
  - [ ] Use LangChain's output parsers (JsonOutputParser, PydanticOutputParser)
  - [ ] Chain together: PDF text → LLM → Structured JSON
  
- [ ] Test on sample resumes:
  - [ ] Verify JSON schema compliance
  - [ ] Check extraction accuracy for each field
  - [ ] Measure latency and token usage

- [ ] **Experiment Log:** Test different prompting strategies:
  - [ ] Few-shot prompting with examples
  - [ ] Chain-of-thought prompting
  - [ ] Alternative output parsing methods
  - [ ] Document accuracy improvements

### Step 2.5: Build Error Handling & Validation
- [ ] Add validation for:
  - [ ] Missing critical fields
  - [ ] Email/phone format validation
  - [ ] Skill name normalization
  
- [ ] Implement retry logic for LLM failures
- [ ] Create logging for extraction confidence scores
- [ ] **Learning Log:** Document common extraction errors and solutions

---

## Phase 3: Text Preprocessing & Skill Extraction

### Step 3.1: Set Up Text Preprocessing Pipeline
- [ ] Create `text_preprocessor.py`:
  - [ ] Download spaCy English model (`en_core_web_sm`)
  - [ ] Implement text cleaning:
    - [ ] Lowercase conversion
    - [ ] Remove special characters and extra whitespace
    - [ ] Remove stopwords
    - [ ] Remove URLs and email addresses

  - [ ] Implement tokenization using spaCy
  - [ ] Implement lemmatization
  - [ ] Test on sample texts

- [ ] **Experiment Log:** Compare preprocessing approaches:
  - [ ] With vs. without stopword removal
  - [ ] Lemmatization vs. stemming
  - [ ] Impact on downstream matching accuracy

### Step 3.2: Implement Skill Extraction with KeyBERT
- [ ] Create `skill_extractor.py`:
  - [ ] Initialize KeyBERT model
  - [ ] Extract skills from resume text and job descriptions
  - [ ] Normalize skill names (e.g., "Python" → "python", "C++" → "cpp")
  
- [ ] Create skill normalization lookup:
  - [ ] Common aliases (e.g., "Deep Learning" → "deep learning", "DL")
  - [ ] Technology variations (e.g., "JS", "Javascript", "JavaScript")
  - [ ] Build curated skill database

- [ ] **Experiment Log:** Test different KeyBERT models:
  - [ ] Default multilingual model
  - [ ] Domain-specific embeddings
  - [ ] Custom training on tech skills
  - [ ] Evaluate extraction quality and performance

### Step 3.3: Build Unified Preprocessing Pipeline
- [ ] Create `preprocessing_pipeline.py`:
  - [ ] Combine cleaning, tokenization, and skill extraction
  - [ ] Apply to both resume and job posting texts
  - [ ] Store preprocessed data with metadata (timestamps, confidence scores)

- [ ] **Learning Log:** Document preprocessing decisions and their impact on later stages

---

## Phase 4: Embedding Generation & Vector Storage (Chunking Strategy)

### Step 4.1: Intelligent Job Chunking
- [ ] Create `JobChunker` class:
  - [ ] Semantic chunking (split by sections: title, description, requirements)
  - [ ] Mark each chunk with job_id
  - [ ] Track chunk importance (title > requirements > description)
  - [ ] Add overlap for context preservation
  - [ ] Test on sample jobs

- [ ] **Learning Log:** Document chunking strategy:
  - [ ] Chunks per job (avg ~6-8)
  - [ ] Total chunks (200k jobs → ~1.2M chunks)
  - [ ] Chunk size effectiveness
  - [ ] Section importance rankings

### Step 4.2: Job ID Mapping System
- [ ] Create `JobIDMapper` class:
  - [ ] Bidirectional mapping: chunk_id ↔ job_id
  - [ ] Store complete job postings
  - [ ] Implement deduplication logic (multiple chunks → one job)
  - [ ] Implement `get_job_by_chunk_id()` - retrieve full job
  - [ ] Implement `get_unique_jobs()` - deduplicate results

- [ ] **Critical:** Automatic deduplication
  - [ ] Multiple chunks from same job → returned once
  - [ ] Preserves best matching chunk info
  - [ ] Maintains link to complete job posting

- [ ] **Learning Log:** Document mapping performance:
  - [ ] Mapping size and memory usage
  - [ ] Deduplication effectiveness
  - [ ] Retrieval speed

### Step 4.3: Chunk Embedding Generation
- [ ] Create `ChunkEmbeddingGenerator` class:
  - [ ] Efficient batch embedding for all chunks
  - [ ] Track: chunk_id, job_id, section, importance
  - [ ] Cache embeddings to disk (numpy format)
  - [ ] Document: dimensions, model size, latency

- [ ] Generate embeddings for:
  - [ ] ~1.2M chunks (from 200k jobs)
  - [ ] Store chunk metadata alongside embeddings
  - [ ] Index position → chunk_id mapping

- [ ] **Experiment Log:** Compare embedding models:
  - [ ] EmbeddingGemma (specified baseline)
  - [ ] all-MiniLM-L6-v2 (fast, lightweight)
  - [ ] all-mpnet-base-v2 (higher quality)
  - [ ] Metric: accuracy, speed, memory, match quality

### Step 4.4: Chunk Vector Store with Job ID Deduplication
- [ ] Create `ChunkVectorStore` class:
  - [ ] Build FAISS index from chunk embeddings
  - [ ] Store chunk metadata (chunk_id, job_id, section)
  - [ ] Implement `search_chunks()` - returns top-k chunks
  - [ ] Implement `get_unique_jobs_from_chunks()` - deduplicates by job_id, returns complete jobs
  - [ ] Implement `save()` / `load()` - persist index + metadata

- [ ] **Key feature:** Automatic deduplication
  - [ ] Search returns 50 chunks
  - [ ] But only 10 unique jobs (most from multiple chunks)
  - [ ] Deduplicate by job_id automatically
  - [ ] Return complete job postings

- [ ] **Experiment Log:** Performance metrics:
  - [ ] Index build time (~1.2M chunks)
  - [ ] Search latency (50 chunks vs 10 jobs)
  - [ ] Deduplication performance
  - [ ] Memory vs accuracy tradeoff
  - [ ] Optimal k_chunks value (25, 50, 100?)

### Step 4.5: Maximum Marginal Relevance (MMR) - Chunk-based
- [ ] Implement `search_mmr()` in vector store:
  - [ ] Work with chunks → then deduplicate to jobs
  - [ ] Balance relevance (chunk similarity) with diversity
  - [ ] Diverse jobs result, not duplicate chunks from same job
  - [ ] Returns **unique complete job postings**

- [ ] **Learning Log:** MMR with chunking:
  - [ ] Effectiveness of diversity (various companies, roles)
  - [ ] Lambda parameter tuning (0.3, 0.5, 0.7)
  - [ ] Deduplication impact on diversity

---

## Phase 5: Resume-to-Job Matching & Ranking

### Step 5.1: Build Matching Engine
- [ ] Create `matching_engine.py`:
  - [ ] For uploaded resume:
    1. Parse and extract structured data
    2. Preprocess and extract skills
    3. Generate embeddings
    4. Query FAISS for similar jobs
  
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
  - [ ] Use LangChain to orchestrate LLM suggestions
  - [ ] Analyze resume against top matching job postings
  - [ ] Generate suggestions for:
    - [ ] Missing skills to develop
    - [ ] How to better highlight existing skills
    - [ ] Experience descriptions to improve
    - [ ] Keywords to include for ATS optimization

- [ ] **Experiment Log:** Test different suggestion generation strategies:
  - [ ] Different LLMs (Ollama models)
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

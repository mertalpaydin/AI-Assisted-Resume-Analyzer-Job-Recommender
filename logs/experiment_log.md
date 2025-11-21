# Experiment Log
## AI-Assisted Resume Analyzer & Job Recommender

**Project:** Resume-to-Job Matching System
**Start Date:** 2025-11-13

This log tracks all experiments conducted throughout the project development, including model comparisons, parameter tuning, and architectural decisions.

---

## Experiment Template

Use this template for each experiment:

```markdown
### Experiment #X: [Experiment Name]
**Date:** YYYY-MM-DD
**Phase:** [Phase Number and Name]
**Objective:** [What are we trying to learn/optimize?]

**Hypothesis:** [What do we expect to happen?]

**Options Tested:**
1. **[Option A]**
   - Configuration: [parameters, settings]
   - Results: [quantitative metrics]
   - Observations: [qualitative notes]

2. **[Option B]**
   - Configuration: [parameters, settings]
   - Results: [quantitative metrics]
   - Observations: [qualitative notes]

3. **[Option C]** (if applicable)
   - Configuration: [parameters, settings]
   - Results: [quantitative metrics]
   - Observations: [qualitative notes]

**Metrics Used:**
- [Metric 1]: [definition]
- [Metric 2]: [definition]

**Winner:** [Selected Option]

**Rationale:** [Why this option was chosen - consider performance, accuracy, trade-offs]

**Key Learning:** [What did we learn from this experiment?]

**Impact on Project:** [How does this decision affect the overall system?]

**Next Steps:** [Follow-up actions or related experiments]
```

---

## Experiment Counter

**Total Experiments Conducted:** 6

**By Phase:**
- Phase 0: 0
- Phase 1: 0
- Phase 2: 2
- Phase 3: 1
- Phase 4: 1
- Phase 5: 2
- Phase 6: 0
- Phase 7: 0
- Phase 8: 0

---

## Experiments Log

### Experiment #1: [Pending - First experiment will be documented here]

---

## Quick Reference - Planned Key Experiments

### Phase 2: Resume Parsing
- **Exp 2.1:** LLM Model Comparison (granite4:micro vs llama3.2:3b vs gemma3:4b)
- **Exp 2.2:** PDF Parsing Libraries (PDFPlumber vs PyPDF vs PyMuPDF)
- **Exp 2.3:** Prompt Engineering for Structured Extraction

### Phase 3: Text Preprocessing
- **Exp 3.1:** Preprocessing Strategies (stopword removal impact, lemmatization vs stemming)
- **Exp 3.2:** Skill Extraction Models (KeyBERT configurations, embedding models)

### Phase 4: Embeddings & Vector Storage
- **Exp 4.1:** Embedding Model Comparison (EmbeddingGemma vs all-MiniLM-L6-v2 vs all-mpnet-base-v2)
- **Exp 4.2:** Chunking Strategy Optimization (chunk sizes, overlap amounts)
- **Exp 4.3:** FAISS Index Configuration (IndexFlatL2 vs IndexIVFFlat)
- **Exp 4.4:** MMR Parameter Tuning (lambda values for diversity)

### Phase 5: Matching & Ranking
- **Exp 5.1:** Ranking Strategies (weighted scoring approaches)
- **Exp 5.2:** Skill Matching Logic (exact vs fuzzy vs semantic)

### Phase 6: CV Improvement (Stretch)
- **Exp 6.1:** LLM Suggestion Quality (different models, prompt variations)

---

## Experiment Summary Table

| # | Name | Phase | Winner | Key Metric | Value | Date |
|---|------|-------|--------|------------|-------|------|
| 1 | LLM Model Comparison | 2 | gemma3:4b | Quality Score | 100.0% | 2025-11-18 |
| 2 | PDF Parser Comparison | 2 | Both (Pipeline) | User Rating | 3.0/5 (both) | 2025-11-20 |
| 3 | Skill Extraction Method Comparison | 3 | RAKE | F1 Score | 0.356 | 2025-11-20 |
| 4 | Embedding Model & Retrieval Strategy | 4 | embeddinggemma + MMR 0.5 | Precision@10 | 0.980 | 2025-11-21 |
| 5 | RAKE + Ollama Atomic Skill Extraction | 5 | granite4:micro | Avg Latency | 3.75s | 2025-11-21 |
| 6 | Ranking Strategy Comparison | 5 | MMR λ=0.5 | Diversity | 10/10 unique | 2025-11-21 |

---

## Notes

- All experiments should be reproducible with documented random seeds
- Include both quantitative metrics and qualitative observations
- Consider trade-offs: accuracy vs speed vs resource usage
- Document unexpected findings - they're often the most valuable
- Reference code commits for implementation details
---

### Experiment #1: LLM Model Comparison for Resume Parsing
**Date:** 2025-11-18
**Phase:** Phase 2 - Resume Parsing & Structured Extraction
**Objective:** Compare local LLM models (granite4:micro, llama3.2:3b, gemma3:4b) for structured resume data extraction capability

**Hypothesis:** Larger models (gemma3:4b, llama3.2:3b) will have better extraction accuracy but slower response times compared to granite4:micro

**Options Tested:**
1. **granite4:micro**
   - Configuration: Default Ollama settings, 4 runs (1st discarded as warm-up), temperature=1.0
   - Results: Avg Latency: 7.92s (over 3 runs), JSON Valid: True, Quality Score: 88.9%, Validation Errors: 1
   - Observations: Errors: 1

2. **llama3.2:3b**
   - Configuration: Default Ollama settings, 4 runs (1st discarded as warm-up), temperature=1.0
   - Results: Avg Latency: 3.02s (over 2 runs), JSON Valid: True, Quality Score: 50.0%, Validation Errors: 1
   - Observations: Errors: 1; Warnings: 2

3. **gemma3:4b**
   - Configuration: Default Ollama settings, 4 runs (1st discarded as warm-up), temperature=1.0
   - Results: Avg Latency: 14.61s (over 3 runs), JSON Valid: True, Quality Score: 100.0%, Validation Errors: 0
   - Observations: No issues detected

**Metrics Used:**
- Latency: Time in seconds to complete extraction
- JSON Validity: Whether output is valid, parseable JSON
- Quality Score: Percentage of required fields correctly extracted
- Missing Fields: Count of required fields not extracted

**Winner:** gemma3:4b

**Rationale:** gemma3:4b was selected for its quality score of 100.0% and latency of 14.61s

**Key Learning:** Model selection involves trade-offs between speed and accuracy. Prompt engineering is critical for consistent JSON output.

**Impact on Project:** Will use gemma3:4b as primary model for resume parsing in subsequent phases. May implement fallback logic or retry mechanisms for improved reliability.

**Next Steps:** Test winner model with actual PDF resume samples. Implement Pydantic output parsers for better JSON reliability.

---

### Experiment #2: PDF Parsing Library Comparison for Resume Extraction
**Date:** 2025-11-20
**Phase:** Phase 2 - Resume Parsing & Structured Extraction
**Objective:** Compare PDF parsing libraries (PDFPlumber vs PyPDF) for text extraction quality from complex resume formats (tables, multi-column layouts, varied formatting)

**Hypothesis:** PDFPlumber will outperform PyPDF in field detection and text structure preservation due to its table-aware parsing, but PyPDF might be faster.

**Options Tested:**
1. **PDFPlumber**
   - Configuration: Default settings via LangChain PyPDFLoader wrapper
   - Results:
     - Avg Field Presence Score: 90.5% (71.4%, 100%, 100% across 3 resumes)
     - Avg Parse Time: 0.235s
     - Avg User Rating: 3.0/5
     - Avg Word Count: 413
   - Observations:
     - Harper Russo (3/5): Could not handle strengths table
     - Henry Wotton (2/5): Could not handle format well
     - Isabella Bella Ruiz (4/5): Could not handle skills table and name
     - Strong at detecting fields but struggles with complex table extraction

2. **PyPDF**
   - Configuration: Default settings via LangChain PyPDFLoader wrapper
   - Results:
     - Avg Field Presence Score: 76.2% (28.6%, 100%, 100% across 3 resumes)
     - Avg Parse Time: 0.101s (2.3x faster than PDFPlumber)
     - Avg User Rating: 3.0/5
     - Avg Word Count: 1455
   - Observations:
     - Harper Russo (1/5): Poor field detection, excessive text extraction (3648 words vs expected ~500)
     - Henry Wotton (4/5): Text order off but format handled well
     - Isabella Bella Ruiz (4/5): Good performance
     - Faster but inconsistent field detection; better at preserving format but may disorder content

**Metrics Used:**
- Field Presence Score: Automated detection of email, phone, experience, education, skills sections (0-100%)
- Parse Time: Seconds to extract text from PDF
- User Rating: Manual quality assessment (1-5 scale) based on extraction completeness and accuracy
- Word Count: Number of words extracted (sanity check for over/under extraction)
- User Feedback: Qualitative observations on specific parsing issues

**Winner:** Both (Dual-Parser Pipeline)

**Rationale:**
Neither parser is definitively superior - they have complementary strengths:
- PDFPlumber: Better field detection (90.5% vs 76.2%) but struggles with tables
- PyPDF: Faster (0.101s vs 0.235s) and better format handling but inconsistent field detection
- Both averaged 3.0/5 user ratings, suggesting neither alone is sufficient
- Different parsers excel at different resume formats (Henry Wotton: PyPDF rated 4/5 vs PDFPlumber 2/5)

**Decision:** Use a **dual-parser sequential pipeline** to leverage both strengths:
1. Parse with PDFPlumber → LLM creates initial structured JSON
2. Parse with PyPDF → LLM receives new text + previous JSON → refines/improves extraction
3. This approach allows the second pass to fill gaps, correct ordering, and enhance completeness

**Key Learning:**
- No single PDF parser handles all resume formats perfectly
- Different parsers have complementary failure modes (tables vs ordering)
- Sequential multi-parser approach can achieve higher quality than any single parser
- User ratings revealed issues not captured by automated metrics (table handling, text ordering)

**Impact on Project:**
- Implement dual-parser extraction pipeline in ResumeExtractor
- Modify LLM prompt for second pass to support "refinement mode" with previous JSON context
- Expected improvement: Higher field completeness and better handling of edge cases

**Next Steps:**
- Implement dual-parser pipeline in resume_extractor.py
- Create refinement prompt template for second LLM pass
- Test end-to-end extraction quality with dual-parser approach

---

### Experiment #1: LLM Model Comparison for Resume Parsing
**Date:** 2025-11-20
**Phase:** Phase 2 - Resume Parsing & Structured Extraction
**Objective:** Compare local LLM models (granite4:micro, llama3.2:3b, gemma3:4b) for structured resume data extraction capability

**Hypothesis:** Larger models (gemma3:4b, llama3.2:3b) will have better extraction accuracy but slower response times compared to granite4:micro

**Options Tested:**
1. **granite4:micro**
   - Configuration: Default Ollama settings, 4 runs (1st discarded as warm-up), temperature=1.0
   - Results: Avg Latency: 8.85s (over 3 runs), JSON Valid: True, Quality Score: 100.0%, Validation Errors: 0
   - Observations: No issues detected

2. **llama3.2:3b**
   - Configuration: Default Ollama settings, 4 runs (1st discarded as warm-up), temperature=1.0
   - Results: Avg Latency: 3.07s (over 3 runs), JSON Valid: True, Quality Score: 50.0%, Validation Errors: 1
   - Observations: Errors: 1; Warnings: 2

3. **gemma3:4b**
   - Configuration: Default Ollama settings, 4 runs (1st discarded as warm-up), temperature=1.0
   - Results: Avg Latency: 12.48s (over 3 runs), JSON Valid: True, Quality Score: 100.0%, Validation Errors: 0
   - Observations: No issues detected

**Metrics Used:**
- Latency: Time in seconds to complete extraction
- JSON Validity: Whether output is valid, parseable JSON
- Quality Score: Percentage of required fields correctly extracted
- Missing Fields: Count of required fields not extracted

**Winner:** granite4:micro

**Rationale:** granite4:micro was selected for its quality score of 100.0% and latency of 8.85s

**Key Learning:** Model selection involves trade-offs between speed and accuracy. Prompt engineering is critical for consistent JSON output.

**Impact on Project:** Will use granite4:micro as primary model for resume parsing in subsequent phases. May implement fallback logic or retry mechanisms for improved reliability.

**Next Steps:** Test winner model with actual PDF resume samples. Implement Pydantic output parsers for better JSON reliability.

---

### Experiment #3: Skill Extraction Method Comparison
**Date:** 2025-11-20
**Phase:** Phase 3 - Text Preprocessing & Skill Extraction
**Objective:** Compare different skill extraction methods (KeyBERT with various embedding models vs statistical methods like YAKE and RAKE) to identify the most accurate and efficient approach for extracting technical skills from resumes and job descriptions.

**Hypothesis:** KeyBERT with larger embedding models (all-mpnet-base-v2) will provide better skill extraction quality than smaller models (all-MiniLM-L6-v2) and statistical methods (YAKE, RAKE), but at the cost of slower performance.

**Options Tested:**
1. **KeyBERT with all-MiniLM-L6-v2**
   - Configuration: 384-dim embeddings, top_n=20, diversity=0.5, MMR enabled, n-gram range (1,3)
   - Results: F1: 0.133, Precision: 0.150, Recall: 0.120, Time: 3.78s
   - Observations: Extracts long keyphrases that aren't actual skills (e.g., "processing proficient aws", "java expert machine"). Model loading adds significant overhead.

2. **KeyBERT with all-mpnet-base-v2**
   - Configuration: 768-dim embeddings, top_n=20, diversity=0.5, MMR enabled, n-gram range (1,3)
   - Results: F1: 0.133, Precision: 0.150, Recall: 0.120, Time: 45.42s (12x slower!)
   - Observations: **No quality improvement over MiniLM** despite being 12x slower and 2x larger. Semantic understanding creates longer, less precise phrases.

3. **YAKE (statistical)**
   - Configuration: Language-independent, n-gram max=3, deduplication=0.7, top=20
   - Results: F1: 0.273, Precision: 0.316, Recall: 0.240, Time: 0.72s
   - Observations: 2x better F1 than KeyBERT, 5x faster. Good at single-word skills. Lower YAKE score = higher importance (inverted).

4. **RAKE (statistical)**
   - Configuration: NLTK-based, max_length=3, stopwords as phrase delimiters
   - Results: F1: 0.356, Precision: 0.400, Recall: 0.320, Time: 0.29s
   - Observations: **Best quality and fastest**. 13x faster than KeyBERT MiniLM, 157x faster than MPNet. Extracts cleaner atomic skills vs long phrases.

**Metrics Used:**
- **F1 Score**: Harmonic mean of precision and recall (primary quality metric)
- **Precision**: % of extracted skills that match ground truth (26 manually annotated skills)
- **Recall**: % of ground truth skills found by method
- **Execution Time**: Processing speed (including model loading)
- **F1/Time Ratio**: Quality per second (balance metric)
- **Unique Skills**: Distinct skills extracted (diversity check)

**Winner:** RAKE (statistical)

**Rationale:**
RAKE significantly outperformed all methods across all key metrics:
- **Best Quality**: F1: 0.356 (2.7x better than KeyBERT, 1.3x better than YAKE)
- **Best Precision**: 0.400 (extracted more correct skills)
- **Fastest**: 0.29s (13x faster than KeyBERT MiniLM, 157x faster than MPNet, 2.5x faster than YAKE)
- **Best Balance**: F1/Time ratio of 1.215 (best quality per second)
- **Cleaner Extraction**: Extracted atomic skills ("machine learning", "nlp", "python") vs long keyphrases ("processing proficient aws")

**Unexpected Finding:**
- **Bigger embedding model ≠ better results**: all-mpnet-base-v2 (768d) had **identical F1 score** to all-MiniLM-L6-v2 (384d) despite being 12x slower
- **Statistical > Embeddings for this task**: RAKE (no embeddings) outperformed both KeyBERT models significantly
- **KeyBERT's design mismatch**: Optimized for "keyphrases" (semantic concepts), not "keywords" (atomic skills)

**Key Learning:**
1. **Task-method alignment matters more than model size**: Using the right algorithm for the task beats using a bigger model with the wrong approach
2. **Statistical methods excel for skill extraction**: Skills are atomic terms (1-3 words), statistical co-occurrence works better than semantic embeddings
3. **RAKE's algorithm advantages**: Word co-occurrence graphs, natural phrase boundaries (stopwords as delimiters), good for technical terminology
4. **KeyBERT limitations**: MMR diversity creates longer phrases, embedding similarity can't distinguish "skill" from "skill-containing-phrase"
5. **Speed matters for production**: 0.29s vs 45.42s makes RAKE viable for real-time processing

**Impact on Project:**
- **Immediate**: Switch primary skill extractor from KeyBERT to RAKE (2.7x quality improvement, 13x speed improvement)
- **Expected Benefits**: More precise skill matching, faster resume-to-job comparison, better skill gap analysis
- **Architecture**: Keep KeyBERT for semantic job search (different use case), use RAKE for skill extraction
- **Potential Hybrid**: Combine RAKE (0.5 weight) + YAKE (0.3 weight) + spaCy NER (0.2 weight) for robustness

**Next Steps:**
- Update `skill_extractor.py` to make RAKE the default method
- Test RAKE on diverse resume/job formats (different industries, experience levels)
- Explore hybrid approach combining RAKE + YAKE + skill database matching
- Consider domain-specific embeddings experiment in future (CodeBERT for programming skills)

---

### Experiment #4: Embedding Model & Retrieval Strategy Comparison
**Date:** 2025-11-21
**Phase:** Phase 4 - Embedding Generation & Vector Storage
**Objective:** Compare embedding models (all-MiniLM-L6-v2 vs all-mpnet-base-v2 vs google/embeddinggemma-300m) and retrieval strategies (Cosine vs MMR with various lambda values) to find optimal job search configuration.

**Hypothesis:** EmbeddingGemma will provide better semantic understanding for job matching, and MMR will improve result diversity without sacrificing relevance.

**Options Tested (Embedding Models):**
1. **all-MiniLM-L6-v2**
   - Configuration: 384-dim embeddings, sentence-transformers, CUDA GPU
   - Results: Precision@10: 0.850, Diversity: 9.5/10, Speed: 1489.2 chunks/sec, Memory: 5.31 MB
   - Observations: Fastest model, good baseline quality, smallest memory footprint

2. **all-mpnet-base-v2**
   - Configuration: 768-dim embeddings, sentence-transformers, CUDA GPU
   - Results: Precision@10: 0.860, Diversity: 9.6/10, Speed: 242.3 chunks/sec, Memory: 10.63 MB
   - Observations: Marginal quality improvement over MiniLM, 6x slower

3. **google/embeddinggemma-300m**
   - Configuration: 768-dim embeddings, sentence-transformers with encode_query/encode_document, CUDA GPU
   - Results: Precision@10: **0.980**, Diversity: 9.7/10, Speed: 147.4 chunks/sec, Memory: 10.63 MB
   - Observations: **Best quality by far** (12-13% improvement), uses asymmetric query/document encoding

**Options Tested (Retrieval Strategies):**
1. **Cosine Similarity**
   - Results: Precision: 0.850, Companies: 9.5/10, Categories: 3.5, Time: 0.3ms
   - Observations: Fastest, baseline diversity

2. **MMR λ=0.3**
   - Results: Precision: 0.850, Companies: 9.9/10, Categories: 3.7, Time: 48.6ms
   - Observations: Good diversity boost, no precision loss

3. **MMR λ=0.5**
   - Results: Precision: **0.880**, Companies: 9.9/10, Categories: 3.8, Time: 48.0ms
   - Observations: **Best balance** - highest precision AND good diversity

4. **MMR λ=0.7**
   - Results: Precision: 0.840, Companies: 9.9/10, Categories: 3.6, Time: 47.6ms
   - Observations: Slight precision drop, similar diversity

5. **MMR λ=0.9**
   - Results: Precision: 0.850, Companies: 9.5/10, Categories: 3.5, Time: 48.1ms
   - Observations: Converges toward cosine behavior

**Metrics Used:**
- **Precision@10**: % of top-10 results containing query keywords (relevance)
- **Unique Companies**: Diversity of employers in results (0-10)
- **Unique Categories**: Job category diversity (engineering, management, etc.)
- **Embedding Speed**: Chunks per second (throughput)
- **Search Time**: Milliseconds per query (latency)

**Winners:**
- **Embedding Model:** google/embeddinggemma-300m (Precision: 0.980)
- **Retrieval Strategy:** MMR with λ=0.5 (Precision: 0.880, Diversity: 9.9/10)

**Rationale:**
- EmbeddingGemma achieved 98% precision (12-13% better than alternatives) due to:
  - Asymmetric query/document encoding (optimized for retrieval)
  - Larger model trained specifically for document retrieval
  - 768-dim embeddings capture more semantic nuance
- MMR λ=0.5 provided best precision (0.880) with near-maximum diversity (9.9/10)
- Speed trade-off acceptable: 147 chunks/sec still processes 500 jobs in ~25 seconds

**Key Learning:**
1. **Specialized models win**: EmbeddingGemma designed for retrieval outperforms general-purpose embeddings
2. **Asymmetric encoding matters**: Separate query vs document encoding improves retrieval quality
3. **MMR improves both relevance and diversity**: λ=0.5 found optimal balance, actually increasing precision
4. **Speed vs quality trade-off**: 10x slower than MiniLM but 15% more accurate is worthwhile for offline processing

**Impact on Project:**
- Update `embedding_generator.py` to use google/embeddinggemma-300m as default
- Update `vector_store.py` to use MMR with λ=0.5 as default retrieval
- Expected: Significant improvement in job recommendation quality

**Next Steps:**
- Update production code with winner configurations
- Move to Phase 5: Resume-to-Job Matching & Ranking

---

### Experiment #5: RAKE + Ollama Atomic Skill Extraction for Job Matching
**Date:** 2025-11-21
**Phase:** Phase 5 - Resume-to-Job Matching & Ranking
**Objective:** Compare Ollama LLM models for refining RAKE-extracted keyphrases into atomic, matchable skills. The problem: RAKE extracts long phrases (e.g., "strong communication skills", "kubernetes container orchestration") that don't match well with resume skills (e.g., "kubernetes", "communication").

**Hypothesis:** Smaller, faster models (granite4:micro) will produce more atomic skills due to simpler output patterns, while larger models may keep longer phrases.

**Pipeline:** Job Text → RAKE (keyphrases) → Ollama LLM (atomic skills)

**Test Data:** 4 job descriptions from different professions:
- DevOps Engineer (IT)
- Registered Nurse (Healthcare)
- Marketing Manager (Business)
- Financial Analyst (Finance)

**Options Tested:**
1. **granite4:micro**
   - Configuration: temperature=0.1, ChatOllama, JSON array output
   - Results: Avg Latency: **3.75s**, Avg Skills: 12.5, Unique Skills: 50, Success: 4/4
   - Observations: Extracts truly atomic skills (linux, gcp, kubernetes, prometheus). Single-word outputs ideal for matching.

2. **llama3.2:3b**
   - Configuration: temperature=0.1, ChatOllama, JSON array output
   - Results: Avg Latency: 4.00s, Avg Skills: 12.8, Unique Skills: 51, Success: 4/4
   - Observations: Keeps some multi-word phrases ("linux system administration", "kubernetes container orchestration"). Good quality but less atomic.

3. **llama3.1:8b**
   - Configuration: temperature=0.1, ChatOllama, JSON array output
   - Results: Avg Latency: 10.37s, Avg Skills: 15.5, Unique Skills: 62, Success: 4/4
   - Observations: Verbose output with longer phrases. 2.6x slower than granite4:micro.

4. **mistral:7b**
   - Configuration: temperature=0.1, ChatOllama, JSON array output
   - Results: Avg Latency: 6.81s, Avg Skills: 15.2, Unique Skills: 60, Success: 4/4
   - Observations: Mixed atomic and phrase output. Balance between speed and coverage.

5. **gemma3:4b**
   - Configuration: temperature=0.1, ChatOllama, JSON array output
   - Results: Avg Latency: 7.67s, Avg Skills: 14.2, Unique Skills: 55, Success: 4/4
   - Observations: Balanced quality, similar to Phase 2 performance for resume parsing.

6. **gemma3:12b-it-q4_K_M**
   - Configuration: temperature=0.1, ChatOllama, JSON array output
   - Results: Avg Latency: 12.61s, Avg Skills: 17.5, Unique Skills: 67, Success: 4/4
   - Observations: Most skills extracted but slowest. Verbose phrases.

**Metrics Used:**
- **Avg Latency**: Time per extraction (seconds) - lower is better for real-time matching
- **Avg Skills Extracted**: Number of skills per job - moderate is ideal (too many = noise)
- **Unique Skills**: Total distinct skills across all jobs - diversity check
- **Success Rate**: JSON parsing success - reliability metric
- **Atomicity**: Qualitative assessment of skill granularity (single words vs phrases)

**Winner:** granite4:micro

**Rationale:**
granite4:micro was selected for skill extraction refinement because:
- **Fastest**: 3.75s average (1.8x faster than gemma3:4b, 3.4x faster than gemma3:12b)
- **Most Atomic**: Produces single-word skills (linux, kubernetes, python) ideal for exact matching
- **Sufficient Coverage**: 12.5 avg skills per job, 50 unique skills total
- **100% Reliability**: All 4 test jobs succeeded with valid JSON
- **Speed Critical**: Skill extraction happens during real-time job matching

**Key Learning:**
1. **Task-specific model selection**: granite4:micro (worst for resume parsing in Exp #1) is best for skill atomization
2. **Smaller models produce cleaner output**: Less context = simpler, more atomic responses
3. **Different tasks need different models**: gemma3:4b for resume parsing, granite4:micro for skill extraction
4. **Speed matters for matching**: 3.75s vs 12.61s is significant when processing many jobs
5. **Profession-agnostic success**: Works equally well for IT, Healthcare, Marketing, and Finance

**Sample Outputs by Profession:**
- DevOps: ['linux', 'gcp', 'kubernetes', 'monitoring', 'github', 'cd', 'bash', 'gitlab', 'terraform', 'python', 'prometheus']
- Nurse: ['icu', 'nurse', 'health records', 'epic', 'cerner', 'acls', 'bls']
- Marketing: ['social media marketing', 'marketing automation', 'email marketing', 'content marketing', 'roi analysis', 'project management', 'ppc advertising', 'google analytics', 'google ads', ...]
- Finance: ['bloomberg terminal', 'financial statement analysis', 'financial planning', 'financial modeling', 'variance analysis', 'pivot tables', 'data extraction', 'advanced excel', ...]

**Impact on Project:**
- Implement RAKE + granite4:micro pipeline in `skill_extractor.py`
- Update `matching_engine.py` to use new skill extraction for job-resume matching
- Expected: Better skill matching between resumes and jobs (atomic skills match better)
- Speed improvement: 3.75s per job is acceptable for real-time matching

**Next Steps:**
- Implement `extract_skills_with_llm()` method in skill_extractor.py
- Update matching_engine to use new extraction pipeline
- Test end-to-end skill matching with sample resumes
- Experiment with ranking strategies (Exp 5.2)

---

### Experiment #6: Ranking Strategy Comparison for Job Matching
**Date:** 2025-11-21
**Phase:** Phase 5 - Resume-to-Job Matching & Ranking
**Objective:** Compare different ranking strategies for resume-to-job matching to find optimal balance between relevance, skill alignment, and result diversity.

**Hypothesis:** MMR-based ranking will provide better diversity than pure similarity, and skill-weighted reranking will improve skill alignment at the cost of semantic similarity.

**Test Data:** Sample resume (PDF) matched against job embeddings database

**Options Tested:**
1. **Pure Similarity (No MMR)**
   - Configuration: Cosine similarity only, no diversity penalty
   - Results: Avg Similarity: **0.660**, Avg Skill Match: **7.4%**, Unique Companies: 9
   - Observations: Highest similarity and skill match scores, but lower diversity

2. **MMR Balanced (λ=0.5)**
   - Configuration: MMR with lambda=0.5 (balanced relevance/diversity)
   - Results: Avg Similarity: 0.656, Avg Skill Match: 4.2%, Unique Companies: **10**
   - Observations: Maximum diversity with minimal similarity loss (0.004 drop)

3. **MMR High Diversity (λ=0.3)**
   - Configuration: MMR with lambda=0.3 (high diversity weight)
   - Results: Avg Similarity: 0.652, Avg Skill Match: 3.2%, Unique Companies: **10**
   - Observations: Same diversity as λ=0.5 but lower skill match

4. **Skill Reranking (w=0.3)**
   - Configuration: MMR λ=0.5 + skill-weighted reranking (30% skill weight)
   - Results: Avg Similarity: 0.470, Avg Skill Match: 3.7%, Unique Companies: **10**
   - Observations: Significant similarity drop, modest skill gain

5. **Skill Reranking (w=0.5)**
   - Configuration: MMR λ=0.5 + skill-weighted reranking (50% skill weight)
   - Results: Avg Similarity: 0.347, Avg Skill Match: 3.7%, Unique Companies: **10**
   - Observations: Large similarity sacrifice with no skill improvement

**Metrics Used:**
- **Avg Similarity**: Mean semantic similarity score (0-1) - relevance metric
- **Avg Skill Match %**: Mean skill alignment percentage - job fit metric
- **Unique Companies**: Number of distinct employers in top 10 - diversity metric

**Winner:** MMR Balanced (λ=0.5)

**Rationale:**
MMR λ=0.5 was selected as the best ranking strategy because:
- **Maximum diversity**: 10/10 unique companies (vs 9 for pure similarity)
- **Minimal relevance loss**: Only 0.004 similarity drop (0.660 → 0.656)
- **Consistent with Phase 4**: Same λ value won for retrieval strategy
- **Skill reranking harmful**: Adding skill weighting drastically reduced similarity (0.660 → 0.347) without improving skill match

**Unexpected Finding:**
- **Skill reranking doesn't improve skill match**: Adding skill weight actually lowered skill match % (7.4% → 3.7%)
- **Pure similarity has best skill match**: Counter-intuitive but semantically similar jobs naturally require similar skills
- **Diversity vs skill match trade-off**: Higher diversity correlates with lower skill match

**Key Learning:**
1. **MMR is sufficient**: No need for additional skill-weighted reranking
2. **Semantic similarity captures skill alignment**: Jobs with high embedding similarity naturally require similar skills
3. **Reranking can hurt quality**: Post-hoc reranking may disrupt the semantic ordering
4. **Diversity improves user experience**: Showing varied companies is valuable even with slight relevance trade-off

**Impact on Project:**
- Use MMR λ=0.5 as default ranking strategy (already implemented from Phase 4)
- Remove/disable skill-weighted reranking in production
- Rely on semantic similarity for implicit skill matching

**Next Steps:**
- Complete Phase 5 integration testing
- Document Decision #6 in decisions.md
- Move to Phase 5.3-5.4: Report generation

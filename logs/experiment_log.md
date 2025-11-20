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

**Total Experiments Conducted:** 2

**By Phase:**
- Phase 0: 0
- Phase 1: 0
- Phase 2: 2
- Phase 3: 0
- Phase 4: 0
- Phase 5: 0
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

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

**Total Experiments Conducted:** 0

**By Phase:**
- Phase 0: 0
- Phase 1: 0
- Phase 2: 0
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
| - | - | - | - | - | - | - |

*(Table will be populated as experiments are completed)*

---

## Notes

- All experiments should be reproducible with documented random seeds
- Include both quantitative metrics and qualitative observations
- Consider trade-offs: accuracy vs speed vs resource usage
- Document unexpected findings - they're often the most valuable
- Reference code commits for implementation details
---

### Experiment #0: Test Experiment
**Date:** 2025-11-13
**Phase:** Phase 0 - Testing
**Objective:** Verify logging utilities work correctly

**Hypothesis:** Logging functions will create proper markdown entries

**Options Tested:**
1. **Option A**
   - Configuration: Test config
   - Results: Success
   - Observations: Works as expected

**Metrics Used:**
- Success Rate: Whether function executes without error

**Winner:** Option A

**Rationale:** Only option tested

**Key Learning:** Logging utilities are functional

**Impact on Project:** Can now track experiments throughout project


**Next Steps:** Use in actual experiments

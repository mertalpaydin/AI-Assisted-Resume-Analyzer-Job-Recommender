# Technical & Architectural Decisions
## AI-Assisted Resume Analyzer & Job Recommender

**Project:** Resume-to-Job Matching System
**Start Date:** 2025-11-13

This document tracks all major technical and architectural decisions made throughout the project, including the rationale and trade-offs considered.

---

## Decision Template

```markdown
### Decision #X: [Decision Title]
**Date:** YYYY-MM-DD
**Phase:** [Phase Number]
**Decision Maker:** [Your name or "Team"]
**Status:** [Proposed / Accepted / Superseded]

**Context:**
[What situation led to this decision? What problem are we solving?]

**Decision:**
[What did we decide to do?]

**Alternatives Considered:**
1. **[Option A]**
   - Pros: [advantages]
   - Cons: [disadvantages]

2. **[Option B]**
   - Pros: [advantages]
   - Cons: [disadvantages]

3. **[Option C]** (if applicable)
   - Pros: [advantages]
   - Cons: [disadvantages]

**Rationale:**
[Why was this option chosen over the alternatives?]

**Trade-offs:**
- [Trade-off 1]: [description]
- [Trade-off 2]: [description]

**Consequences:**
- Positive: [expected benefits]
- Negative: [expected costs or limitations]
- Risks: [potential issues to monitor]

**Related Decisions:**
- [Links to related decision records]

**Review Date:** [When should we revisit this decision?]
```

---

## Decisions Log

### Decision #1: Select gemma3:4b as Primary LLM for Resume Parsing
**Date:** 2025-11-18
**Phase:** Phase 2 - Resume Parsing & Structured Extraction
**Status:** Accepted

**Context:**
Need to select a local LLM model for extracting structured data from resume PDFs. Three models were available via Ollama: granite4:micro (2.1GB), llama3.2:3b (2.0GB), and gemma3:4b (3.3GB). Requirements include high accuracy extraction, consistent output, handling of edge cases (multiple emails, promotions, complex formatting), and reasonable latency (<20s).

**Decision:**
Use **gemma3:4b** as the primary LLM model for resume parsing operations.

**Alternatives Considered:**
1. **granite4:micro**
   - Pros: Fast (7.92s avg latency), good quality (88.9%), smallest model size, reasonable accuracy
   - Cons: Inconsistent across runs, 11.1% error rate, struggles with edge cases

2. **llama3.2:3b**
   - Pros: Fastest (3.02s avg latency), smallest inference time
   - Cons: Poor quality (50% accuracy), unreliable (1/4 runs failed to produce valid JSON), many validation errors, insufficient skill extraction

3. **gemma3:4b**
   - Pros: Perfect quality (100%), zero validation errors, perfect consistency (4/4 runs scored 100%), handles all edge cases, no hallucinations
   - Cons: Slowest (14.61s avg latency), larger model size

**Rationale:**
gemma3:4b was selected despite being the slowest model because:
- **Perfect accuracy** (100%) is critical for production reliability - cannot afford 11-50% error rates
- **Zero validation errors** across all test runs demonstrates robustness
- **Perfect consistency** (all 4 runs scored 100%) shows reliability under varying conditions
- **Handles complex edge cases** (multiple emails, promotions, 40+ skills) that other models failed
- The 14.61s latency is acceptable for a resume parsing operation (not user-interactive)
- Extra ~7 seconds compared to granite4:micro is worth the 11.1% accuracy improvement
- Production systems prioritize correctness over speed for non-realtime operations

**Trade-offs:**
- Speed vs Accuracy: Accepted 1.8x slower performance for perfect accuracy
- Model Size vs Quality: Larger model (3.3GB vs 2.0-2.1GB) justified by perfect extraction
- Resource Usage: Higher memory/compute justified by zero error rate

**Consequences:**
- Positive: Perfect resume parsing reliability, no failed extractions, consistent structured output, handles all edge cases
- Negative: Slower response time (14.6s vs 7.9s vs 3.0s), higher memory usage, longer model load time
- Risks: May need optimization for high-volume processing, consider caching or batch processing strategies

**Related Decisions:**
- Experiment #1: LLM Model Comparison (logs/experiment_log.md)
- Decision will inform Phase 2 Step 2.4 (Resume Extractor implementation)

**Review Date:** After completing Phase 2.4 - may consider hybrid approach (fast model with gemma3:4b fallback) if speed becomes critical

---

### Decision #2: Implement Dual-Parser Sequential Pipeline for PDF Extraction
**Date:** 2025-11-20
**Phase:** Phase 2 - Resume Parsing & Structured Extraction
**Status:** Accepted

**Context:**
After testing PDFPlumber and PyPDF on 3 challenging resume samples with varied formats (tables, multi-column layouts), both parsers showed complementary strengths and weaknesses. PDFPlumber achieved 90.5% field detection but struggled with table extraction. PyPDF was 2.3x faster (0.101s vs 0.235s) and better at format handling but had inconsistent field detection (76.2%). Both received identical user ratings (3.0/5), indicating neither alone provides sufficient quality for production use.

**Decision:**
Implement a **dual-parser sequential pipeline** where both parsers are used in succession:
1. **First pass**: Parse PDF with PDFPlumber → LLM extracts initial structured JSON
2. **Second pass**: Parse PDF with PyPDF → LLM receives new text + previous JSON → refines/improves extraction

**Alternatives Considered:**
1. **PDFPlumber only**
   - Pros: Better field detection (90.5%), table-aware parsing, structured extraction
   - Cons: Struggles with complex tables (user feedback: "could not handle strengths table", "could not handle skills table"), slower (0.235s), missed formatting nuances

2. **PyPDF only**
   - Pros: Faster (0.101s, 2.3x speedup), better format handling, good text preservation
   - Cons: Inconsistent field detection (76.2%), text ordering issues, poor performance on Harper Russo resume (1/5 rating, 3648 words extracted vs expected ~500)

3. **Parallel dual-parser with voting/merging**
   - Pros: Could leverage both parsers simultaneously
   - Cons: Complex merging logic, potential conflicts, no clear resolution strategy for disagreements, wouldn't benefit from sequential refinement

4. **Dynamic parser selection based on PDF characteristics**
   - Pros: Could optimize for specific resume formats
   - Cons: Requires upfront classification, may misclassify, doesn't combine strengths of both parsers

**Rationale:**
The sequential dual-parser approach was selected because:
- **Complementary strengths**: PDFPlumber's field detection + PyPDF's format handling = comprehensive extraction
- **Refinement paradigm**: Second LLM pass can fill gaps, correct errors, and improve completeness using context from first pass
- **Handles edge cases**: Different parsers excel at different formats (e.g., Henry Wotton: PyPDF 4/5 vs PDFPlumber 2/5)
- **No clear winner**: Both parsers tied at 3.0/5 user rating, suggesting both contribute value
- **LLM-powered fusion**: LLM can intelligently merge information from both sources rather than hard-coded logic
- **Minimal cost**: Additional 0.1s parse time + ~15s LLM time is acceptable for offline resume processing
- **User feedback validation**: Issues like "could not handle table" (PDFPlumber) and "order off" (PyPDF) are addressed by combining both

**Trade-offs:**
- Speed vs Quality: Accept 2x parse time (0.235s + 0.101s = 0.336s) + 2x LLM calls (~30s total) for improved extraction quality
- Complexity vs Robustness: More complex pipeline but significantly more robust to varied resume formats
- Cost vs Accuracy: Double PDF parsing + LLM calls, but critical for production-quality extraction

**Consequences:**
- Positive:
  - Higher field completeness and accuracy across diverse resume formats
  - Better table handling (PyPDF can supplement PDFPlumber's table gaps)
  - Better text ordering (PDFPlumber can supplement PyPDF's ordering issues)
  - More robust to format variations (leverages strengths of both parsers)
  - LLM can intelligently reconcile differences and prefer better extraction

- Negative:
  - ~0.1s additional parse time (minor impact)
  - ~15s additional LLM inference time (acceptable for offline processing)
  - Increased complexity in ResumeExtractor implementation
  - Need to design refinement prompt carefully

- Risks:
  - Second LLM pass could introduce errors if not prompted correctly
  - May need prompt engineering to prevent second pass from overwriting correct first-pass data
  - Potential for information conflicts between parsers (mitigation: instruct LLM to prefer more complete/coherent data)

**Related Decisions:**
- Decision #1: Select gemma3:4b as Primary LLM (used for both passes)
- Experiment #2: PDF Parser Comparison (logs/experiment_log.md)

**Review Date:** After Phase 2 completion - evaluate extraction quality improvement, consider optimizations if latency becomes an issue

---

## Decision Summary Table

| # | Title | Phase | Status | Date | Impact |
|---|-------|-------|--------|------|--------|
| 1 | Select gemma3:4b as Primary LLM for Resume Parsing | 2 | Accepted | 2025-11-18 | High |
| 2 | Implement Dual-Parser Sequential Pipeline for PDF Extraction | 2 | Accepted | 2025-11-20 | High |


---

## Decisions to Make (Future)

### Phase 2
- [x] ~~PDF parsing library selection (after Experiment 2.2)~~ - COMPLETED: Dual-parser pipeline (Decision #2)
- [x] ~~LLM model selection for resume parsing (after Experiment 2.1)~~ - COMPLETED: gemma3:4b (Decision #1)
- [x] ~~Output parser approach (Pydantic vs JSON)~~ - COMPLETED: Using Pydantic (implemented in Step 2.3)

### Phase 3
- [ ] spaCy model size (sm vs md vs lg)
- [ ] Skill extraction approach (KeyBERT vs alternatives)

### Phase 4
- [ ] Embedding model selection (after Experiment 4.1)
- [ ] FAISS index type (after Experiment 4.3)
- [ ] MMR lambda parameter (after Experiment 4.4)

### Phase 5
- [ ] Ranking strategy (after Experiment 5.1)
- [ ] Skill matching logic (after Experiment 5.2)

---

## Notes

- All major technical decisions should be documented before implementation
- Include evidence from experiments when available
- Review and update status as project evolves
- Link to relevant experiment logs for data-driven decisions
- Consider impact on: performance, maintainability, scalability, cost
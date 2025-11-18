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

## Decision Summary Table

| # | Title | Phase | Status | Date | Impact |
|---|-------|-------|--------|------|--------|
| 1 | Select gemma3:4b as Primary LLM for Resume Parsing | 2 | Accepted | 2025-11-18 | High |


---

## Decisions to Make (Future)

### Phase 2
- [ ] PDF parsing library selection (after Experiment 2.2)
- [x] ~~LLM model selection for resume parsing (after Experiment 2.1)~~ - COMPLETED: gemma3:4b (Decision #1)
- [ ] Output parser approach (Pydantic vs JSON)

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
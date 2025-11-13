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

### Decision #1: Use Markdown for Documentation
**Date:** 2025-11-13
**Phase:** Phase 0 - Documentation Infrastructure
**Status:** Accepted

**Context:**
Need to establish a documentation format that is human-readable, version-control friendly, and supports structured learning capture throughout the project.

**Decision:**
Use Markdown (.md) format for all project documentation including learning logs, experiment logs, and decision records.

**Alternatives Considered:**
1. **Markdown**
   - Pros: Plain text, version-control friendly, widely supported, readable without tools
   - Cons: Limited formatting options compared to rich text

2. **Jupyter Notebooks**
   - Pros: Can embed code and visualizations, interactive
   - Cons: JSON-based format difficult to diff, not ideal for pure documentation

3. **Google Docs / Word**
   - Pros: Rich formatting, comments, collaboration features
   - Cons: Not version-control friendly, requires external tools

**Rationale:**
Markdown provides the best balance of simplicity, readability, and version control integration. Perfect for documentation that will live alongside code in a Git repository.

**Trade-offs:**
- Accept limited formatting in exchange for better version control and portability
- Can't embed interactive elements, but documentation focus is on text

**Consequences:**
- Positive: Documentation can be reviewed in pull requests, easily searchable, portable
- Negative: Need external tools for diagrams (can link to image files)
- Risks: None significant

**Related Decisions:**
- Decision #2: Git-based version control

---

### Decision #2: Chunking Strategy for Job Postings
**Date:** 2025-11-13
**Phase:** Phase 4 - Planning
**Status:** Accepted

**Context:**
Job postings vary significantly in length (some have extensive descriptions). Need to decide whether to embed entire job postings as single vectors or split them into semantic chunks.

**Decision:**
Implement intelligent chunking strategy with job_id mapping:
- Split each job posting into ~6-8 semantic chunks (title, description, requirements, etc.)
- Maintain bidirectional mapping between chunks and complete job postings
- Automatically deduplicate results (multiple chunks → single unique job)

**Alternatives Considered:**
1. **Chunking with job_id mapping** (chosen)
   - Pros: Better matching precision, handles long descriptions, captures nuanced requirements
   - Cons: More complex implementation, larger vector store (~1.2M chunks vs 200K jobs)

2. **Single embedding per job**
   - Pros: Simpler implementation, smaller vector store, faster initial setup
   - Cons: Loss of precision for long descriptions, difficulty capturing specific sections

3. **Hybrid approach** (summary embedding + detail chunks)
   - Pros: Balanced approach
   - Cons: Most complex, unclear benefit over pure chunking

**Rationale:**
Chunking provides significantly better matching accuracy for detailed job descriptions. The additional complexity of job_id mapping is manageable and provides automatic deduplication. FAISS can easily handle 1.2M vectors.

**Trade-offs:**
- Accept increased vector store size for better matching accuracy
- Accept implementation complexity for automatic deduplication
- Gain: chunk-level precision, better handling of long descriptions

**Consequences:**
- Positive: More accurate job matches, captures nuanced requirements, scalable
- Negative: Requires job_id mapping implementation, slightly more memory usage
- Risks: Must ensure deduplication works correctly to avoid showing duplicate jobs

**Related Decisions:**
- Decision on FAISS index type (Phase 4)
- Decision on MMR parameters (Phase 4)

---

### Decision #3: Local LLMs via Ollama
**Date:** 2025-11-13
**Phase:** Phase 0 - Planning
**Status:** Accepted

**Context:**
Need to choose between cloud-based LLMs (OpenAI, Anthropic) vs local LLMs for resume parsing and CV improvement suggestions. Privacy, cost, and learning objectives are key considerations.

**Decision:**
Use local LLMs via Ollama (granite4:micro, llama3.2:3b, gemma3:4b) for all LLM tasks.

**Alternatives Considered:**
1. **Local LLMs via Ollama** (chosen)
   - Pros: No API costs, privacy (resumes stay local), learning opportunity, offline capability
   - Cons: Requires local compute, may have lower quality than GPT-4

2. **Cloud LLMs (OpenAI GPT-3.5/4)**
   - Pros: Higher quality, no local compute needed, simpler setup
   - Cons: API costs, privacy concerns with resumes, requires internet

3. **Hybrid approach**
   - Pros: Use local for development, cloud for production
   - Cons: Complex, two codepaths to maintain

**Rationale:**
This is a learning project focused on understanding LLM capabilities. Local models provide hands-on experience with model comparison, while keeping resume data private and avoiding API costs.

**Trade-offs:**
- Accept potentially lower quality outputs for privacy and learning benefits
- Need to invest time in local setup and model comparison
- Gain: deeper understanding of LLM behavior, no ongoing costs

**Consequences:**
- Positive: Complete privacy, no API costs, learn model differences, portable
- Negative: Requires GPU/CPU resources, may need prompt engineering iteration
- Risks: Model quality may require fallback strategies

**Related Decisions:**
- Experiment 2.1: LLM Model Comparison

---

## Decision Summary Table

| # | Title | Phase | Status | Date | Impact |
|---|-------|-------|--------|------|--------|
| 1 | Use Markdown for Documentation | 0 | Accepted | 2025-11-13 | Low |
| 2 | Chunking Strategy for Job Postings | 4 | Accepted | 2025-11-13 | High |
| 3 | Local LLMs via Ollama | 0 | Accepted | 2025-11-13 | High |

---

## Decisions to Make (Future)

### Phase 1
- [ ] Python package manager (pip vs uv vs poetry)
- [ ] Virtual environment approach

### Phase 2
- [ ] PDF parsing library selection (after Experiment 2.2)
- [ ] LLM model selection for resume parsing (after Experiment 2.1)
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
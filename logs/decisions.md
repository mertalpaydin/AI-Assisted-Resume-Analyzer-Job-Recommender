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

### Decision #3: Use RAKE as Primary Skill Extraction Method
**Date:** 2025-11-20
**Phase:** Phase 3 - Text Preprocessing & Skill Extraction
**Status:** Accepted

**Context:**
Need to select a skill extraction method for parsing technical skills from resumes and job descriptions. Initial implementation used KeyBERT (with all-MiniLM-L6-v2 embeddings), but comprehensive testing revealed significant issues with extraction quality. Experiment #3 compared 4 methods: KeyBERT with two embedding models (all-MiniLM-L6-v2, all-mpnet-base-v2) and two statistical methods (YAKE, RAKE) on a sample resume with 26 ground truth skills.

**Decision:**
Use **RAKE (Rapid Automatic Keyword Extraction)** as the primary skill extraction method, replacing KeyBERT.

**Alternatives Considered:**
1. **KeyBERT with all-MiniLM-L6-v2 (current)**
   - Pros: Semantic understanding, configurable n-grams, MMR diversity
   - Cons: Poor F1 score (0.133), extracts long non-skill phrases ("processing proficient aws"), slow (3.78s), only 15% precision

2. **KeyBERT with all-mpnet-base-v2**
   - Pros: Higher-quality 768-dim embeddings, better semantic representation
   - Cons: **Identical F1 to MiniLM** (0.133) despite being 12x slower (45.42s), no quality gain from larger model

3. **YAKE (statistical)**
   - Pros: 2x better F1 than KeyBERT (0.273), 5x faster (0.72s), language-independent, good for single-word skills
   - Cons: Lower precision than RAKE (0.316 vs 0.400), 2.5x slower than RAKE

4. **RAKE (statistical)**
   - Pros: **Best F1 score (0.356)**, best precision (0.400), **fastest (0.29s)**, extracts atomic skills, no model loading overhead
   - Cons: NLTK dependency (minimal), may need refinement for highly specialized terminology

**Rationale:**
RAKE was selected based on overwhelming experimental evidence:
- **Best Quality**: F1 of 0.356 (2.7x better than KeyBERT, 1.3x better than YAKE)
- **Best Precision**: 0.400 (4 out of 10 extracted skills are correct vs 1.5/10 for KeyBERT)
- **Fastest**: 0.29s (13x faster than KeyBERT MiniLM, 157x faster than MPNet, 2.5x faster than YAKE)
- **Best Balance**: F1/Time ratio of 1.215 (optimal quality per second)
- **Cleaner Extraction**: Produces atomic skills ("machine learning", "python", "docker") vs KeyBERT's problematic long phrases ("java expert machine", "learn ml projects")
- **Algorithm Fit**: RAKE's word co-occurrence graphs and stopword-based phrase boundaries are ideal for technical terminology extraction
- **No Model Overhead**: Purely statistical, no embedding models to download/load

**Unexpected Finding**: Larger embedding models provided zero quality improvement - all-mpnet-base-v2 (768d) had identical F1 to all-MiniLM-L6-v2 (384d) while being 12x slower, demonstrating that **task-method alignment matters more than model size**.

**Trade-offs:**
- Semantic Understanding vs Statistical: Accept loss of semantic capabilities for 2.7x quality improvement
- Flexibility vs Performance: RAKE less configurable than KeyBERT but massively better results
- Model Dependency vs Speed: Remove transformer dependency for 13-157x speedup

**Consequences:**
- Positive:
  - 2.7x improvement in skill extraction quality (F1: 0.133 → 0.356)
  - 13x faster processing (3.78s → 0.29s)
  - More precise skill matching for resume-job comparison
  - Better skill gap analysis with atomic skills
  - Reduced dependencies (no sentence-transformers models to download)
  - Real-time processing viable (0.29s per document)

- Negative:
  - Loss of semantic similarity capabilities (but these weren't helping for skill extraction)
  - Need to maintain NLTK stopwords data
  - Less configurable than KeyBERT (fewer tuning parameters)

- Risks:
  - May need domain-specific refinement for highly specialized fields
  - Could miss multi-word skills not captured by co-occurrence (mitigation: combine with skill normalization database)

**Related Decisions:**
- Experiment #3: Skill Extraction Method Comparison (logs/experiment_log.md)
- Will integrate with SkillNormalizer for mapping variations to canonical forms
- Keep KeyBERT available for future semantic job search (different use case)

**Review Date:** After Phase 4 - consider hybrid approach (RAKE + YAKE + skill database matching) for improved robustness

---

### Decision #4: Use EmbeddingGemma + MMR λ=0.5 for Job Search
**Date:** 2025-11-21
**Phase:** Phase 4 - Embedding Generation & Vector Storage
**Status:** Accepted

**Context:**
Need to select embedding model and retrieval strategy for semantic job search. Tested 3 embedding models (all-MiniLM-L6-v2, all-mpnet-base-v2, google/embeddinggemma-300m) on 500 jobs (3627 chunks) with 10 diverse test queries. Also compared cosine similarity vs MMR with lambda values 0.3-0.9.

**Decision:**
Use **google/embeddinggemma-300m** for embeddings with **MMR λ=0.5** for retrieval.

**Alternatives Considered:**
1. **all-MiniLM-L6-v2**
   - Pros: Fastest (1489 chunks/sec), smallest memory (5.31 MB), 384-dim embeddings
   - Cons: Lower precision (85.0%), general-purpose model not optimized for retrieval

2. **all-mpnet-base-v2**
   - Pros: Higher quality than MiniLM, 768-dim embeddings
   - Cons: Only 1% precision improvement (86.0%) but 6x slower (242 chunks/sec)

3. **google/embeddinggemma-300m**
   - Pros: **Best precision (98.0%)**, asymmetric query/doc encoding, retrieval-optimized
   - Cons: Slowest (147 chunks/sec), requires sentence-transformers encode_query/encode_document

**Rationale:**
EmbeddingGemma + MMR λ=0.5 was selected because:
- **12-13% precision improvement**: 98% vs 85-86% for alternatives
- **Retrieval-optimized**: Asymmetric encoding (encode_query vs encode_document) is designed for search
- **MMR λ=0.5 optimal**: Highest precision (88%) AND near-maximum diversity (9.9/10)
- **Acceptable speed**: 147 chunks/sec processes 500 jobs in ~25 seconds (batch processing)

**Trade-offs:**
- Speed vs Quality: Accept 10x slower embedding for 15% precision gain
- Complexity vs Performance: Require separate query/document encoding methods
- Memory vs Quality: 768-dim embeddings use 2x memory but significantly better quality

**Consequences:**
- Positive: Significantly improved job recommendation relevance, better result diversity
- Negative: Slower embedding generation, requires model download (~1GB)
- Risks: May need optimization for very large job databases (>10K jobs)

**Related Decisions:**
- Experiment #4: Embedding Model & Retrieval Strategy Comparison (logs/experiment_log.md)

**Review Date:** After Phase 5 - evaluate end-to-end matching quality with real resumes

---

### Decision #5: Use granite4:micro for RAKE + LLM Atomic Skill Extraction
**Date:** 2025-11-21
**Phase:** Phase 5 - Resume-to-Job Matching & Ranking
**Status:** Accepted

**Context:**
RAKE (Decision #3) extracts keyphrases from job descriptions, but these phrases are too long for effective skill matching (e.g., "strong communication skills", "kubernetes container orchestration"). Need to refine RAKE output into atomic skills (e.g., "communication", "kubernetes") that match resume skills. Tested 6 Ollama models on 4 job descriptions from different professions (IT, Healthcare, Marketing, Finance).

**Decision:**
Use **granite4:micro** via RAKE + Ollama pipeline for extracting atomic skills from job descriptions.

**Pipeline:** Job Text → RAKE (keyphrases) → granite4:micro LLM (atomic skills)

**Alternatives Considered:**
1. **granite4:micro**
   - Pros: Fastest (3.75s), produces truly atomic skills (single words), 100% success rate
   - Cons: Fewer total skills extracted (12.5 avg)

2. **llama3.2:3b**
   - Pros: Similar speed (4.00s), good quality
   - Cons: Keeps some multi-word phrases, less atomic output

3. **gemma3:4b**
   - Pros: Balanced quality, proven reliable from Phase 2
   - Cons: 2x slower (7.67s), less atomic than granite4:micro

4. **Larger models (llama3.1:8b, gemma3:12b)**
   - Pros: More skills extracted, higher coverage
   - Cons: 2.6-3.4x slower, verbose phrases reduce matching precision

**Rationale:**
granite4:micro selected for skill atomization because:
- **Speed**: 3.75s avg is critical for real-time matching (vs 12.61s for gemma3:12b)
- **Atomicity**: Produces single-word skills (linux, kubernetes, python) ideal for exact matching
- **Task fit**: Different task = different model (gemma3:4b for resume parsing, granite4:micro for atomization)
- **Profession-agnostic**: Works for IT, Healthcare, Marketing, Finance equally well
- **100% reliability**: All test jobs succeeded with valid JSON

**Trade-offs:**
- Coverage vs Precision: Accept fewer skills (12.5) for better matching precision
- Speed vs Verbosity: Prioritize speed for real-time matching
- Model Size vs Task Fit: Smallest model produces cleanest output for this specific task

**Consequences:**
- Positive:
  - Better skill matching between resumes and jobs (atomic skills match exactly)
  - Fast extraction (3.75s/job) enables real-time matching
  - Works across all professions (not IT-specific)
  - Simple integration with existing RAKE pipeline

- Negative:
  - Additional LLM call per job during matching (~4s overhead)
  - May miss some compound skills (but acceptable trade-off)

- Risks:
  - Model availability (granite4:micro must be pulled in Ollama)
  - May need caching for repeated job extractions

**Related Decisions:**
- Decision #3: Use RAKE as Primary Skill Extraction Method (RAKE + LLM is extension)
- Experiment #5: RAKE + Ollama Atomic Skill Extraction (logs/experiment_log.md)

**Review Date:** After testing end-to-end skill matching with sample resumes

---

### Decision #6: Use MMR λ=0.5 for Ranking (No Skill Reranking)
**Date:** 2025-11-21
**Phase:** Phase 5 - Resume-to-Job Matching & Ranking
**Status:** Accepted

**Context:**
Need to select a ranking strategy for resume-to-job matching results. Options include pure similarity scoring, MMR with various lambda values, and skill-weighted reranking. Experiment #6 tested 5 strategies on a sample resume against the job database, measuring similarity scores, skill match percentages, and result diversity.

**Decision:**
Use **MMR with λ=0.5** as the default ranking strategy. **Do not use skill-weighted reranking** - semantic similarity alone is sufficient for quality results.

**Alternatives Considered:**
1. **Pure Similarity (No MMR)**
   - Pros: Highest similarity (0.660), highest skill match (7.4%)
   - Cons: Lower diversity (9/10 companies), may show repetitive results

2. **MMR λ=0.5 (Balanced)**
   - Pros: **Maximum diversity (10/10)**, minimal similarity loss (0.656), consistent with Phase 4
   - Cons: Slightly lower similarity than pure mode

3. **MMR λ=0.3 (High Diversity)**
   - Pros: Same diversity as λ=0.5
   - Cons: Lower skill match (3.2%), more relevance sacrifice than λ=0.5

4. **Skill Reranking (w=0.3, w=0.5)**
   - Pros: Could improve skill alignment
   - Cons: **Drastically reduced similarity** (0.660 → 0.347), **no skill improvement** (7.4% → 3.7%)

**Rationale:**
MMR λ=0.5 without skill reranking was selected because:
- **Maximum diversity**: 10/10 unique companies shows users varied opportunities
- **Minimal trade-off**: Only 0.004 similarity drop (0.660 → 0.656) for 11% diversity gain
- **Consistency**: Same λ value won in Phase 4 retrieval experiments
- **Skill reranking harmful**: Counter-intuitively, adding skill weighting reduced both similarity AND skill match
- **Semantic similarity implicit skill matching**: High embedding similarity naturally correlates with similar skill requirements

**Unexpected Finding:**
Skill-weighted reranking actually made skill match worse (7.4% → 3.7%), suggesting that semantic similarity already captures skill alignment and reranking disrupts the optimal ordering.

**Trade-offs:**
- Diversity vs Relevance: Accept 0.6% similarity drop for 11% diversity improvement
- Complexity vs Quality: Simpler approach (no reranking) produces better results

**Consequences:**
- Positive:
  - Maximum result diversity improves user experience
  - Simpler ranking logic (just MMR, no reranking)
  - Consistent approach across retrieval and ranking
  - Faster matching (no extra reranking step)

- Negative:
  - None significant - simpler and better

- Risks:
  - May need revisiting if skill match becomes critical requirement

**Related Decisions:**
- Decision #4: Use EmbeddingGemma + MMR λ=0.5 (same λ value)
- Experiment #6: Ranking Strategy Comparison (logs/experiment_log.md)

**Review Date:** After user testing - collect feedback on result diversity vs relevance preference

---

### Decision #7: Use Semantic Similarity (EmbeddingGemma) for Skill Matching
**Date:** 2025-11-25
**Phase:** Phase 6.5 - UI Improvement & Bug Fix
**Status:** Accepted

**Context:**
Current skill matching implementation shows 0% skill matches for most resumes due to exact string matching after lowercase conversion. This breaks the skill gap analysis feature entirely. Resume skills like "SQL database management" don't match job skills like "sql queries" despite clear semantic overlap. Tested 5 matching strategies (Exact, Fuzzy, Semantic, Keyword, Hybrid) on Peter Boyd's Data Scientist resume against a matched job posting.

**Decision:**
Use **semantic similarity with EmbeddingGemma** (threshold: 0.65) for matching resume skills against job skills.

**Alternatives Considered:**
1. **Exact Match (Current)**
   - Pros: Instant matching (<1ms), no dependencies
   - Cons: **Completely broken** - 0% match rate despite clear overlaps, requires identical strings

2. **Fuzzy String Matching (Levenshtein)**
   - Pros: Handles typos, fast (~96ms)
   - Cons: 0% match rate at 70% threshold, fails for different phrasing ("sql queries" vs "sql database management" = 57%)

3. **Semantic Similarity (EmbeddingGemma)**
   - Pros: **50% match rate**, captures meaning beyond words, model already loaded
   - Cons: Slower (~230ms per job)

4. **Keyword/N-gram Overlap**
   - Pros: Very fast (<1ms)
   - Cons: 0% match rate, misses semantic relationships

5. **Hybrid (Fuzzy + Keyword)**
   - Pros: Combines multiple signals
   - Cons: 0% match rate, both components too weak

**Rationale:**
Semantic similarity with EmbeddingGemma was selected because:
- **Only working solution**: 50% match rate vs 0% for all other methods
- **Captures semantic meaning**: "sql queries" ↔ "sql database management" (similarity: 0.766)
- **Model already available**: EmbeddingGemma loaded for job matching (no new dependency)
- **Proven quality**: Phase 4 showed 98% precision for job retrieval
- **Acceptable latency**: 230ms for full skill comparison vs 20-45s for resume extraction
- **Proper matches**: Successfully matched:
  - "sql queries" ← "sql database management" (0.766)
  - "data models" ← "statistical modeling" (0.721)
  - "predictive modeling" ← "statistical modeling" (0.785)

**Key Insight:**
Fuzzy matching fails for semantic concepts because edit distance doesn't measure meaning. "sql queries" and "sql database management" share only "sql" in common, yielding 57% fuzzy similarity (below 70% threshold) despite being related SQL concepts. Semantic embeddings solve this by encoding meaning, not just characters.

**Trade-offs:**
- Speed vs Quality: Accept 230ms latency for 50% match rate (vs <1ms for 0% match)
- Simplicity vs Correctness: Require embedding model loaded but get working skill matching
- Threshold Tuning: 0.65 balances precision (avoid false matches) with recall (find real matches)

**Consequences:**
- Positive:
  - **Fixes broken feature**: Skill matching works (0% → 50% match rate)
  - Accurate skill gap identification for users
  - Better match quality indicators in UI
  - Trust in system recommendations restored
  - Consistent with Phase 4/5 semantic approach
  - No new model dependencies

- Negative:
  - 230ms latency per job for skill matching
  - Requires EmbeddingGemma model loaded in memory
  - Need to manage embedding caching for resume skills

- Risks:
  - May need threshold tuning for different domains
  - Embedding model must remain loaded during matching

**Implementation:**
- Replace exact matching in `matching_engine.py::_analyze_skills()`
- Use EmbeddingGemma's `encode()` method for skill embeddings
- Cache resume skill embeddings (reuse across jobs)
- Threshold: 0.65 cosine similarity for match
- Fallback: None needed (semantic matching handles all cases)

**Related Decisions:**
- Decision #4: Use EmbeddingGemma (same model for job search and skill matching)
- Experiment #7: Skill Matching Strategy Comparison (logs/experiment_log.md)

**Review Date:** After testing on 10+ diverse resumes - verify 40-60% typical match rate holds across professions

---

### Decision #8: Use gemma3:4b for Job Skill Extraction
**Date:** 2025-11-26
**Phase:** Phase 6.5 - Skill Quality Improvement
**Status:** Accepted

**Context:**
Current job skill extraction uses RAKE → granite4:micro pipeline (Decision #5), but produces poor quality results:
- Extracts generic category names ("programming languages", "technologies and frameworks") instead of actual skills
- Includes recruitment noise ("june 5th start date", "fully remote work environment")
- Missing specific technical requirements
- Example output: ["programming languages", "data science", "computer science"] instead of ["python", "sql", "machine learning"]

CV extraction uses gemma3:4b directly with excellent results (8 specific skills). Need to improve job skill extraction to match CV quality for better skill matching.

**Decision:**
Use **gemma3:4b** (direct LLM extraction) for job skill extraction, replacing RAKE → granite4:micro pipeline.

**Alternatives Considered:**

1. **gemma3:4b (Direct LLM)**
   - Pros: 8 specific, atomic skills (python, sql, machine learning, classification, regression, clustering, databases, data types), no noise, consistent quality
   - Cons: 1.63s average latency (2.1x slower than RAKE hybrid)

2. **granite4:micro (Direct LLM)**
   - Pros: Fastest (0.44s), valid skills
   - Cons: Too few skills (only 3: python, sql, machine learning), misses important concepts (regression, clustering, databases)

3. **RAKE → granite4:micro (Current Pipeline)**
   - Pros: Moderate speed (0.76s)
   - Cons: **Unusable quality** - generic categories instead of skills, RAKE noise ("june 5th start date"), 7 generic terms vs 0 actual skills

**Rationale:**
gemma3:4b selected for job skill extraction because:
- **Best quality**: 8 specific, atomic technical skills vs 3 (granite4:micro) or 0 usable (RAKE hybrid)
- **No noise**: Zero generic categories, recruitment language, or soft skills
- **Acceptable speed**: 1.63s is reasonable for job matching (vs 20-45s for resume extraction)
- **Completeness**: Captures both high-level concepts (machine learning) and specific techniques (regression, clustering)
- **Consistency**: Same model for both CV and job extraction - symmetric pipeline
- **Better skill matching**: Actual skills enable semantic similarity matching (Decision #7)

**Key Insight:**
RAKE → LLM fails for job descriptions because:
- Job ads have different linguistic structure than resumes (marketing language, benefits, dates)
- RAKE candidates from jobs are too noisy for LLM refinement to work
- Direct LLM extraction bypasses the noisy intermediate step
- gemma3:4b (4B params) can distinguish actual skills from recruitment language better than granite4:micro (2B params) refining noisy RAKE output

**Trade-offs:**
- Speed vs Quality: Accept 1.63s (vs 0.76s) for usable skill extraction (8 specific skills vs 0)
- Pipeline Consistency: Same model (gemma3:4b) for both CVs and jobs simplifies architecture
- Method Consistency: Different approaches for CVs (still works well with RAKE) vs jobs (direct LLM)

**Consequences:**
- Positive:
  - **Fixes unusable skill extraction**: Generic categories → actual skills
  - Improved skill match percentages (comparing actual vs actual skills)
  - Better skill gap analysis (specific missing skills instead of categories like "programming languages")
  - Improved recommendation quality and user trust
  - Simpler pipeline (direct LLM vs RAKE → LLM)
  - Consistent model choice across CV and job extraction

- Negative:
  - 1.63s latency per job (vs 0.76s), but acceptable within 20-45s total matching time
  - Additional LLM calls during job matching
  - gemma3:4b must be available in Ollama

- Risks:
  - May need prompt tuning for different job posting styles
  - Latency may become issue if processing very large job databases

**Implementation:**
- Update `SkillExtractor.extract_skills()` to use gemma3:4b for job descriptions
- Or create dedicated method `extract_job_skills()` with gemma3:4b
- Keep RAKE-based extraction for resume skills (still works well)
- Use same prompt pattern as CV extraction (focus on technical skills, exclude soft skills/benefits)

**Related Decisions:**
- Decision #1: Use gemma3:4b for Resume Parsing (now also for job skill extraction)
- Decision #5: Use granite4:micro for RAKE + LLM (superseded for jobs)
- Decision #7: Use Semantic Similarity for Skill Matching (requires actual skills not categories)
- Experiment #8: Job Skill Extraction Method Comparison (logs/experiment_log.md)

**Review Date:** After Phase 6.5 completion - verify improved skill match percentages on diverse resumes

---

## Decision Summary Table

| # | Title | Phase | Status | Date | Impact |
|---|-------|-------|--------|------|--------|
| 1 | Select gemma3:4b as Primary LLM for Resume Parsing | 2 | Accepted | 2025-11-18 | High |
| 2 | Implement Dual-Parser Sequential Pipeline for PDF Extraction | 2 | Accepted | 2025-11-20 | High |
| 3 | Use RAKE as Primary Skill Extraction Method | 3 | Superseded (CVs only) | 2025-11-20 | High |
| 4 | Use EmbeddingGemma + MMR λ=0.5 for Job Search | 4 | Accepted | 2025-11-21 | High |
| 5 | Use granite4:micro for RAKE + LLM Atomic Skill Extraction | 5 | Superseded | 2025-11-21 | High |
| 6 | Use MMR λ=0.5 for Ranking (No Skill Reranking) | 5 | Accepted | 2025-11-21 | High |
| 7 | Use Semantic Similarity (EmbeddingGemma) for Skill Matching | 6.5 | Accepted | 2025-11-25 | Critical |
| 8 | Use gemma3:4b for Job Skill Extraction | 6.5 | Accepted | 2025-11-26 | High |


---

## Decisions to Make (Future)

### Phase 2
- [x] ~~PDF parsing library selection (after Experiment 2.2)~~ - COMPLETED: Dual-parser pipeline (Decision #2)
- [x] ~~LLM model selection for resume parsing (after Experiment 2.1)~~ - COMPLETED: gemma3:4b (Decision #1)
- [x] ~~Output parser approach (Pydantic vs JSON)~~ - COMPLETED: Using Pydantic (implemented in Step 2.3)

### Phase 3
- [x] ~~spaCy model size (sm vs md vs lg)~~ - COMPLETED: Using en_core_web_sm (sufficient for preprocessing)
- [x] ~~Skill extraction approach (KeyBERT vs alternatives)~~ - COMPLETED: RAKE (Decision #3)

### Phase 4
- [x] ~~Embedding model selection (after Experiment 4.1)~~ - COMPLETED: google/embeddinggemma-300m (Decision #4)
- [x] ~~FAISS index type~~ - COMPLETED: Using IndexFlatIP (inner product for cosine similarity)
- [x] ~~MMR lambda parameter (after Experiment 4.4)~~ - COMPLETED: λ=0.5 (Decision #4)

### Phase 5
- [x] ~~Ranking strategy (after Experiment 5.1)~~ - COMPLETED: MMR λ=0.5, no skill reranking (Decision #6)
- [x] ~~Skill matching logic (after Experiment 5.2)~~ - COMPLETED: Semantic similarity sufficient (Decision #6)

---

## Notes

- All major technical decisions should be documented before implementation
- Include evidence from experiments when available
- Review and update status as project evolves
- Link to relevant experiment logs for data-driven decisions
- Consider impact on: performance, maintainability, scalability, cost
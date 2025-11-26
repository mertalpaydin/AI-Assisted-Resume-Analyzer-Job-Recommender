# AI-Assisted Resume Analyzer & Job Recommender

A comprehensive NLP pipeline that intelligently matches resumes to relevant job postings, identifies skill gaps, and provides personalized recommendations for career development.

## 🎯 Project Overview

This project combines state-of-the-art NLP techniques with local LLMs and semantic search to create a production-ready resume analysis system. Given a candidate's resume (PDF), the system:

1. **Parses & Structures** the resume using LangChain and local LLMs
2. **Analyzes & Extracts** key skills, experience, and qualifications
3. **Generates Embeddings** for semantic similarity matching
4. **Matches & Ranks** relevant job postings using multiple similarity metrics
5. **Identifies Skill Gaps** between the candidate's profile and job requirements
6. **Provides Recommendations** for career development

### Key Features

- **Local LLM Processing**: Uses Ollama with multiple models (Granite, Llama, Gemma) for privacy-preserving resume parsing
- **Semantic Search**: Implements FAISS vector database with Maximum Marginal Relevance for diverse, relevant recommendations
- **Skill Gap Analysis**: Automatic identification of missing skills and career development opportunities
- **Flexible Similarity Metrics**: Compares cosine similarity, FAISS, and MMR approaches
- **End-to-End Pipeline**: Complete NLP workflow from raw PDF to actionable recommendations
- **Production-Ready Code**: Comprehensive logging, error handling, and modular architecture
- **Interactive UI** (Optional): Streamlit app for user-friendly resume upload and result visualization

---

## 🏗️ Architecture

### System Architecture

```
┌─────────────────┐
│  Resume (PDF)   │
└────────┬────────┘
         │
         ▼
┌─────────────────────────────┐
│  PDF Parsing & LLM          │
│  (LangChain + Ollama)       │
│  → Structured JSON          │
└────────┬────────────────────┘
         │
         ▼
┌─────────────────────────────┐
│  Text Preprocessing         │
│  & Skill Extraction         │
│  (spaCy + KeyBERT)         │
└────────┬────────────────────┘
         │
         ▼
┌─────────────────────────────┐
│  Embedding Generation       │
│  (sentence-transformers)    │
└────────┬────────────────────┘
         │
         ▼
┌─────────────────────────────┐
│  Vector Search & Matching   │
│  (FAISS + MMR)             │
│  Against Job Database       │
└────────┬────────────────────┘
         │
         ▼
┌─────────────────────────────┐
│  Skill Gap Analysis         │
│  & Ranking Engine           │
└────────┬────────────────────┘
         │
         ▼
┌─────────────────────────────┐
│  Results & Recommendations  │
│  (JSON/Markdown/HTML)      │
└─────────────────────────────┘
```

### Technology Stack

**Core Dependencies:**
- **LangChain**: PDF parsing (PDFPlumber + PyPDF dual-parser pipeline)
- **Ollama**: Local LLM inference - **gemma3:4b** (resume & job skill extraction)
- **sentence-transformers**:
  - **EmbeddingGemma** (google/embeddinggemma-300m) - Document embeddings
  - **all-MiniLM-L6-v2** - Skill similarity matching
- **FAISS**: Vector similarity search with MMR (λ=0.5)
- **Streamlit**: Interactive web UI with smart caching
- **Plotly**: Interactive visualizations

**Data & Format:**
- **Input**: English-only PDF resumes
- **Reference Data**: US Job Postings dataset (2023-05-05, 220K+ jobs) in JSON format
- **Output**: Structured JSON, Markdown reports, HTML visualizations

---

## 🚀 Quick Start

### Prerequisites

- **Python 3.10+**
- **Ollama** (installed and running with models downloaded)
- **FAISS-CPU**
- **RAM**: 4-8 GB minimum (depending on model selection)

### Installation

1. **Install UV** (if not already installed)
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
# On Windows: powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

2. **Clone the repository**
```bash
git clone <repository-url>
cd resume-analyzer
```

3. **Create virtual environment with UV**
```bash
uv venv
source venv/bin/activate  # On Windows: .venv\Scripts\activate
```

4. **Install all dependencies**
```bash
uv pip install -r requirements.txt
```

5. **Download spaCy model**
```bash
uv run python -m spacy download en_core_web_sm
```

6. **Verify setup**
```bash
uv run python verify_setup.py
```

### Running the Pipeline

**Streamlit UI (Recommended)**
```bash
streamlit run app.py
```

Then:
1. Upload your resume PDF
2. Adjust slider for number of matches (5-20)
3. View results in 4 tabs: Overview, Job Matches, Skills Analysis, Export

**Python API**
```python
from src.matching_engine import MatchingEngine
from pathlib import Path

# Initialize engine
engine = MatchingEngine(embeddings_path="data/embeddings")

# Match resume
result = engine.match_resume_pdf(Path("resume.pdf"), top_k=10)

# Access results
for match in result.matches:
    print(f"{match.rank}. {match.job.title} at {match.job.company}")
    print(f"   Similarity: {match.similarity_score:.2%}")
    print(f"   Skill Match: {match.skill_match['match_percentage']:.0f}%")
```

---

## 📁 Project Structure

```
resume-analyzer/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── .gitignore
├── setup.py
│
├── data/
│   ├── job_postings.json             # Job postings dataset
│   ├── sample_resumes/               # Test resumes
│   │   ├── resume_1.pdf
│   │   ├── resume_2.pdf
│   │   └── ...
│   └── embeddings_index.faiss        # Cached FAISS index
│
├── src/
│   ├── __init__.py
│   ├── main.py                        # CLI entry point
│   ├── app.py                         # Streamlit UI (optional)
│   ├── pipeline.py                    # Main orchestration
│   │
│   ├── components/
│   │   ├── __init__.py
│   │   ├── pdf_parser.py             # PDF extraction with LangChain
│   │   ├── llm_extractor.py          # LLM-powered structured extraction
│   │   ├── text_preprocessor.py      # Text cleaning & preprocessing
│   │   ├── skill_extractor.py        # KeyBERT-based skill extraction
│   │   ├── embedding_generator.py    # Embedding model interface
│   │   ├── vector_store.py           # FAISS & search implementation
│   │   ├── matching_engine.py        # Resume-to-job matching
│   │   ├── skill_gap_analyzer.py     # Gap identification
│   │   └── llm_suggestions.py        # CV improvement suggestions (optional)
│   │
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── logging_utils.py          # Centralized logging
│   │   ├── config.py                 # Configuration management
│   │   ├── constants.py              # Constants & schema definitions
│   │   └── helpers.py                # Utility functions
│   │
│   └── models/
│       ├── __init__.py
│       ├── resume_schema.py          # Pydantic models
│       └── recommendation_schema.py
│
├── tests/
│   ├── __init__.py
│   ├── test_pdf_parser.py
│   ├── test_llm_extractor.py
│   ├── test_preprocessing.py
│   ├── test_embeddings.py
│   ├── test_matching.py
│   ├── test_skill_gap.py
│   └── test_integration.py
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_model_comparison.ipynb
│   └── 03_results_analysis.ipynb
│
├── logs/
│   ├── learning_log.md               # Learning documentation
│   ├── experiment_log.md             # Experiment tracking
│   ├── decisions.md                  # Architecture decisions
│   ├── challenges.md                 # Issues & solutions
│   └── debug.log                     # Runtime logs
│
└── docs/
    ├── architecture.md
    ├── api_reference.md
    ├── model_selection.md
    └── deployment_guide.md
```

---

## 🔬 Key Components

### 1. PDF Parsing & LLM Extraction (`components/pdf_parser.py`, `components/llm_extractor.py`)

Converts unstructured PDF resumes into structured JSON using LangChain and local LLMs.

**Features:**
- Multi-page resume support
- Handles various formatting styles
- LLM-powered field extraction with output validation
- Configurable model selection (Granite, Llama, Gemma)

**Output Schema:**
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

### 2. Text Preprocessing & Skill Extraction (`components/text_preprocessor.py`, `components/skill_extractor.py`)

Cleans and normalizes text, then extracts key technical and soft skills using KeyBERT.

**Operations:**
- Lowercasing and special character removal
- Tokenization and lemmatization (spaCy)
- Stopword removal
- Skill normalization (e.g., "Python" → "python", "JS" → "javascript")

### 3. Embedding Generation (`components/embedding_generator.py`)

Generates semantic embeddings for resumes and job postings for similarity matching.

**Supported Models:**
- EmbeddingGemma (default, specified in requirements)
- all-MiniLM-L6-v2 (fast, lightweight)
- all-mpnet-base-v2 (higher quality, slower)
- Custom fine-tuned embeddings (optional)

### 4. Vector Search & FAISS (`components/vector_store.py`)

Implements chunk-level semantic search with automatic job ID deduplication for complete job retrieval.

**Architecture:** Chunking + Job ID Mapping
- Smart semantic chunking (by sections: title, requirements, description)
- One embedding per chunk (better matching accuracy)
- Automatic deduplication by job_id
- Returns complete job postings (whole ads, not fragments)

**Key Advantage:**
- ✅ Better matching accuracy (chunk-level precision captures nuanced skills)
- ✅ Efficient retrieval (multiple chunks from same job → one result)
- ✅ Complete information returned (full job postings)
- ✅ Automatic deduplication (mark chunks, deduplicate by job_id)
- ✅ Scales well with long job descriptions

**Search Methods:**
- **Cosine Similarity** (sklearn baseline)
- **FAISS IndexFlatL2** (Euclidean distance)
- **FAISS IndexIVFFlat** (approximate NN for scale)
- **Maximum Marginal Relevance (MMR)** - diverse jobs (not duplicate chunks)

### 5. Matching & Ranking (`components/matching_engine.py`)

Orchestrates the matching pipeline and implements ranking strategies.

**Process:**
1. Generate embeddings for candidate resume
2. Query vector store for similar jobs
3. Rank by weighted similarity scores
4. Apply business rules (experience level, location, etc.)

### 6. Skill Gap Analysis (`components/skill_gap_analyzer.py`)

Identifies skill overlaps and gaps between candidate and job requirements.

**Outputs:**
- Matched skills (with confidence scores)
- Missing skills (ranked by importance)
- Extra skills (candidate advantages)
- Overall skill match percentage

---

## 🧪 Experiments & Model Comparisons

### LLM Model Comparison

Three models tested via Ollama for resume parsing:

| Model               | Size   | Speed       | Accuracy  | Best For                             | Selected |
| ------------------- | ------ | ----------- | --------- | ------------------------------------ | -------- |
| **Granite 4 Micro** | ~2.1GB | ⚡ Very Fast | Good      | Fast iteration, resource-constrained | |
| **Llama 3.2 3B**    | ~2.0GB | ⚡ Fast      | Very Good | Balanced speed/quality               | |
| **Gemma 3 4B**      | ~3.3GB | Medium      | Excellent | High-quality extraction              | ✅ |

**Selected:** **gemma3:4b** for both resume AND job skill extraction (100% extraction accuracy, consistent output)

**Key Learnings:**
- gemma3:4b: Perfect accuracy (100%), handles complex edge cases, worth the extra latency
- Larger models produce more granular skills ("classification", "regression" vs just "machine learning")
- LLM-only extraction cleaner than hybrid approaches (RAKE adds noise)

See `/logs/experiment_log.md` for detailed performance metrics.

### Embedding Model Comparison

Tested multiple sentence-transformers models on 500 jobs (3627 chunks):

| Model | Precision@10 | Speed | Memory | Selected |
|-------|--------------|-------|--------|----------|
| all-MiniLM-L6-v2 | 85.0% | 1489/s | 5.31 MB | (Skill matching) |
| all-mpnet-base-v2 | 86.0% | 242/s | 10.63 MB | |
| **google/embeddinggemma-300m** | **98.0%** | 147/s | 10.63 MB | ✅ |

**Selected:** **EmbeddingGemma** for document embeddings (98% precision, 12-13% better than alternatives)

**Key Finding:** EmbeddingGemma uses asymmetric encoding (encode_query vs encode_document) optimized for retrieval tasks

### Similarity Search Methods

Tested retrieval strategies on real data:

| Strategy | Precision | Companies | Categories | Selected |
|----------|-----------|-----------|------------|----------|
| Cosine | 0.850 | 9.5/10 | 3.5 | |
| MMR λ=0.3 | 0.850 | 9.9/10 | 3.7 | |
| **MMR λ=0.5** | **0.880** | 9.9/10 | 3.8 | ✅ |
| MMR λ=0.7 | 0.840 | 9.9/10 | 3.6 | |

**Selected:** **MMR λ=0.5** - highest precision (0.880) with maximum diversity (9.9/10 unique companies)

**Key Learning:** MMR actually improved precision (not just diversity) by reducing redundant results

### Skill Matching Strategy

**Critical Decision:** Switched from exact string matching to **semantic similarity** (Phase 6.5)

| Method | Avg Skill Match | Matched Skills | Selected |
|--------|----------------|----------------|----------|
| Exact String Match | 0.0% | 0 | ❌ (Broken) |
| **Semantic Similarity (threshold=0.5)** | **37.6%** | 2-5 per job | ✅ |

**Model:** all-MiniLM-L6-v2 (fast, accurate for short skill phrases)

**Example:** "SQL database management" ↔ "sql queries" (0.65 similarity) ✅ Matched

See `/logs/decisions.md` for detailed rationale and `/logs/experiment_log.md` for all experiments.

---

## 📊 Results & Metrics

### Success Criteria

- ✅ Resume parsing accuracy: **100%** (gemma3:4b with dual-parser fallback)
- ✅ Skill extraction: gemma3:4b produces atomic, technical skills
- ✅ Skill matching: **37.6% avg** with semantic similarity (was 0% with exact match)
- ✅ Recommendation relevance: MMR λ=0.5 achieves 88% precision with max diversity
- ✅ Production-grade code: Comprehensive logging, error handling, graceful fallbacks

### Sample Output

```json
{
  "resume": {
    "full_name": "Jane Doe",
    "skills": ["python", "machine learning", "pytorch", "sql", "aws"],
    "experience_years": 5
  },
  "matches": [
    {
      "rank": 1,
      "job_id": "12345",
      "job_title": "Senior ML Engineer",
      "company": "Tech Corp",
      "similarity_score": 0.89,
      "skill_match_percentage": 85,
      "matched_skills": ["python", "pytorch", "machine learning"],
      "missing_skills": ["tensorflow", "kubernetes"],
      "reasoning": "Strong ML background with Python expertise..."
    },
    {
      "rank": 2,
      "job_id": "12346",
      "job_title": "Data Engineer",
      "company": "Data Systems Inc",
      "similarity_score": 0.82,
      "skill_match_percentage": 78,
      "matched_skills": ["python", "sql", "aws"],
      "missing_skills": ["spark", "hive"],
      "reasoning": "Good data infrastructure background..."
    }
  ]
}
```

---

## 🔧 Configuration

Edit `src/utils/config.py` to customize:

```python
# LLM Configuration (Ollama)
LLM_MODEL = "gemma3:4b"  # Selected for 100% accuracy
LLM_TEMPERATURE = 0.1  # Low temp for consistent extraction

# Embedding Configuration
EMBEDDING_MODEL_DOCUMENTS = "google/embeddinggemma-300m"  # 98% precision
EMBEDDING_MODEL_SKILLS = "sentence-transformers/all-MiniLM-L6-v2"  # Skill matching
EMBEDDING_DIM = 768

# Search Configuration
SEARCH_METHOD = "mmr"  # Always enabled (best results)
TOP_K_DEFAULT = 5  # Initial load
TOP_K_MAX = 20  # Slider maximum
MMR_LAMBDA = 0.5  # Balanced relevance/diversity

# Skill Matching
SKILL_SIMILARITY_THRESHOLD = 0.5  # Semantic matching threshold

# Output
OUTPUT_FORMATS = ["json", "markdown", "html"]
```

---

## 📚 Learning & Documentation

This project emphasizes continuous documentation and experimentation tracking:

- **`/logs/learning_log.md`** - Main learning outcomes by phase
- **`/logs/experiment_log.md`** - Detailed experiment results and comparisons
- **`/logs/decisions.md`** - Technical decisions with rationale
- **`/logs/challenges.md`** - Problems encountered and solutions
- **`/logs/debug.log`** - Runtime debugging information

### Key Learnings

**NLP & Embeddings:**
- Trade-offs between embedding model size, speed, and quality
- Impact of preprocessing on downstream matching accuracy
- Importance of skill normalization for reliable extraction

**Vector Search:**
- Cosine similarity vs. approximate nearest neighbors
- MMR for diverse, non-redundant recommendations
- Scaling considerations for production systems

**LLM Engineering:**
- Structured output extraction from unstructured text
- Prompt optimization for specific domains
- Model selection based on inference constraints

**System Design:**
- Modular, testable pipeline architecture
- Importance of comprehensive logging for debugging
- Production-ready error handling and validation

---

## 🧪 Testing

Run comprehensive test suite:

```bash
# Unit tests
pytest tests/test_*.py -v

# Integration tests
pytest tests/test_integration.py -v

# With coverage
pytest tests/ --cov=src --cov-report=html
```

Key test areas:
- PDF parsing with various resume formats
- LLM extraction accuracy and consistency
- Text preprocessing pipeline
- Embedding generation and similarity
- End-to-end matching pipeline
- Skill gap analysis accuracy

---

## 🚀 Deployment

### Local Development

```bash
streamlit run src/app.py
# Access at http://localhost:8501
```

### Docker

```bash
docker build -t resume-analyzer .
docker run -p 8501:8501 resume-analyzer
```

### Production Considerations

- Batch processing for large-scale job databases
- Caching embeddings and FAISS indices
- Model quantization for faster inference
- Load balancing for concurrent resume uploads
- Database integration for persistent results
- API rate limiting and monitoring

See `/docs/deployment_guide.md` for detailed deployment instructions.

---

## 🔮 Future Enhancements

- [ ] **Speed Optimizations** - Extraction speeds is relatively slow, find ways to improve it
- [ ] **AI Suggestions** - Add ollama and Gemini API suggestions for extracted matches
- [ ] **Multi-language Support** - Extend beyond English
- [ ] **OCR - Non-text PDF integration** - Integrate an OCR model for PDF parsing

---

## 📖 API Reference

For detailed API documentation, see `/docs/api_reference.md`

### Main Pipeline

```python
from src.pipeline import ResumeMatcher

matcher = ResumeMatcher(
    job_data_path="data/job_postings.json",
    embedding_model="sentence-transformers/all-MiniLM-L6-v2",
    llm_model="llama3.2:3b"
)

# Single resume matching
results = matcher.match_resume("resume.pdf", top_k=10)

# Batch processing
results_batch = matcher.match_resumes_batch(["resume1.pdf", "resume2.pdf"])
```

### Individual Components

```python
# Parse resume
from src.components.pdf_parser import PDFParser
parser = PDFParser()
text = parser.parse("resume.pdf")

# Extract structured data
from src.components.llm_extractor import LLMExtractor
extractor = LLMExtractor(model="llama3.2:3b")
resume_data = extractor.extract(text)

# Generate embeddings
from src.components.embedding_generator import EmbeddingGenerator
embedder = EmbeddingGenerator(model="sentence-transformers/all-MiniLM-L6-v2")
embedding = embedder.embed(text)

# Search jobs
from src.components.vector_store import VectorStore
store = VectorStore(index_path="data/embeddings_index.faiss")
results = store.search(embedding, k=10, method="mmr")
```

---

## 🤝 Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Add tests for new functionality
4. Update documentation
5. Submit a pull request

---

## 📝 License

MIT license

---

## 📧 Contact & Questions

For questions or feedback:
- Create an issue in the repository
- Contact the maintainer for specific technical questions
- See `/logs/` for detailed learning outcomes and experiments

---

### Libraries & Tools

- [LangChain Documentation](https://python.langchain.com/)
- [FAISS Documentation](https://github.com/facebookresearch/faiss)
- [Ollama Models](https://ollama.ai/models)
- [sentence-transformers](https://www.sbert.net/)
- [spaCy Documentation](https://spacy.io/)
- [KeyBERT Documentation](https://maartengr.github.io/KeyBERT/)

---

**Last Updated:** November 2025  
**Status:** Active Development / Learning Project  
**Maintained By:** Mert Alp Aydin

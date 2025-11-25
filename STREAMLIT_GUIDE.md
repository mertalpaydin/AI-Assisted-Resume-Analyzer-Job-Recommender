# Streamlit App Guide

## Running the Application

To start the Streamlit web application:

```bash
streamlit run app.py
```

The app will automatically open in your browser at `http://localhost:8501`

## Features

### 1. Resume Upload
- Upload PDF resumes through the web interface
- Automatic parsing and extraction of resume data
- Support for various resume formats

### 2. Configuration Options (Sidebar)
- **Number of job matches**: Choose between 5-20 top matches (default: 10)
- **Use MMR for diversity**: Toggle Maximum Marginal Relevance for diverse results (default: ON)

### 3. Results Tabs

#### 📊 Overview
- Key metrics dashboard (total matches, average similarity, average skill match, search time)
- **Match Quality by Job**: Interactive chart showing skill match vs. similarity for each job
- **Most Valuable Skills**: Bar chart highlighting your most in-demand skills

#### 💼 Job Matches
- Detailed job cards for each matched position
- **Filter by title/company**: Search through results
- **Min similarity slider**: Filter jobs by minimum similarity threshold
- Expandable sections for:
  - Skills breakdown (matched, missing)
  - Full job description

#### 🎯 Skills Analysis
- **Development Recommendations**: AI-generated suggestions for skill improvement
- **Skill Profile**: Overview of your total skills and most valuable ones
- **All Skills**: Complete list of your extracted skills

#### 📥 Export
- Download results in multiple formats:
  - **JSON**: For programmatic use
  - **Markdown**: For documentation
  - **HTML**: For viewing in browser
- Live preview of Markdown and HTML reports

## Prerequisites

Before running the app, ensure you have:

1. **Generated job embeddings**:
   ```bash
   python src/generate_job_embeddings.py
   ```
   This creates the FAISS index and embeddings in `data/embeddings/`

2. **Ollama running** with required models:
   - `gemma3:4b` - for resume parsing
   - `granite4:micro` - for skill extraction

3. **Dependencies installed**:
   ```bash
   uv pip install -r requirements.txt
   ```

## Performance Tips

- The matching engine is cached after first load for faster subsequent runs
- Processing time depends on:
  - Resume complexity: ~5-15 seconds for parsing
  - Number of matches requested: ~2-5 seconds for retrieval
  - Total: ~10-20 seconds for complete pipeline

## Troubleshooting

### Error: "No embeddings found"
**Solution**: Run `python src/generate_job_embeddings.py` first

### Error: "Failed to connect to Ollama"
**Solution**: Ensure Ollama is running (`ollama serve`) and models are downloaded

### Slow performance
**Solution**:
- Reduce number of matches in sidebar
- Check Ollama is using GPU if available
- Close other resource-intensive applications

## Architecture

```
User uploads PDF
    ↓
Resume Parser (gemma3:4b)
    ↓
Embedding Generator (EmbeddingGemma)
    ↓
FAISS Vector Search (MMR λ=0.5)
    ↓
Skill Gap Analyzer (RAKE + granite4:micro)
    ↓
Report Generator
    ↓
Interactive Results Display
```

## Next Steps (Phase 7)

Planned enhancements:
- AI-generated match insights using Ollama
- Deep insights with Gemini API integration
- Async streaming results for faster UI feedback
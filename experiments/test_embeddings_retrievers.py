"""
Phase 4 Experiments - Embedding Models and Retrieval Strategies

This script compares:
1. Embedding models: all-MiniLM-L6-v2 vs all-mpnet-base-v2 vs embeddinggemma
2. Retrieval strategies: Cosine similarity vs MMR with different lambda values

Evaluation metrics:
- Relevance: Keyword-based precision (how many top results contain query keywords)
- Diversity: Unique companies, job categories, and locations in results
- Performance: Embedding speed, search latency, memory usage

Author: Mert Alp Aydin
Date: 2025-11-21
"""

import sys
import time
import json
import logging
import re
from pathlib import Path
from datetime import datetime
from collections import Counter
import numpy as np
import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from job_chunker import JobChunker, load_jobs_from_jsonl
from job_id_mapper import JobIDMapper
from embedding_generator import ChunkEmbeddingGenerator
from vector_store import ChunkVectorStore
from logging_utils import setup_logger

logger = setup_logger(__name__)

# Check CUDA availability
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {DEVICE}")
if DEVICE == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")


class EmbeddingGemmaGenerator:
    """Wrapper for EmbeddingGemma via sentence-transformers."""

    def __init__(self, model_name: str = 'google/embeddinggemma-300m'):
        from sentence_transformers import SentenceTransformer
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        if DEVICE == 'cuda':
            self.model = self.model.to(DEVICE)
        self.embedding_dim = 768

    def embed_text(self, text: str) -> np.ndarray:
        # Use encode_query for query texts
        return self.model.encode_query(text).astype('float32')

    def embed_texts(self, texts, show_progress=True):
        # Use encode_document for document texts
        if show_progress:
            print(f"  Embedding {len(texts)} texts...")
        return self.model.encode_document(texts, show_progress_bar=show_progress).astype('float32')

    def embed_chunks(self, chunks, show_progress=True):
        texts = [chunk.text for chunk in chunks]
        chunk_ids = [chunk.chunk_id for chunk in chunks]
        embeddings = self.embed_texts(texts, show_progress)
        return embeddings, chunk_ids

    def get_sentence_embedding_dimension(self):
        return self.embedding_dim


def calculate_keyword_precision(query: str, job_descriptions: list, k: int = 10) -> float:
    """
    Calculate precision based on keyword overlap.
    Returns ratio of top-k results that contain at least one query keyword.
    """
    # Extract keywords from query (lowercase, remove common words)
    query_words = set(re.findall(r'\b[a-zA-Z]{3,}\b', query.lower()))
    stopwords = {'the', 'and', 'for', 'with', 'senior', 'junior', 'experience'}
    query_keywords = query_words - stopwords

    matches = 0
    for desc in job_descriptions[:k]:
        desc_lower = desc.lower()
        if any(kw in desc_lower for kw in query_keywords):
            matches += 1

    return matches / min(k, len(job_descriptions)) if job_descriptions else 0


def calculate_diversity_metrics(jobs: list) -> dict:
    """Calculate comprehensive diversity metrics for search results."""
    companies = [j.company for j in jobs]
    titles = [j.title for j in jobs]

    # Title category diversity (extract role type)
    categories = []
    for title in titles:
        title_lower = title.lower()
        if 'engineer' in title_lower or 'developer' in title_lower:
            categories.append('engineering')
        elif 'manager' in title_lower or 'director' in title_lower:
            categories.append('management')
        elif 'analyst' in title_lower or 'scientist' in title_lower:
            categories.append('analytics')
        elif 'designer' in title_lower:
            categories.append('design')
        elif 'sales' in title_lower or 'account' in title_lower:
            categories.append('sales')
        else:
            categories.append('other')

    return {
        'unique_companies': len(set(companies)),
        'unique_categories': len(set(categories)),
        'company_entropy': -sum((c/len(companies)) * np.log(c/len(companies) + 1e-10)
                                for c in Counter(companies).values()) if companies else 0,
        'category_distribution': dict(Counter(categories))
    }


def run_embedding_model_experiment(jobs, all_chunks, mapper):
    """
    Compare embedding models on quality and performance with comprehensive metrics.
    """
    print("\n" + "="*80)
    print("EXPERIMENT 1: EMBEDDING MODEL COMPARISON")
    print("="*80)

    models = [
        ('all-MiniLM-L6-v2', 384, 'sentence-transformers'),
        ('all-mpnet-base-v2', 768, 'sentence-transformers'),
        ('google/embeddinggemma-300m', 768, 'embeddinggemma'),
    ]

    print(f"\nTest data: {len(jobs)} jobs, {len(all_chunks)} chunks")
    print(f"Device: {DEVICE}")

    # More comprehensive test queries with expected keywords
    test_queries = [
        {"query": "Senior Software Engineer Python backend APIs REST", "keywords": ["python", "software", "engineer", "api", "backend"]},
        {"query": "Data Scientist machine learning TensorFlow deep learning NLP", "keywords": ["data", "scientist", "machine", "learning", "tensorflow"]},
        {"query": "Full stack developer React Node.js JavaScript frontend", "keywords": ["react", "node", "javascript", "developer", "frontend"]},
        {"query": "Project Manager Agile Scrum team leadership", "keywords": ["project", "manager", "agile", "scrum"]},
        {"query": "DevOps Engineer Kubernetes Docker CI/CD cloud AWS", "keywords": ["devops", "kubernetes", "docker", "aws", "cloud"]},
        {"query": "Product Manager roadmap stakeholder user research", "keywords": ["product", "manager", "roadmap"]},
        {"query": "UX Designer user experience wireframes Figma prototyping", "keywords": ["ux", "designer", "user", "experience", "figma"]},
        {"query": "Database Administrator SQL PostgreSQL Oracle performance", "keywords": ["database", "sql", "postgresql", "oracle", "admin"]},
        {"query": "Security Engineer penetration testing cybersecurity compliance", "keywords": ["security", "engineer", "cyber", "compliance"]},
        {"query": "Marketing Manager digital campaigns analytics SEO", "keywords": ["marketing", "manager", "digital", "seo"]},
    ]

    results = {}

    for model_name, dim, model_type in models:
        model_key = f"{model_name}" if model_type != 'ollama' else f"{model_name}_ollama"
        print(f"\n--- Testing {model_name} (type={model_type}) ---")

        try:
            # Load model with CUDA support
            start_time = time.time()
            if model_type == 'embeddinggemma':
                generator = EmbeddingGemmaGenerator(model_name=model_name)
                dim = generator.embedding_dim
            else:
                generator = ChunkEmbeddingGenerator(model_name=model_name)
                # Enable CUDA if available
                if DEVICE == 'cuda':
                    generator.model = generator.model.to(DEVICE)
                dim = generator.embedding_dim
            model_load_time = time.time() - start_time

            print(f"  Model loaded in {model_load_time:.2f}s (dim={dim})")

            # Embed chunks
            start_time = time.time()
            embeddings, chunk_ids = generator.embed_chunks(all_chunks, show_progress=True)
            embed_time = time.time() - start_time

            # Build vector store
            store = ChunkVectorStore(embedding_dim=dim)
            store.add_embeddings(embeddings, chunk_ids)

            # Test search quality with comprehensive metrics
            all_precisions = []
            all_diversities = []
            query_details = []

            for q in test_queries:
                query = q["query"]
                expected_keywords = q["keywords"]

                query_embedding = generator.embed_text(query)

                # Get top results
                search_results = store.search_with_job_dedup(
                    query_embedding, mapper, k_jobs=10, k_chunks=50, use_mmr=False
                )

                # Calculate keyword precision
                job_texts = [f"{job.title} {job.description}" for job, _, _ in search_results]
                precision = calculate_keyword_precision(query, job_texts, k=10)
                all_precisions.append(precision)

                # Calculate diversity
                result_jobs = [job for job, _, _ in search_results]
                diversity = calculate_diversity_metrics(result_jobs)
                all_diversities.append(diversity['unique_companies'])

                query_details.append({
                    'query': query,
                    'precision': precision,
                    'top_5_titles': [job.title for job, _, _ in search_results[:5]],
                    'top_score': search_results[0][1] if search_results else 0,
                    'diversity': diversity
                })

            results[model_key] = {
                'dim': dim,
                'model_load_time': model_load_time,
                'embed_time': embed_time,
                'chunks_per_sec': len(all_chunks) / embed_time,
                'memory_mb': embeddings.nbytes / (1024 * 1024),
                'avg_precision': np.mean(all_precisions),
                'avg_diversity': np.mean(all_diversities),
                'query_details': query_details
            }

            print(f"  Embedding: {embed_time:.2f}s ({len(all_chunks)/embed_time:.1f} chunks/sec)")
            print(f"  Memory: {embeddings.nbytes / (1024*1024):.2f} MB")
            print(f"  Avg Precision@10: {results[model_key]['avg_precision']:.3f}")
            print(f"  Avg Diversity: {results[model_key]['avg_diversity']:.1f} unique companies")

        except Exception as e:
            import traceback
            print(f"  ERROR: {e}")
            traceback.print_exc()
            results[model_key] = {'error': str(e)}

    return results


def run_retrieval_strategy_experiment(jobs, all_chunks, mapper, generator, store):
    """
    Compare retrieval strategies: Cosine vs MMR with different lambda values.
    Uses pre-built embeddings from embedding experiment.
    """
    print("\n" + "="*80)
    print("EXPERIMENT 2: RETRIEVAL STRATEGY COMPARISON")
    print("="*80)

    print(f"\nTest data: {len(jobs)} jobs, {len(all_chunks)} chunks")

    # More comprehensive test queries
    test_queries = [
        "Senior Software Engineer Python backend APIs REST",
        "Data Scientist machine learning TensorFlow deep learning NLP",
        "Full stack developer React Node.js JavaScript frontend",
        "Project Manager Agile Scrum team leadership",
        "DevOps Engineer Kubernetes Docker CI/CD cloud AWS",
        "Product Manager roadmap stakeholder user research",
        "UX Designer user experience wireframes Figma prototyping",
        "Database Administrator SQL PostgreSQL Oracle performance",
        "Security Engineer penetration testing cybersecurity compliance",
        "Marketing Manager digital campaigns analytics SEO",
    ]

    strategies = [
        ('cosine', False, 0.0),
        ('mmr_0.3', True, 0.3),
        ('mmr_0.5', True, 0.5),
        ('mmr_0.7', True, 0.7),
        ('mmr_0.9', True, 0.9),
    ]

    results = {}

    for strategy_name, use_mmr, lambda_val in strategies:
        print(f"\n--- Testing {strategy_name} ---")

        all_precisions = []
        all_diversities = []
        all_category_diversities = []
        query_results = []
        total_time = 0

        for query in test_queries:
            query_embedding = generator.embed_text(query)

            start_time = time.time()
            if use_mmr:
                search_results = store.search_with_job_dedup(
                    query_embedding, mapper, k_jobs=10, k_chunks=50,
                    use_mmr=True, lambda_mult=lambda_val
                )
            else:
                search_results = store.search_with_job_dedup(
                    query_embedding, mapper, k_jobs=10, k_chunks=50, use_mmr=False
                )
            search_time = time.time() - start_time
            total_time += search_time

            # Calculate precision
            job_texts = [f"{job.title} {job.description}" for job, _, _ in search_results]
            precision = calculate_keyword_precision(query, job_texts, k=10)
            all_precisions.append(precision)

            # Calculate diversity metrics
            result_jobs = [job for job, _, _ in search_results]
            diversity = calculate_diversity_metrics(result_jobs)
            all_diversities.append(diversity['unique_companies'])
            all_category_diversities.append(diversity['unique_categories'])

            query_results.append({
                'query': query,
                'precision': precision,
                'top_5_titles': [job.title for job, _, _ in search_results[:5]],
                'diversity': diversity,
                'search_time': search_time
            })

        avg_precision = np.mean(all_precisions)
        avg_diversity = np.mean(all_diversities)
        avg_category_diversity = np.mean(all_category_diversities)
        avg_time = total_time / len(test_queries)

        results[strategy_name] = {
            'use_mmr': use_mmr,
            'lambda': lambda_val,
            'avg_precision': avg_precision,
            'avg_unique_companies': avg_diversity,
            'avg_unique_categories': avg_category_diversity,
            'avg_search_time': avg_time,
            'query_results': query_results
        }

        print(f"  Precision@10: {avg_precision:.3f}")
        print(f"  Avg unique companies: {avg_diversity:.1f}/10")
        print(f"  Avg unique categories: {avg_category_diversity:.1f}")
        print(f"  Avg search time: {avg_time*1000:.1f}ms")

    return results


def print_experiment_summary(embed_results, retrieval_results):
    """Print formatted experiment summary."""
    print("\n" + "="*80)
    print("EXPERIMENT SUMMARY")
    print("="*80)

    print("\n### Embedding Model Comparison ###")
    print(f"{'Model':<30} {'Dim':>6} {'Time (s)':>10} {'Speed':>12} {'Prec@10':>10} {'Diversity':>10}")
    print("-" * 90)

    for model, data in embed_results.items():
        if 'error' not in data:
            print(f"{model:<30} {data['dim']:>6} {data['embed_time']:>10.1f} {data['chunks_per_sec']:>10.1f}/s {data['avg_precision']:>10.3f} {data['avg_diversity']:>10.1f}")
        else:
            print(f"{model:<30} ERROR: {data['error'][:40]}")

    print("\n### Retrieval Strategy Comparison ###")
    print(f"{'Strategy':<15} {'Precision':>12} {'Companies':>12} {'Categories':>12} {'Time (ms)':>12}")
    print("-" * 65)

    for strategy, data in retrieval_results.items():
        print(f"{strategy:<15} {data['avg_precision']:>12.3f} {data['avg_unique_companies']:>10.1f}/10 {data['avg_unique_categories']:>10.1f} {data['avg_search_time']*1000:>12.1f}")


def save_experiment_results(embed_results, retrieval_results, output_path):
    """Save experiment results to JSON file."""
    results = {
        'timestamp': datetime.now().isoformat(),
        'embedding_model_experiment': embed_results,
        'retrieval_strategy_experiment': retrieval_results
    }

    # Convert numpy types to Python types for JSON serialization
    def convert(obj):
        if isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=convert)

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    logger.info("=" * 70)
    logger.info("PHASE 4 EXPERIMENT: Embedding Models & Retrieval Strategies")
    logger.info("=" * 70)

    # Load test data (500 jobs for comprehensive experiments)
    data_path = Path(__file__).parent.parent / "data" / "techmap-jobs_us_2023-05-05.json"

    NUM_JOBS = 500  # Increase for more comprehensive testing

    logger.info(f"Loading {NUM_JOBS} jobs for experiments...")
    print(f"Loading {NUM_JOBS} jobs for experiments...")
    jobs = load_jobs_from_jsonl(str(data_path), limit=NUM_JOBS)

    # Prepare chunks once (reused by both experiments)
    print("\nChunking jobs...")
    chunker = JobChunker()
    mapper = JobIDMapper()
    all_chunks = []

    for job in jobs:
        chunks = chunker.chunk_job(job)
        mapper.add_job(job, chunks)
        all_chunks.extend(chunks)

    print(f"Created {len(all_chunks)} chunks from {len(jobs)} jobs")
    print(f"Average chunks per job: {len(all_chunks)/len(jobs):.1f}")

    # Run embedding model experiment
    embed_results = run_embedding_model_experiment(jobs, all_chunks, mapper)

    # Use best performing model for retrieval experiment
    # Build embeddings with the selected model for retrieval experiment
    print("\n\nPreparing for retrieval experiment with best model...")
    best_model = 'all-MiniLM-L6-v2'  # Fast and good quality
    generator = ChunkEmbeddingGenerator(model_name=best_model)
    if DEVICE == 'cuda':
        generator.model = generator.model.to(DEVICE)

    embeddings, chunk_ids = generator.embed_chunks(all_chunks, show_progress=True)
    store = ChunkVectorStore(embedding_dim=generator.embedding_dim)
    store.add_embeddings(embeddings, chunk_ids)

    # Run retrieval strategy experiment
    retrieval_results = run_retrieval_strategy_experiment(jobs, all_chunks, mapper, generator, store)

    # Print summary
    print_experiment_summary(embed_results, retrieval_results)

    # Log results
    logger.info("EMBEDDING MODEL RESULTS:")
    for model, data in embed_results.items():
        if 'error' not in data:
            logger.info(f"  {model}: Precision={data['avg_precision']:.3f}, Speed={data['chunks_per_sec']:.1f}/s")

    logger.info("RETRIEVAL STRATEGY RESULTS:")
    for strategy, data in retrieval_results.items():
        logger.info(f"  {strategy}: Precision={data['avg_precision']:.3f}, Companies={data['avg_unique_companies']:.1f}/10")

    # Save results
    output_path = Path(__file__).parent / "results" / "phase4_experiment_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_experiment_results(embed_results, retrieval_results, output_path)

    logger.info(f"Results saved to: {output_path}")
    logger.info("PHASE 4 EXPERIMENT COMPLETE")

    print("\n" + "="*80)
    print("EXPERIMENT COMPLETE")
    print("="*80)
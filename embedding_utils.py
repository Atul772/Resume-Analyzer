# embedding_utils.py - Enhanced Version

from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Dict, Union, Optional, Tuple
import hashlib
import pickle
import os
from functools import lru_cache
import re
from sklearn.metrics.pairwise import cosine_similarity

# Model configuration
MODEL_NAME = "all-mpnet-base-v2"
CACHE_DIR = ".embedding_cache"
MAX_CACHE_SIZE = 1000

# Global model instance
_model = None
_model_cache = {}

# Create cache directory if it doesn't exist
if not os.path.exists(CACHE_DIR):
    os.makedirs(CACHE_DIR)


def get_model(model_name: str = MODEL_NAME) -> SentenceTransformer:
    """
    Load and cache the sentence transformer model.
    Supports multiple models.
    """
    global _model_cache
    
    if model_name not in _model_cache:
        print(f"Loading model: {model_name}")
        _model_cache[model_name] = SentenceTransformer(model_name)
    
    return _model_cache[model_name]


def preprocess_text(text: str) -> str:
    """
    Preprocess text for better embedding quality.
    """
    if not text:
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove excessive whitespace
    text = ' '.join(text.split())
    
    # Remove special characters but keep important ones
    text = re.sub(r'[^\w\s\-\+\#\.\@\/]', ' ', text)
    
    # Remove multiple spaces
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()


def get_text_hash(text: str) -> str:
    """Generate a hash for text to use as cache key."""
    return hashlib.md5(text.encode()).hexdigest()


@lru_cache(maxsize=MAX_CACHE_SIZE)
def embed_text(text: str, model_name: str = MODEL_NAME, 
               use_cache: bool = True, normalize: bool = True) -> np.ndarray:
    """
    Embed a single text with caching support.
    
    Args:
        text: Text to embed
        model_name: Model to use for embedding
        use_cache: Whether to use caching
        normalize: Whether to normalize the embedding
    
    Returns:
        Embedding vector as numpy array
    """
    if not text:
        model = get_model(model_name)
        return np.zeros(model.get_sentence_embedding_dimension())
    
    # Check cache first
    if use_cache:
        cache_key = f"{model_name}_{get_text_hash(text)}"
        cache_file = os.path.join(CACHE_DIR, f"{cache_key}.pkl")
        
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
            except:
                pass  # If cache read fails, compute embedding
    
    # Preprocess text
    processed_text = preprocess_text(text)
    
    # Generate embedding
    model = get_model(model_name)
    embedding = model.encode(processed_text, show_progress_bar=False, convert_to_numpy=True)
    
    # Normalize if requested
    if normalize:
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
    
    # Cache the result
    if use_cache:
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(embedding, f)
        except:
            pass  # Ignore cache write errors
    
    return embedding


def embed_texts(texts: List[str], model_name: str = MODEL_NAME, 
                batch_size: int = 32, show_progress: bool = False,
                normalize: bool = True) -> np.ndarray:
    """
    Embed a list of texts with batch processing.
    
    Args:
        texts: List of texts to embed
        model_name: Model to use for embedding
        batch_size: Batch size for processing
        show_progress: Whether to show progress bar
        normalize: Whether to normalize embeddings
    
    Returns:
        Matrix of embeddings (num_texts x embedding_dim)
    """
    if not texts:
        return np.array([])
    
    # Preprocess all texts
    processed_texts = [preprocess_text(text) for text in texts]
    
    # Remove empty texts but keep track of indices
    non_empty_indices = [i for i, text in enumerate(processed_texts) if text]
    non_empty_texts = [processed_texts[i] for i in non_empty_indices]
    
    if not non_empty_texts:
        model = get_model(model_name)
        dim = model.get_sentence_embedding_dimension()
        return np.zeros((len(texts), dim))
    
    # Generate embeddings
    model = get_model(model_name)
    embeddings = model.encode(
        non_empty_texts, 
        batch_size=batch_size,
        show_progress_bar=show_progress,
        convert_to_numpy=True
    )
    
    # Normalize if requested
    if normalize:
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = np.divide(embeddings, norms, where=norms != 0)
    
    # Reconstruct full embedding matrix with zeros for empty texts
    model_dim = embeddings.shape[1]
    full_embeddings = np.zeros((len(texts), model_dim))
    full_embeddings[non_empty_indices] = embeddings
    
    return full_embeddings


def calculate_similarity(embedding1: np.ndarray, embedding2: np.ndarray) -> float:
    """
    Calculate cosine similarity between two embeddings.
    """
    if embedding1.ndim == 1:
        embedding1 = embedding1.reshape(1, -1)
    if embedding2.ndim == 1:
        embedding2 = embedding2.reshape(1, -1)
    
    similarity = cosine_similarity(embedding1, embedding2)[0, 0]
    return float(similarity)


def calculate_similarity_matrix(embeddings: np.ndarray) -> np.ndarray:
    """
    Calculate pairwise similarity matrix for a set of embeddings.
    """
    return cosine_similarity(embeddings)


def find_most_similar(query_embedding: np.ndarray, 
                     candidate_embeddings: np.ndarray, 
                     top_k: int = 5) -> List[Tuple[int, float]]:
    """
    Find most similar embeddings from candidates.
    
    Returns:
        List of (index, similarity_score) tuples
    """
    if query_embedding.ndim == 1:
        query_embedding = query_embedding.reshape(1, -1)
    
    similarities = cosine_similarity(query_embedding, candidate_embeddings)[0]
    
    # Get top k indices
    top_indices = np.argsort(similarities)[::-1][:top_k]
    
    # Return index and score pairs
    return [(int(idx), float(similarities[idx])) for idx in top_indices]


def embed_text_chunks(text: str, chunk_size: int = 512, 
                     overlap: int = 50, model_name: str = MODEL_NAME) -> np.ndarray:
    """
    Embed long text by chunking with overlap.
    Returns average of chunk embeddings.
    """
    if not text:
        model = get_model(model_name)
        return np.zeros(model.get_sentence_embedding_dimension())
    
    # Split into words
    words = text.split()
    
    if len(words) <= chunk_size:
        return embed_text(text, model_name=model_name)
    
    # Create chunks with overlap
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = ' '.join(words[i:i + chunk_size])
        chunks.append(chunk)
    
    # Embed all chunks
    chunk_embeddings = embed_texts(chunks, model_name=model_name)
    
    # Return average embedding
    return np.mean(chunk_embeddings, axis=0)


def get_model_info(model_name: str = MODEL_NAME) -> Dict[str, any]:
    """
    Get information about the model.
    """
    model = get_model(model_name)
    
    return {
        "model_name": model_name,
        "embedding_dimension": model.get_sentence_embedding_dimension(),
        "max_sequence_length": model.max_seq_length,
        "model_card": str(model)
    }


def clear_cache():
    """Clear the embedding cache."""
    import shutil
    
    if os.path.exists(CACHE_DIR):
        shutil.rmtree(CACHE_DIR)
        os.makedirs(CACHE_DIR)
        print(f"Cleared embedding cache at {CACHE_DIR}")


def reduce_embedding_dimension(embeddings: np.ndarray, 
                             target_dim: int = 128, 
                             method: str = 'pca') -> Tuple[np.ndarray, any]:
    """
    Reduce embedding dimensions for storage or visualization.
    
    Args:
        embeddings: Original embeddings
        target_dim: Target dimension
        method: Reduction method ('pca' or 'tsne')
    
    Returns:
        Reduced embeddings and the fitted reducer
    """
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    
    if embeddings.shape[1] <= target_dim:
        return embeddings, None
    
    if method == 'pca':
        reducer = PCA(n_components=target_dim)
    elif method == 'tsne':
        reducer = TSNE(n_components=min(target_dim, 3), random_state=42)
    else:
        raise ValueError(f"Unknown reduction method: {method}")
    
    reduced = reducer.fit_transform(embeddings)
    return reduced, reducer


# Specialized embedding functions for different content types

def embed_resume(resume_data: Dict[str, any], model_name: str = MODEL_NAME) -> np.ndarray:
    """
    Create a comprehensive embedding for a resume by combining different sections.
    """
    sections_weights = {
        "summary": 0.20,
        "skills": 0.30,
        "experience": 0.30,
        "projects": 0.10,
        "education": 0.10
    }
    
    embeddings = []
    weights = []
    
    # Summary
    if resume_data.get("summary"):
        embeddings.append(embed_text(resume_data["summary"], model_name=model_name))
        weights.append(sections_weights["summary"])
    
    # Skills
    if resume_data.get("skills"):
        skills_text = " ".join(resume_data["skills"])
        embeddings.append(embed_text(skills_text, model_name=model_name))
        weights.append(sections_weights["skills"])
    
    # Experience
    if resume_data.get("experience"):
        exp_texts = []
        for exp in resume_data["experience"]:
            if isinstance(exp, dict):
                exp_text = f"{exp.get('job_title', '')} {exp.get('description', '')}"
                exp_texts.append(exp_text)
        if exp_texts:
            exp_combined = " ".join(exp_texts[:3])  # Use top 3 experiences
            embeddings.append(embed_text(exp_combined, model_name=model_name))
            weights.append(sections_weights["experience"])
    
    # Projects
    if resume_data.get("projects"):
        projects_text = " ".join(resume_data["projects"][:3])  # Top 3 projects
        embeddings.append(embed_text(projects_text, model_name=model_name))
        weights.append(sections_weights["projects"])
    
    # Education
    if resume_data.get("education"):
        edu_text = " ".join(resume_data["education"])
        embeddings.append(embed_text(edu_text, model_name=model_name))
        weights.append(sections_weights["education"])
    
    if not embeddings:
        model = get_model(model_name)
        return np.zeros(model.get_sentence_embedding_dimension())
    
    # Weighted average of embeddings
    embeddings = np.array(embeddings)
    weights = np.array(weights)
    weights = weights / weights.sum()  # Normalize weights
    
    weighted_embedding = np.average(embeddings, axis=0, weights=weights)
    
    # Normalize final embedding
    norm = np.linalg.norm(weighted_embedding)
    if norm > 0:
        weighted_embedding = weighted_embedding / norm
    
    return weighted_embedding


def embed_job_description(job_data: Dict[str, str], model_name: str = MODEL_NAME) -> np.ndarray:
    """
    Create an embedding for a job description with weighted sections.
    """
    sections_weights = {
        "title": 0.25,
        "description": 0.35,
        "requirements": 0.30,
        "company": 0.10
    }
    
    embeddings = []
    weights = []
    
    # Job title (most important)
    if job_data.get("job_title"):
        embeddings.append(embed_text(job_data["job_title"], model_name=model_name))
        weights.append(sections_weights["title"])
    
    # Description
    if job_data.get("description"):
        embeddings.append(embed_text(job_data["description"], model_name=model_name))
        weights.append(sections_weights["description"])
    
    # Requirements
    if job_data.get("requirements"):
        embeddings.append(embed_text(job_data["requirements"], model_name=model_name))
        weights.append(sections_weights["requirements"])
    
    # Company (for context)
    if job_data.get("company"):
        embeddings.append(embed_text(job_data["company"], model_name=model_name))
        weights.append(sections_weights["company"])
    
    if not embeddings:
        model = get_model(model_name)
        return np.zeros(model.get_sentence_embedding_dimension())
    
    # Weighted average of embeddings
    embeddings = np.array(embeddings)
    weights = np.array(weights)
    weights = weights / weights.sum()  # Normalize weights
    
    weighted_embedding = np.average(embeddings, axis=0, weights=weights)
    
    # Normalize final embedding
    norm = np.linalg.norm(weighted_embedding)
    if norm > 0:
        weighted_embedding = weighted_embedding / norm
    
    return weighted_embedding


def semantic_search(query: str, documents: List[str], 
                   top_k: int = 5, threshold: float = 0.0) -> List[Tuple[int, float, str]]:
    """
    Perform semantic search on a list of documents.
    
    Args:
        query: Search query
        documents: List of documents to search
        top_k: Number of top results to return
        threshold: Minimum similarity threshold
    
    Returns:
        List of (index, similarity_score, document) tuples
    """
    if not documents:
        return []
    
    # Embed query and documents
    query_embedding = embed_text(query)
    doc_embeddings = embed_texts(documents)
    
    # Calculate similarities
    similarities = cosine_similarity(query_embedding.reshape(1, -1), doc_embeddings)[0]
    
    # Get indices sorted by similarity
    sorted_indices = np.argsort(similarities)[::-1]
    
    # Filter by threshold and get top k
    results = []
    for idx in sorted_indices[:top_k]:
        if similarities[idx] >= threshold:
            results.append((int(idx), float(similarities[idx]), documents[idx]))
    
    return results


def get_embedding_statistics(embeddings: np.ndarray) -> Dict[str, float]:
    """
    Calculate statistics for a set of embeddings.
    """
    if embeddings.size == 0:
        return {}
    
    return {
        "mean_norm": float(np.mean(np.linalg.norm(embeddings, axis=1))),
        "std_norm": float(np.std(np.linalg.norm(embeddings, axis=1))),
        "mean_similarity": float(np.mean(calculate_similarity_matrix(embeddings))),
        "min_similarity": float(np.min(calculate_similarity_matrix(embeddings))),
        "max_similarity": float(np.max(calculate_similarity_matrix(embeddings))),
        "embedding_dim": embeddings.shape[1],
        "num_embeddings": embeddings.shape[0]
    }


def augment_text_for_embedding(text: str, augmentation_type: str = "paraphrase") -> List[str]:
    """
    Augment text to create variations for more robust embeddings.
    
    Args:
        text: Original text
        augmentation_type: Type of augmentation ('paraphrase', 'synonym', 'noise')
    
    Returns:
        List of augmented texts including original
    """
    augmented = [text]  # Always include original
    
    if augmentation_type == "paraphrase":
        # Simple paraphrasing by reordering sentences
        sentences = text.split('. ')
        if len(sentences) > 1:
            import random
            shuffled = sentences.copy()
            random.shuffle(shuffled)
            augmented.append('. '.join(shuffled))
    
    elif augmentation_type == "synonym":
        # Replace some words with common alternatives
        synonyms = {
            "develop": "create",
            "create": "build",
            "manage": "oversee",
            "implement": "deploy",
            "analyze": "examine",
            "design": "architect"
        }
        augmented_text = text
        for word, synonym in synonyms.items():
            if word in augmented_text.lower():
                augmented_text = augmented_text.replace(word, synonym)
        if augmented_text != text:
            augmented.append(augmented_text)
    
    elif augmentation_type == "noise":
        # Add minor noise by removing some stop words
        stop_words = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for"}
        words = text.split()
        filtered = [w for w in words if w.lower() not in stop_words]
        if len(filtered) < len(words):
            augmented.append(' '.join(filtered))
    
    return augmented


def create_query_expansion(query: str, expansion_terms: int = 3) -> str:
    """
    Expand query with related terms for better search results.
    """
    # Common expansions for job-related queries
    expansions = {
        "developer": ["programmer", "engineer", "coder"],
        "ml": ["machine learning", "ai", "artificial intelligence"],
        "data": ["analytics", "analysis", "insights"],
        "senior": ["lead", "principal", "experienced"],
        "junior": ["entry level", "associate", "beginner"],
        "full stack": ["fullstack", "full-stack", "frontend backend"],
        "backend": ["server side", "api", "database"],
        "frontend": ["client side", "ui", "user interface"]
    }
    
    expanded_query = query
    words = query.lower().split()
    
    added_terms = []
    for word in words:
        if word in expansions and len(added_terms) < expansion_terms:
            added_terms.extend(expansions[word][:expansion_terms - len(added_terms)])
    
    if added_terms:
        expanded_query = f"{query} {' '.join(added_terms)}"
    
    return expanded_query


def batch_process_embeddings(texts: List[str], 
                           batch_size: int = 32,
                           save_path: Optional[str] = None) -> np.ndarray:
    """
    Process large number of texts in batches and optionally save.
    """
    all_embeddings = []
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        batch_embeddings = embed_texts(batch, show_progress=True)
        all_embeddings.append(batch_embeddings)
        
        print(f"Processed batch {i//batch_size + 1}/{(len(texts) + batch_size - 1)//batch_size}")
    
    # Combine all embeddings
    final_embeddings = np.vstack(all_embeddings)
    
    # Save if path provided
    if save_path:
        np.save(save_path, final_embeddings)
        print(f"Saved embeddings to {save_path}")
    
    return final_embeddings


# Test and utility functions
def test_embedding_quality(test_pairs: List[Tuple[str, str, bool]]) -> Dict[str, float]:
    """
    Test embedding quality with known similar/dissimilar pairs.
    
    Args:
        test_pairs: List of (text1, text2, should_be_similar) tuples
    
    Returns:
        Dictionary with accuracy metrics
    """
    correct = 0
    similarities = []
    
    for text1, text2, should_be_similar in test_pairs:
        emb1 = embed_text(text1)
        emb2 = embed_text(text2)
        similarity = calculate_similarity(emb1, emb2)
        similarities.append(similarity)
        
        # Threshold for similarity
        threshold = 0.7
        is_similar = similarity >= threshold
        
        if is_similar == should_be_similar:
            correct += 1
    
    accuracy = correct / len(test_pairs) if test_pairs else 0
    
    return {
        "accuracy": accuracy,
        "mean_similarity": np.mean(similarities),
        "std_similarity": np.std(similarities),
        "min_similarity": np.min(similarities),
        "max_similarity": np.max(similarities)
    }
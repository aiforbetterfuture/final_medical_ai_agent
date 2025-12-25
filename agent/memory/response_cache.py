"""
Response Cache: Semantic similarity-based response caching

Caches responses and retrieves them when similar questions are asked,
reducing LLM calls and improving response time.
"""

from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import random
import re


@dataclass
class CachedResponse:
    """Represents a cached query-response pair."""
    query: str
    response: str
    embedding: Optional[List[float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    access_count: int = 0


class ResponseCache:
    """
    Semantic similarity-based response cache.

    Uses embeddings to find similar questions and return cached responses.
    """

    def __init__(
        self,
        max_cache_size: int = 100,
        similarity_threshold: float = 0.85,
        cache_ttl_minutes: int = 60,
        model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    ):
        """
        Initialize response cache.

        Args:
            max_cache_size: Maximum number of cached responses
            similarity_threshold: Minimum similarity score for cache hit (0-1)
            cache_ttl_minutes: Time-to-live for cached responses in minutes
            model_name: Sentence transformer model for embeddings
        """
        self.max_cache_size = max_cache_size
        self.similarity_threshold = similarity_threshold
        self.cache_ttl = timedelta(minutes=cache_ttl_minutes)
        self.model_name = model_name

        self.cache: List[CachedResponse] = []

        # Statistics
        self.stats = {
            'total_queries': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'tokens_saved': 0,
            'time_saved_ms': 0
        }

        # Initialize embedding model lazily
        self._embedding_model = None

    def _get_embedding_model(self):
        """Lazy load embedding model."""
        if self._embedding_model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._embedding_model = SentenceTransformer(self.model_name)
            except ImportError:
                # Fallback: use simple text-based similarity
                self._embedding_model = "fallback"

        return self._embedding_model

    def _compute_embedding(self, text: str) -> Optional[List[float]]:
        """Compute embedding for text."""
        model = self._get_embedding_model()

        if model == "fallback":
            # Simple fallback: character n-gram based pseudo-embedding
            return None

        try:
            embedding = model.encode(text, convert_to_tensor=False)
            return embedding.tolist() if hasattr(embedding, 'tolist') else list(embedding)
        except Exception as e:
            print(f"[WARNING] Embedding computation failed: {e}")
            return None

    def _compute_similarity(self, emb1: Optional[List[float]], emb2: Optional[List[float]], text1: str, text2: str) -> float:
        """Compute similarity between two embeddings or texts."""
        if emb1 is not None and emb2 is not None:
            # Cosine similarity
            try:
                import numpy as np
                dot_product = np.dot(emb1, emb2)
                norm1 = np.linalg.norm(emb1)
                norm2 = np.linalg.norm(emb2)
                if norm1 == 0 or norm2 == 0:
                    return 0.0
                return float(dot_product / (norm1 * norm2))
            except ImportError:
                # Fallback without numpy
                dot = sum(a * b for a, b in zip(emb1, emb2))
                norm1 = sum(a * a for a in emb1) ** 0.5
                norm2 = sum(b * b for b in emb2) ** 0.5
                if norm1 == 0 or norm2 == 0:
                    return 0.0
                return dot / (norm1 * norm2)

        # Fallback: simple text-based similarity
        return self._simple_text_similarity(text1, text2)

    def _simple_text_similarity(self, text1: str, text2: str) -> float:
        """Simple Jaccard similarity on word tokens."""
        words1 = set(re.findall(r'\w+', text1.lower()))
        words2 = set(re.findall(r'\w+', text2.lower()))

        if not words1 or not words2:
            return 0.0

        intersection = words1 & words2
        union = words1 | words2

        return len(intersection) / len(union) if union else 0.0

    def find_similar(self, query: str) -> Optional[Tuple[CachedResponse, float]]:
        """
        Find the most similar cached query.

        Args:
            query: User query to search for

        Returns:
            Tuple of (CachedResponse, similarity_score) if found, None otherwise
        """
        self.stats['total_queries'] += 1

        if not self.cache:
            self.stats['cache_misses'] += 1
            return None

        # Remove expired entries
        self._cleanup_expired()

        # Compute query embedding
        query_embedding = self._compute_embedding(query)

        # Find most similar
        best_match = None
        best_score = 0.0

        for cached in self.cache:
            similarity = self._compute_similarity(
                query_embedding,
                cached.embedding,
                query,
                cached.query
            )

            if similarity > best_score:
                best_score = similarity
                best_match = cached

        if best_match and best_score >= self.similarity_threshold:
            best_match.access_count += 1
            self.stats['cache_hits'] += 1
            return (best_match, best_score)

        self.stats['cache_misses'] += 1
        return None

    def add(self, query: str, response: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Add a query-response pair to cache.

        Args:
            query: User query
            response: Agent response
            metadata: Optional metadata
        """
        # Don't cache empty responses
        if not response or not response.strip():
            return

        # Compute embedding
        embedding = self._compute_embedding(query)

        cached = CachedResponse(
            query=query,
            response=response,
            embedding=embedding,
            metadata=metadata or {},
            timestamp=datetime.now()
        )

        self.cache.append(cached)

        # Evict oldest if over capacity
        if len(self.cache) > self.max_cache_size:
            # Sort by access count (ascending) and age (oldest first)
            self.cache.sort(key=lambda x: (x.access_count, x.timestamp))
            self.cache.pop(0)

    def _cleanup_expired(self) -> None:
        """Remove expired cache entries."""
        now = datetime.now()
        self.cache = [
            c for c in self.cache
            if now - c.timestamp < self.cache_ttl
        ]

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        hit_rate = 0.0
        if self.stats['total_queries'] > 0:
            hit_rate = self.stats['cache_hits'] / self.stats['total_queries']

        return {
            **self.stats,
            'cache_size': len(self.cache),
            'hit_rate': hit_rate
        }

    def update_stats(self, tokens_saved: int = 0, time_saved_ms: int = 0) -> None:
        """Update cache statistics."""
        self.stats['tokens_saved'] += tokens_saved
        self.stats['time_saved_ms'] += time_saved_ms

    def clear(self) -> None:
        """Clear all cached responses."""
        self.cache.clear()


class ResponseStyleVariator:
    """
    Applies stylistic variations to cached responses.

    Prevents repetitive responses by varying phrasing while maintaining meaning.
    """

    def __init__(self):
        """Initialize style variator."""
        self.variation_templates = {
            'prefix': [
                "",
                "요약하자면, ",
                "간단히 말씀드리면, ",
                "정리하면, ",
            ],
            'connector': [
                ". ",
                ". 또한, ",
                ". 추가로, ",
                ". 그리고 ",
            ],
            'suffix': [
                "",
                " 도움이 되셨기를 바랍니다.",
                " 추가 질문이 있으시면 말씀해 주세요.",
                "",
            ]
        }

    def vary_response(self, response: str, variation_level: float = 0.3) -> str:
        """
        Apply stylistic variations to response.

        Args:
            response: Original response text
            variation_level: Level of variation (0.0 = none, 1.0 = maximum)

        Returns:
            Varied response text
        """
        if not response or variation_level <= 0:
            return response

        # For low variation, just return original
        if random.random() > variation_level:
            return response

        varied = response

        # Apply random prefix
        if random.random() < variation_level * 0.3:
            prefix = random.choice(self.variation_templates['prefix'])
            varied = prefix + varied

        # Apply random suffix
        if random.random() < variation_level * 0.3:
            suffix = random.choice(self.variation_templates['suffix'])
            varied = varied.rstrip('.!? ') + suffix

        # Sentence reordering for higher variation levels
        if variation_level > 0.6 and random.random() < 0.3:
            sentences = re.split(r'([.!?]+\s+)', varied)
            # Recombine sentences (simple implementation)
            if len(sentences) > 3:
                # Just shuffle middle sentences slightly
                middle = sentences[1:-1]
                random.shuffle(middle)
                varied = sentences[0] + ''.join(middle) + sentences[-1]

        return varied
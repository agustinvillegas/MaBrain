import sys
import time
import numpy as np
import torch

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

def _log(msg):
    print(f"  [EB] {msg}", flush=True)


def _char_ngrams(word, n=3):
    return {word[i:i + n] for i in range(len(word) - n + 1)}


def _jaccard(a, b):
    if not a or not b:
        return 0.0
    intersection = a & b
    union = a | b
    return len(intersection) / max(len(union), 1)


class EmbeddingBridge:

    def __init__(self, model_name=None, max_cache=10000):
        self.model_name = model_name or DEFAULT_MODEL
        self.model = None
        self.model_loaded = False
        self._embedding_cache = {}    # concept -> np.array(384,)
        self._cache_order = []        # LRU tracking
        self._max_cache = max_cache

    def _ensure_model(self):
        if self.model_loaded:
            return True
        t0 = time.time()
        _log(f"_ensure_model START")
        try:
            _log(f"importing sentence_transformers... t={time.time()-t0:.1f}s")
            from sentence_transformers import SentenceTransformer
            _log(f"imported OK t={time.time()-t0:.1f}s")

            _log(f"torch.cuda.is_available()... t={time.time()-t0:.1f}s")
            device = "cuda" if torch.cuda.is_available() else "cpu"
            _log(f"device={device} t={time.time()-t0:.1f}s")

            _log(f"SentenceTransformer({self.model_name}, device={device})... t={time.time()-t0:.1f}s")
            self.model = SentenceTransformer(self.model_name, device=device)
            self.model_loaded = True
            _log(f"Modelo en: {device.upper()} ({time.time()-t0:.1f}s)")
            return True
        except Exception as e:
            _log(f"FAILED ({time.time()-t0:.1f}s): {e}")
            return False

    def get_embedding(self, concept):
        """Single-concept embedding with LRU cache."""
        if concept in self._embedding_cache:
            return self._embedding_cache[concept]
        if not self._ensure_model():
            return None
        try:
            emb = self.model.encode([concept], convert_to_numpy=True,
                                    normalize_embeddings=True,
                                    show_progress_bar=False)[0]
            # LRU evict
            if len(self._embedding_cache) >= self._max_cache:
                old = self._cache_order.pop(0)
                del self._embedding_cache[old]
            self._embedding_cache[concept] = emb
            self._cache_order.append(concept)
            return emb
        except Exception:
            return None

    def encode(self, words, batch_size=256):
        if not self._ensure_model():
            return None
        if not words:
            return None
        # Check cache for each word, encode only uncached
        to_encode = [w for w in words if w not in self._embedding_cache]
        if to_encode:
            try:
                new_embs = self.model.encode(
                    to_encode, convert_to_numpy=True, normalize_embeddings=True,
                    batch_size=batch_size, show_progress_bar=False,
                )
                for w, emb in zip(to_encode, new_embs):
                    if len(self._embedding_cache) >= self._max_cache:
                        old = self._cache_order.pop(0)
                        del self._embedding_cache[old]
                    self._embedding_cache[w] = emb
                    self._cache_order.append(w)
            except Exception:
                pass
        # Build result from cache
        result = []
        for w in words:
            if w in self._embedding_cache:
                result.append(self._embedding_cache[w])
            else:
                return None  # encoding failed for this word
        return np.array(result)

    def _char_sim(self, a, b):
        na = _char_ngrams(a, 2) | _char_ngrams(a, 3)
        nb = _char_ngrams(b, 2) | _char_ngrams(b, 3)
        return _jaccard(na, nb)

    def _char_sim_top_k(self, word, candidates, k=8):
        """Cheap pre-filter: return top-k candidates by char-ngram similarity (no MiniLM)."""
        scored = [(c, self._char_sim(word, c)) for c in candidates]
        scored.sort(key=lambda x: -x[1])
        return [c for c, s in scored[:k]]

    def prewarm(self, words, max_words=2000):
        """Pre-compute embeddings for known words (up to max_words).
        
        Full prewarm (67k words) is too slow on GTX 1050 Ti (~3 min).
        We cache the first 2000 most common concepts — enough to accelerate 
        the 2-stage filter without the startup cost.
        """
        if not words:
            return
        if not self._ensure_model():
            return
        subset = words[:max_words]
        _log(f"prewarm: encoding {len(subset)}/{len(words)} words...")
        t0 = time.time()
        self.encode(subset)
        _log(f"prewarm done ({time.time()-t0:.1f}s)")

    def similarity(self, word_a, word_b):
        if not self.model_loaded:
            return self._char_sim(word_a, word_b)
        # Batch encode [a, b] in one call instead of 2 separate get_embedding
        embs = self.encode([word_a, word_b])
        if embs is None or len(embs) != 2:
            return self._char_sim(word_a, word_b)
        char_sim = self._char_sim(word_a, word_b)
        model_sim = float(np.dot(embs[0], embs[1]))
        return min(max(model_sim, char_sim), 1.0)

    def closest(self, word, candidates, top_k=5, min_score=0.0):
        if not candidates:
            return []

        # Pre-filter with cheap char-ngram to reduce MiniLM calls
        if self.model_loaded and len(candidates) > 16:
            candidates = self._char_sim_top_k(word, candidates, k=max(top_k * 3, 16))

        scored = []
        for c in candidates:
            char_sim = self._char_sim(word, c)
            emb_c = self.get_embedding(c)
            if emb_c is not None:
                emb_w = self.get_embedding(word)
                if emb_w is not None:
                    model_sim = float(np.dot(emb_w, emb_c))
                    sim = max(model_sim, char_sim)
                else:
                    sim = char_sim
            else:
                sim = char_sim
            scored.append((c, sim))

        scored.sort(key=lambda x: -x[1])
        return [(c, s) for c, s in scored if s >= min_score][:top_k]

    def closest_cell_word(self, word, brain, top_k=1, min_score=0.0):
        known = [c.word for c in brain.cells.values() if c.word]
        if not known:
            return None
        matches = self.closest(word, known, top_k=top_k, min_score=min_score)
        if not matches:
            return None
        return matches[0]

import numpy as np
import torch

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def _char_ngrams(word, n=3):
    return {word[i:i + n] for i in range(len(word) - n + 1)}


def _jaccard(a, b):
    if not a or not b:
        return 0.0
    intersection = a & b
    union = a | b
    return len(intersection) / max(len(union), 1)


class EmbeddingBridge:

    def __init__(self, model_name=None):
        self.model_name = model_name or DEFAULT_MODEL
        self.model = None
        self.model_loaded = False

    def _ensure_model(self):
        if self.model_loaded:
            return True
        try:
            from sentence_transformers import SentenceTransformer
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model = SentenceTransformer(self.model_name, device=device)
            self.model_loaded = True
            print(f"  [EmbeddingBridge] Modelo en: {device.upper()}")
            return True
        except Exception as e:
            print(f"  [EmbeddingBridge] MiniLM load failed: {e}")
            return False

    def encode(self, words, batch_size=256):
        if not self._ensure_model():
            return None
        if not words:
            return None
        try:
            return self.model.encode(
                list(words),
                convert_to_numpy=True,
                normalize_embeddings=True,
                batch_size=batch_size,
                show_progress_bar=False,
            )
        except Exception:
            return None

    def _char_sim(self, a, b):
        na = _char_ngrams(a, 2) | _char_ngrams(a, 3)
        nb = _char_ngrams(b, 2) | _char_ngrams(b, 3)
        return _jaccard(na, nb)

    def similarity(self, word_a, word_b):
        char_sim = self._char_sim(word_a, word_b)
        emb = self.encode([word_a, word_b])
        if emb is not None:
            model_sim = float(np.dot(emb[0], emb[1]))
            return min(max(model_sim, char_sim), 1.0)
        return char_sim

    def closest(self, word, candidates, top_k=5, min_score=0.0):
        if not candidates:
            return []

        # Compute scores: max(model_sim, char_sim) for each candidate
        embs = None
        if self.model_loaded:
            all_words = [word] + list(candidates)
            embs = self.encode(all_words)

        scored = []
        for i, c in enumerate(candidates):
            char_sim = self._char_sim(word, c)
            if embs is not None and i + 1 < len(embs):
                model_sim = float(np.dot(embs[0], embs[i + 1]))
                sim = max(model_sim, char_sim)
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

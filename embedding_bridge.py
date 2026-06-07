import os
import math



def _char_ngrams(word, n=3):
    return {word[i:i + n] for i in range(len(word) - n + 1)}


def _jaccard(a, b):
    if not a or not b:
        return 0.0
    intersection = a & b
    union = a | b
    return len(intersection) / max(len(union), 1)


class EmbeddingBridge:

    def __init__(self, model_path=None):
        self.model = None
        self.model_loaded = False
        self._use_wv = False

        if model_path and os.path.exists(model_path):
            self._load_model(model_path)

    def _load_model(self, path):
        try:
            from gensim.models.fasttext import FastText

            try:
                from gensim.models.fasttext import load_facebook_model
                self.model = load_facebook_model(path)
                self.model_loaded = True
                self._use_wv = True
                return
            except Exception:
                pass

            self.model = FastText.load(path)
            self.model_loaded = True
            self._use_wv = True
        except Exception:
            try:
                import fasttext
                self.model = fasttext.load_model(path)
                self.model_loaded = True
                self._use_wv = False
            except Exception:
                pass

    def _get_vec(self, word):
        try:
            if self._use_wv:
                return self.model.wv[word]
            return self.model[word]
        except Exception:
            return None

    def similarity(self, word_a, word_b):

        ngrams_a = _char_ngrams(word_a, 2) | _char_ngrams(word_a, 3)
        ngrams_b = _char_ngrams(word_b, 2) | _char_ngrams(word_b, 3)
        char_sim = _jaccard(ngrams_a, ngrams_b)

        if self.model_loaded:
            vec_a = self._get_vec(word_a)
            vec_b = self._get_vec(word_b)
            if vec_a is not None and vec_b is not None:
                try:
                    dot = sum(float(va) * float(vb) for va, vb in zip(vec_a, vec_b))
                    norm_a = math.sqrt(sum(float(v) ** 2 for v in vec_a))
                    norm_b = math.sqrt(sum(float(v) ** 2 for v in vec_b))
                    if norm_a * norm_b > 0:
                        model_sim = dot / (norm_a * norm_b)
                        return max(model_sim, char_sim)
                except Exception:
                    pass

        return char_sim

    def closest(self, word, candidates, top_k=5, min_score=0.0):

        if not candidates:
            return []

        scored = [(c, self.similarity(word, c)) for c in candidates]
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


import json
import os
import sys
import time
import math
from collections import defaultdict, Counter    


# ── Char n-gram utilities ─────────────────────────────────────────
def char_ngrams(word, n=3):
    return {word[i:i + n] for i in range(len(word) - n + 1)}


def char_ngrams_multi(word):
    """Combined bigrams + trigrams for better coverage."""
    return char_ngrams(word, 2) | char_ngrams(word, 3)


def jaccard(a, b):
    if not a or not b:
        return 0.0
    intersection = a & b
    union = a | b
    return len(intersection) / max(len(union), 1)


# ── Disjoint Set (Union-Find) ─────────────────────────────────────
class DisjointSet: 
    def __init__(self):
        self.parent = {}
        self.rank = {}

    def find(self, x):
        if x not in self.parent:
            self.parent[x] = x
            self.rank[x] = 0
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            self.parent[ra] = rb
        elif self.rank[ra] > self.rank[rb]:
            self.parent[rb] = ra
        else:
            self.parent[rb] = ra
            self.rank[ra] += 1


# ── Load triplets ─────────────────────────────────────────────────
def load_triplets(paths):
    """Load JSONL triplets from multiple files."""
    records = []
    for path in paths:
        if not os.path.exists(path):
            print(f"  WARNING: file not found, skipping: {path}")
            continue
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    # Normalize
                    rec["from"] = rec["from"].lower().replace("-", "_").strip()
                    rec["to"] = rec["to"].lower().replace("-", "_").strip()
                    rec["relation"] = rec["relation"].upper().strip()
                    if len(rec["from"]) >= 2 and len(rec["to"]) >= 2:
                        records.append(rec)
                except (json.JSONDecodeError, KeyError):
                    continue
    return records


# ── Char-ngram based concept clustering ──────────────────────────
def cluster_by_char_ngrams(records, threshold=0.55):
    """Group similar concept labels using char n-gram Jaccard similarity.
    Returns DisjointSet mapping each variant → canonical cluster root.
    """
    # Collect all unique concept words
    words = set()
    for rec in records:
        words.add(rec["from"])
        words.add(rec["to"])

    words = sorted(words)
    print(f"  Unique concept words: {len(words):,}")

    # Precompute n-grams
    word_ngrams = {w: char_ngrams_multi(w) for w in words}

    # Simple blocking: group by first 2 chars, then compare within each block
    blocks = defaultdict(list)
    for w in words:
        block_key = w[:2] if len(w) >= 2 else w
        blocks[block_key].append(w)

    ds = DisjointSet()
    comparisons = 0
    merges = 0

    for block_key, block_words in blocks.items():
        for i in range(len(block_words)):
            for j in range(i + 1, len(block_words)):
                comparisons += 1
                a, b = block_words[i], block_words[j]
                sim = jaccard(word_ngrams[a], word_ngrams[b])
                if sim >= threshold:
                    ds.union(a, b)
                    merges += 1

    print(f"  Char-ngram comparisons: {comparisons:,}")
    print(f"  Merges: {merges:,}")
    print(f"  Canonical concepts: {len({ds.find(w) for w in words}):,}")

    return ds


# ── Embedding-based concept clustering ───────────────────────────
def cluster_by_embeddings(records, threshold=0.85):
    """Use EmbeddingBridge (MiniLM) to refine char-ngram clusters."""
    from embedding_bridge import EmbeddingBridge
    bridge = EmbeddingBridge()

    words = sorted(set(
        rec["from"] for rec in records
    ) | set(rec["to"] for rec in records))
    print(f"  Computing embeddings for {len(words):,} words...")

    # Load model and encode
    if not bridge._ensure_model():
        print("  WARNING: MiniLM not available, using char-ngram only")
        return None

    embs = bridge.encode(words, batch_size=128)
    if embs is None:
        print("  WARNING: embedding failed, using char-ngram only")
        return None

    import numpy as np
    # Normalize
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    norms[norms == 0] = 1
    embs = embs / norms

    # Greedy clustering: iterate, for each word find closest prototype
    prototypes = []
    proto_words = []
    proto_embs = []  # list of arrays
    cluster_map = {}  # word -> root

    for word, emb in zip(words, embs):
        if not proto_embs:
            prototypes.append(word)
            proto_words.append(word)
            proto_embs.append(emb)
            cluster_map[word] = word
            continue

        # Find closest prototype
        sims = np.dot(proto_embs, emb)
        best_idx = np.argmax(sims)
        best_sim = float(sims[best_idx])

        if best_sim >= threshold:
            cluster_map[word] = proto_words[best_idx]
        else:
            prototypes.append(word)
            proto_words.append(word)
            proto_embs.append(emb)
            cluster_map[word] = word

    print(f"  Embedding clusters: {len(prototypes):,}")
    print(f"  Reduction: {len(words)} -> {len(prototypes)} ({(1-len(prototypes)/len(words))*100:.0f}%)")

    return cluster_map


# ── Build canonical mapping ───────────────────────────────────────
def build_canonical_map(records, char_clusters=None, embed_clusters=None):
    """Build final mapping from variant → canonical label.

    Priority: embed mapping > char-ngram mapping > raw word.
    """
    words = sorted(set(
        rec["from"] for rec in records
    ) | set(rec["to"] for rec in records))

    # Count frequency of each word across all triplets
    freq = Counter()
    for rec in records:
        freq[rec["from"]] += 1
        freq[rec["to"]] += 1

    # Build mapping
    variant_to_canonical = {}

    if embed_clusters:
        # Use embedding clusters directly
        for word in words:
            root = embed_clusters.get(word, word)
            variant_to_canonical[word] = root
    elif char_clusters:
        # Within each char-ngram cluster, pick the most frequent word as canonical
        cluster_groups = defaultdict(list)
        for word in words:
            root = char_clusters.find(word)
            cluster_groups[root].append(word)

        for root, members in cluster_groups.items():
            # Pick most frequent; tiebreak by length (shorter = more canonical)
            canonical = max(members, key=lambda w: (freq.get(w, 0), -len(w)))
            for member in members:
                variant_to_canonical[member] = canonical
    else:
        # No clustering: each word is its own canonical
        for word in words:
            variant_to_canonical[word] = word

    return variant_to_canonical


# ── Remap triplets + dedup ────────────────────────────────────────
def canonicalize_triplets(records, variant_map):
    """Remap all triplet words to canonical forms, deduplicate."""
    canonical_records = set()
    stats = defaultdict(int)

    for rec in records:
        from_c = variant_map.get(rec["from"], rec["from"])
        to_c = variant_map.get(rec["to"], rec["to"])

        if from_c == to_c:
            continue  # skip self-loops

        # Deduplicate: use tuple as set key
        key = (from_c, rec["relation"], to_c)
        canonical_records.add(key)

        # Track sources with dedup
        stats["total_raw"] += 1

    print(f"  Raw triplets: {stats['total_raw']:,}")
    print(f"  Canonical triplets (deduped): {len(canonical_records):,}")
    print(f"  Reduction: {(1 - len(canonical_records)/max(stats['total_raw'],1))*100:.0f}%")

    # Convert to list of dicts
    result = []
    for from_c, rel, to_c in sorted(canonical_records):
        result.append({
            "from": from_c,
            "relation": rel,
            "to": to_c,
        })

    return result


# ── Stats ─────────────────────────────────────────────────────────
def print_stats(triplets):
    """Print relation distribution and coverage stats."""
    rel_counts = Counter(t["relation"] for t in triplets)
    concepts = set()
    for t in triplets:
        concepts.add(t["from"])
        concepts.add(t["to"])

    print(f"\n  --- Statistics ---")
    print(f"  Canonical concepts: {len(concepts):,}")
    print(f"  Total triplets: {len(triplets):,}")
    print(f"  Relation distribution:")
    for rel, cnt in rel_counts.most_common():
        print(f"    {rel}: {cnt:,}")
    print(f"  Avg outgoing degree: {len(triplets)/max(len(concepts),1):.1f}")


# ── Export to Layer 1 format ──────────────────────────────────────
def export_layer1(triplets, output_path):
    """Export in the TRAINING_DESIGN.md Layer 1 format."""
    concepts = set()
    for t in triplets:
        concepts.add(t["from"])
        concepts.add(t["to"])

    # For each concept, collect outgoing relations
    concept_relations = defaultdict(list)
    for t in triplets:
        concept_relations[t["from"]].append({
            "relation": t["relation"],
            "to": t["to"],
        })

    dataset = {
        "version": "v1.0",
        "description": f"MaBrain Capa 1: {len(triplets):,} triplets, {len(concepts):,} concepts",
        "triplets": triplets,
        "concepts": sorted(concepts),
        "relation_distribution": dict(Counter(t["relation"] for t in triplets).most_common()),
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)
    print(f"  Saved: {output_path} ({os.path.getsize(output_path)/1024/1024:.1f} MB)")


# ── Main ──────────────────────────────────────────────────────────
def main():
    import argparse
    parser = argparse.ArgumentParser(description="Canonicalize triplets into Layer 1 dataset")
    parser.add_argument("--input", "-i", nargs="+", required=True,
                        help="Input JSONL triplet files")
    parser.add_argument("--output", "-o", default="data/processed/layer1_triplets.json",
                        help="Output Layer 1 JSON path")
    parser.add_argument("--char-threshold", type=float, default=0.55,
                        help="Char-ngram Jaccard threshold (default: 0.55)")
    parser.add_argument("--embed", action="store_true",
                        help="Use MiniLM embedding clustering (slow, more accurate)")
    parser.add_argument("--embed-threshold", type=float, default=0.85,
                        help="Embedding cosine threshold (default: 0.85)")
    args = parser.parse_args()

    print("=== Canonicalization Pipeline ===\n")
    start_time = time.time()

    # 1. Load
    print("Loading triplets...")
    records = load_triplets(args.input)
    print(f"  Loaded: {len(records):,} records from {len(args.input)} file(s)")

    if not records:
        print("  No records loaded. Exiting.")
        sys.exit(1)

    # 2. Char-ngram clustering (fast)
    print("\nChar-ngram clustering...")
    char_ds = cluster_by_char_ngrams(records, threshold=args.char_threshold)

    # 3. Optional embedding clustering
    embed_map = None
    if args.embed:
        print("\nEmbedding clustering (MiniLM)...")
        embed_map = cluster_by_embeddings(records, threshold=args.embed_threshold)

    # 4. Build canonical mapping
    print("\nBuilding canonical mapping...")
    variant_map = build_canonical_map(records, char_clusters=char_ds, embed_clusters=embed_map)

    # 5. Remap + dedup
    print("\nCanonicalizing triplets...")
    canonical_triplets = canonicalize_triplets(records, variant_map)

    # 6. Stats
    print_stats(canonical_triplets)

    # 7. Export
    export_layer1(canonical_triplets, args.output)

    elapsed = time.time() - start_time
    print(f"\nTotal time: {elapsed/60:.1f} min")


if __name__ == "__main__":
    main()

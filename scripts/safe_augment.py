"""
Safe semantic augmentation using IS_A clusters: substitute only within same category.
"""
import json
import os
import sys
from collections import defaultdict

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
DATA_DIR = os.path.join(PROJECT_ROOT, "datasets")
from embedding_bridge import EmbeddingBridge

TARGET = 50000
SIM_MIN = 0.30


def load_base(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)["sequences"]


def build_clusters(sequences):
    """
    Build IS_A clusters: {parent: [child1, child2, ...]}
    Also build a concept->parents map for reverse lookup.
    """
    parent_to_children = defaultdict(set)
    concept_to_parents = defaultdict(set)
    for s in sequences:
        words = s["sequence"]
        rels = s.get("relations", [])
        for i in range(len(words) - 1):
            if rels and i < len(rels) and rels[i] == "IS_A":
                child, parent = words[i], words[i + 1]
                parent_to_children[parent].add(child)
                concept_to_parents[child].add(parent)
    return dict(parent_to_children), dict(concept_to_parents)


def build_cohort(clusters, concept_to_parents, concept):
    """Return all concepts that share a direct parent with this concept."""
    parents = concept_to_parents.get(concept, set())
    cohort = set()
    for p in parents:
        if p in clusters:
            for child in clusters[p]:
                if child != concept:
                    cohort.add(child)
    return cohort


def safe_augment(sequences, clusters, concept_to_parents, bridge, target=TARGET):
    """
    For each (A, B, R):
    - Find A' in same IS_A cluster as A with sim >= threshold
    - Generate (A', B, R)
    - Find B' in same IS_A cluster as B with sim >= threshold
    - Generate (A, B', R)
    """
    vocab = sorted({w for s in sequences for w in s["sequence"]})

    existing = set()
    for s in sequences:
        words = s["sequence"]
        rels = s.get("relations", [])
        for i in range(len(words) - 1):
            if rels and i < len(rels):
                existing.add((words[i], words[i + 1], rels[i]))

    new_seqs = []

    # Pre-compute cohorts for concepts that have IS_A parents
    cohort_map = {}
    for c in vocab:
        cohort = build_cohort(clusters, concept_to_parents, c)
        if cohort:
            cohort_map[c] = list(cohort)

    for idx, s in enumerate(sequences):
        words = s["sequence"]
        rels = s.get("relations", [])
        if not rels:
            continue

        for i in range(len(words) - 1):
            a, b = words[i], words[i + 1]
            rel = rels[i]

            for pos, original in [(0, a), (1, b)]:
                cohort = cohort_map.get(original, [])
                if not cohort:
                    continue

                # Get top similar from cohort
                matches = bridge.closest(original, cohort, top_k=5, min_score=SIM_MIN)
                for sub, _ in matches:
                    if sub == original:
                        continue
                    new_seq = [sub, b] if pos == 0 else [a, sub]
                    key = (new_seq[0], new_seq[1], rel)
                    if key not in existing:
                        existing.add(key)
                        new_seqs.append({
                            "sequence": new_seq,
                            "relations": [rel],
                        })
                        if len(new_seqs) >= target:
                            return new_seqs

        if (idx + 1) % 500 == 0:
            print("    ... {}/{} sequences, {} new seqs".format(
                idx + 1, len(sequences), len(new_seqs)), end="\r")

    return new_seqs


def main():
    input_path = os.path.join(DATA_DIR, "dataset_v7_en.json")
    output_path = os.path.join(DATA_DIR, "dataset_v7_en_large.json")

    if not os.path.exists(input_path):
        print("  Run expand_dataset_en.py first.")
        return

    print("  Loading base...")
    base = load_base(input_path)
    print("  Base: {} sequences".format(len(base)))

    clusters, concept_to_parents = build_clusters(base)
    n_clusters = len(clusters)
    n_cohort = sum(1 for v in concept_to_parents.values() if v)
    print("  IS_A clusters: {}, concepts with parents: {}".format(n_clusters, n_cohort))

    print("  Loading embeddings...")
    bridge = EmbeddingBridge()

    print("  Safe augmentation...")
    new_seqs = safe_augment(base, clusters, concept_to_parents, bridge)

    all_seqs = base + new_seqs
    seen = set()
    deduped = []
    for s in all_seqs:
        key = (tuple(s["sequence"]), tuple(s.get("relations", [])))
        if key not in seen:
            seen.add(key)
            deduped.append(s)

    final_vocab = set()
    for s in deduped:
        final_vocab.update(s["sequence"])

    print("\n  Generated: {} new sequences".format(len(new_seqs)))
    print("  Total: {} sequences, {} words".format(len(deduped), len(final_vocab)))

    dataset = {
        "name": "conceptos_rutas_v7_en_large",
        "description": "v7 EN large (safe): {} seqs, {} words".format(
            len(deduped), len(final_vocab)),
        "sequences": deduped,
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)
    print("  Saved: {}".format(output_path))


if __name__ == "__main__":
    main()

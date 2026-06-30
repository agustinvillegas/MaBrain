"""
Generate reasoning paths (Capa 2) from a loaded Layer 1 graph.

Walk graph from seed nodes via BFS up to depth 3-6, following
coherent relation sequences. Output is a set of multi-hop paths
suitable for reward_thought() training.

Usage:
    python generate_reasoning_paths.py --layer1 data/processed/layer1_triplets.json \\
        --output data/processed/layer2_paths.json
"""
import json
import os
import sys
import time
import random
from collections import defaultdict, Counter


# ── Relation coherence rules ─────────────────────────────────────
# Which relations can follow which in a reasoning path
COHERENT_TRANSITIONS = {
    "causal_chains": {
        "desc": "A causes B which causes C which causes D",
        "relations": ["CAUSE", "EFFECT_OF"],
        "allow_recursive": True,
    },
    "functional_chains": {
        "desc": "A functions as B which is used for C",
        "relations": ["FUNCTION", "USED_FOR", "FUNCTION_OF"],
        "allow_recursive": True,
    },
    "hierarchical_chains": {
        "desc": "A IS_A B IS_A C (taxonomy)",
        "relations": ["IS_A", "HAS_INVERSE"],
        "allow_recursive": True,
    },
    "partonomic_chains": {
        "desc": "A PART_OF B CONTAINS C",
        "relations": ["PART_OF", "CONTAINS"],
        "allow_recursive": True,
    },
    "property_chains": {
        "desc": "A HAS_PROPERTY B IS_A property",
        "relations": ["HAS_PROPERTY", "PROPERTY_OF"],
        "allow_recursive": False,
    },
    "location_chains": {
        "desc": "A LOCATED_IN B CONTAINED_IN C",
        "relations": ["LOCATED_IN", "CONTAINED_IN"],
        "allow_recursive": True,
    },
}

# Allowed relations (union of all coherent chain types)
ALLOWED_RELATIONS = set()
for chain_type, config in COHERENT_TRANSITIONS.items():
    ALLOWED_RELATIONS.update(config["relations"])

# Single-step relations (can appear standalone but not chain)
SINGLE_STEP_RELATIONS = {"OPPOSITE", "SIMILAR_TO", "MADE_OF", "SYMBOL_OF"}

# Domain hints for automatic labeling
DOMAIN_HINTS = {
    "animal": "biology",
    "plant": "biology",
    "body": "biology",
    "human": "biology",
    "cell": "biology",
    "dna": "biology",
    "nature": "ecology",
    "weather": "ecology",
    "physics": "physics",
    "force": "physics",
    "gravity": "physics",
    "motion": "physics",
    "energy": "physics",
    "chemistry": "chemistry",
    "technology": "technology",
    "computer": "technology",
    "software": "technology",
    "city": "geography",
    "country": "geography",
    "river": "geography",
    "food": "food",
    "emotion": "psychology",
    "mind": "psychology",
    "social": "society",
    "society": "society",
    "economy": "society",
    "art": "culture",
    "music": "culture",
}


# ── Graph construction ────────────────────────────────────────────
def build_graph(triplets):
    """Build adjacency: concept -> [(relation, target_concept), ...]"""
    graph = defaultdict(list)
    concepts = set()
    for t in triplets:
        graph[t["from"]].append((t["relation"], t["to"]))
        concepts.add(t["from"])
        concepts.add(t["to"])
    return graph, concepts


# ── BFS path generation ──────────────────────────────────────────
def generate_paths(graph, seed_nodes, min_depth=2, max_depth=5,
                   paths_per_seed=20, max_total=100000):
    """BFS from each seed node, extract coherent paths."""
    random.seed(42)
    all_paths = []
    attempted = set()  # avoid revisiting same (start, node, relation_seq)

    # Shuffle seeds for diversity
    seeds = sorted(seed_nodes)
    random.shuffle(seeds)

    for seed in seeds:
        if len(all_paths) >= max_total:
            break

        # BFS queue: (current_node, path_words, path_relations)
        queue = [(seed, [seed], [])]
        visited_paths = set()

        while queue and len(all_paths) < max_total:
            current, words, rels = queue.pop(0)

            if len(words) >= min_depth and len(rels) >= 1:
                # Check coherence
                if rels:  # at least one relation
                    path_key = tuple(words)
                    if path_key not in visited_paths:
                        visited_paths.add(path_key)
                        # Label path type
                        path_type = classify_path(rels)

                        # Determine domain
                        domain = infer_domain(words)

                        all_paths.append({
                            "path": words,
                            "relations": rels,
                            "type": path_type,
                            "domain": domain,
                            "difficulty": max(1, len(words) - 1),
                            "length": len(words),
                        })

            if len(words) >= max_depth:
                continue

            for rel, target in graph.get(current, []):
                if target not in words:  # avoid cycles
                    # Check if this relation is coherent with previous
                    if rels and not is_coherent(rels[-1], rel):
                        continue
                    if len(rels) > 0 and rels[-1] == rel and rel == "IS_A":
                        # Allow IS_A chains but prevent infinite loops
                        pass
                    queue.append((target, words + [target], rels + [rel]))

    return all_paths[:max_total]


def is_coherent(prev_rel, next_rel):
    """Check if two relations can follow each other in a reasoning path."""
    # Same chain type = coherent
    for chain_type, config in COHERENT_TRANSITIONS.items():
        rels = config["relations"]
        if prev_rel in rels and next_rel in rels:
            return True
    return False


def classify_path(relations):
    """Classify a path based on its relation sequence."""
    # Count relation types
    counts = Counter(relations)
    dominant = counts.most_common(1)[0][0]

    for chain_type, config in COHERENT_TRANSITIONS.items():
        if dominant in config["relations"]:
            # Check if all relations belong to this chain
            if all(r in config["relations"] for r in relations):
                return chain_type

    return f"mixed_{dominant.lower()}"


def infer_domain(words):
    """Infer domain from words in the path."""
    domain_scores = defaultdict(int)
    for word in words:
        for key_word, domain in DOMAIN_HINTS.items():
            if key_word in word:
                domain_scores[domain] += 1
    if domain_scores:
        return max(domain_scores, key=domain_scores.get)
    return "general"


# ── Stats ─────────────────────────────────────────────────────────
def print_path_stats(paths):
    print(f"\n  --- Capa 2 Statistics ---")
    print(f"  Total paths: {len(paths):,}")

    type_counts = Counter(p["type"] for p in paths)
    print(f"  Path types:")
    for t, c in type_counts.most_common():
        print(f"    {t}: {c}")

    domain_counts = Counter(p["domain"] for p in paths)
    print(f"  Domains:")
    for d, c in domain_counts.most_common():
        print(f"    {d}: {c}")

    lengths = [p["length"] for p in paths]
    if lengths:
        print(f"  Length: min={min(lengths)}, max={max(lengths)}, avg={sum(lengths)/len(lengths):.1f}")

    unique_concepts = set()
    for p in paths:
        unique_concepts.update(p["path"])
    print(f"  Unique concepts in paths: {len(unique_concepts):,}")


# ── Main ──────────────────────────────────────────────────────────
def main():
    import argparse
    parser = argparse.ArgumentParser(description="Generate reasoning paths from Layer 1 graph (Capa 2)")
    parser.add_argument("--layer1", required=True,
                        help="Path to layer1_triplets.json")
    parser.add_argument("--output", "-o", default="data/processed/layer2_paths.json",
                        help="Output path for Capa 2 paths")
    parser.add_argument("--min-depth", type=int, default=2,
                        help="Minimum path depth (inclusive, default: 2)")
    parser.add_argument("--max-depth", type=int, default=5,
                        help="Maximum path depth (default: 5)")
    parser.add_argument("--paths-per-seed", type=int, default=20,
                        help="Max paths per seed node (default: 20)")
    parser.add_argument("--max-paths", type=int, default=100000,
                        help="Max total paths (default: 100K)")
    parser.add_argument("--seed-count", type=int, default=5000,
                        help="Number of seed nodes to sample (default: 5000)")
    args = parser.parse_args()

    print("=== Reasoning Path Generation (Capa 2) ===\n")

    # Load Layer 1
    print(f"Loading Layer 1 from: {args.layer1}")
    with open(args.layer1, "r", encoding="utf-8") as f:
        data = json.load(f)
    triplets = data.get("triplets", [])
    print(f"  Loaded {len(triplets):,} triplets")

    # Build graph
    print("Building adjacency graph...")
    graph, concepts = build_graph(triplets)
    print(f"  Concepts: {len(concepts):,}")
    print(f"  Avg degree: {sum(len(v) for v in graph.values())/max(len(graph),1):.1f}")

    # Select seed nodes (top by outgoing degree)
    seeds = sorted(graph.keys(), key=lambda c: -len(graph[c]))
    seed_sample = seeds[:min(args.seed_count, len(seeds))]
    print(f"  Seed nodes: {len(seed_sample):,} (top by degree)")

    # Generate paths
    print(f"Generating paths (depth {args.min_depth}-{args.max_depth})...")
    start_time = time.time()
    paths = generate_paths(
        graph, seed_sample,
        min_depth=args.min_depth,
        max_depth=args.max_depth,
        paths_per_seed=args.paths_per_seed,
        max_total=args.max_paths,
    )
    elapsed = time.time() - start_time
    print(f"  Generated {len(paths):,} paths in {elapsed/60:.1f} min")

    # Stats
    print_path_stats(paths)

    # Save
    dataset = {
        "version": "v1.0",
        "description": f"Capa 2: {len(paths):,} reasoning paths generated from Layer 1",
        "source_layer1": args.layer1,
        "generation_params": {
            "min_depth": args.min_depth,
            "max_depth": args.max_depth,
            "paths_per_seed": args.paths_per_seed,
            "seed_count": min(args.seed_count, len(seed_sample)),
        },
        "paths": paths,
    }
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)
    print(f"\nSaved: {args.output} ({os.path.getsize(args.output)/1024/1024:.1f} MB)")


if __name__ == "__main__":
    main()

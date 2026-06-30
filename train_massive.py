"""
Massive training loop for MaBrain (bio-inspired reasoning, not ML).

Following TRAINING_DESIGN.md:
  FASE 1: Load Layer 1 triplets → build graph structure
  FASE 2: Walk and reward Layer 2 paths → strengthen useful routes
  FASE 3: Register Layer 3 schemas → enable structural analogy
  FASE 4: Prune noise + consolidate

Usage:
    python train_massive.py \\
        --layer1 data/processed/layer1_triplets.json \\
        --layer2 data/processed/layer2_paths.json \\
        --layer3 data/processed/layer3_schemas.json \\
        --output data/brain_states/brain_massive_v1.json \\
        --epochs 3
"""
import json
import os
import sys
import time
import math
from collections import Counter

from braincell import Brain, Braincell, Synapse


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ── FASE 1: Load Layer 1 triplets ──────────────────────────────
def load_layer1(brain, triplets):
    """Load all triplets into the graph structure."""
    print("\n--- FASE 1: Loading Layer 1 triplets ---")
    start = time.time()
    n_created = 0
    n_skipped = 0

    for t in triplets:
        from_word = t.get("from", "").strip()
        to_word = t.get("to", "").strip()
        relation = t.get("relation", "").strip()

        if not from_word or not to_word or not relation:
            n_skipped += 1
            continue
        if from_word == to_word:
            n_skipped += 1
            continue

        cell_a = brain.get_or_create_cell(from_word)
        cell_b = brain.get_or_create_cell(to_word)
        brain.connect(cell_a, cell_b, to_word, relation=relation)
        n_created += 1

    elapsed = time.time() - start
    print(f"  Created {n_created} edges ({n_skipped} skipped)")
    print(f"  Cells: {len(brain.cells):,}, Synapses: {len(brain.synapses):,}")
    print(f"  Time: {elapsed:.1f}s")

    # Auto-infer missing relations
    print("  Running auto_infer_relations...")
    brain.auto_infer_relations()

    return brain


# ── FASE 2: Walk + reward Layer 2 paths ────────────────────────
def load_layer2(brain, paths, epochs=3):
    """Walk each path and apply reward_thought to strengthen it."""
    print(f"\n--- FASE 2: Training Layer 2 paths ({epochs} epochs) ---")
    start = time.time()
    total_rewards = 0

    for epoch in range(epochs):
        epoch_start = time.time()
        epoch_rewards = 0

        for path_data in paths:
            words = path_data.get("path", [])
            rels = path_data.get("relations", [])
            if len(words) < 2:
                continue

            current = brain.get_or_create_cell(words[0])
            if current.word is None:
                current.word = words[0]

            # Walk the path, ensuring each synapse exists and is activated
            syns_in_path = []
            valid_path = True

            for i in range(1, len(words)):
                target_word = words[i]
                rel = rels[i - 1] if i - 1 < len(rels) else None

                target = brain.get_or_create_cell(target_word)
                if target.word is None:
                    target.word = target_word

                syn = brain.connect(current, target, target_word, relation=rel)
                if syn is None:
                    valid_path = False
                    break

                # Activate trace as if it was selected
                syn.activation_trace += syn.strength
                syn.inference_usage += 1
                target.activation += syn.strength
                syns_in_path.append(syn)
                current = target

            if not valid_path or not syns_in_path:
                continue

            # Reward proportional to path length (longer paths = more reward)
            score = 1.0 / max(len(syns_in_path), 1)
            brain.reward_thought(score)
            epoch_rewards += 1

        elapsed_epoch = time.time() - epoch_start
        total_rewards += epoch_rewards
        print(f"  Epoch {epoch + 1}: {epoch_rewards} paths rewarded ({elapsed_epoch:.1f}s)")

    elapsed = time.time() - start
    print(f"  Total rewarded: {total_rewards} path-walks")
    print(f"  Synapses: {len(brain.synapses):,} (may include new ones from path creation)")
    print(f"  Time: {elapsed/60:.1f} min")

    return brain


# ── FASE 3: Register Layer 3 schemas ────────────────────────────
def load_layer3(brain, schemas):
    """Register schema instances in the graph.
    Most instances are already in the graph from Layer 1.
    This phase just verifies and optionally strengthens them."""
    print(f"\n--- FASE 3: Verifying Layer 3 schemas ---")
    start = time.time()
    verified = 0
    missing = 0

    for schema in schemas:
        for instance in schema.get("instances", []):
            # Schema instances have role-based keys like {"entity": "dog", "class": "mammal"}
            # We need to reconstruct word pairs from the relation sequence
            rels = schema.get("relation_sequence", [])
            words = list(instance.values())

            if len(words) < 2 or len(rels) < 1:
                continue

            for i in range(len(words) - 1):
                a, b = words[i], words[i + 1]
                rel = rels[i] if i < len(rels) else rels[-1]

                cell_a = brain.get_or_create_cell(a, fuzzy=True)
                cell_b = brain.get_or_create_cell(b, fuzzy=True)

                # Verify connection exists
                found = False
                for syn in cell_a.synapses_out:
                    if syn.target.id == cell_b.id and syn.relation == rel:
                        # Strengthen known schema edges
                        syn.strength *= 1.05
                        found = True
                        break

                if found:
                    verified += 1
                else:
                    # Try to create it (may not be in Layer 1)
                    if cell_a.word and cell_b.word:
                        brain.connect(cell_a, cell_b, b, relation=rel)
                        missing += 1

    elapsed = time.time() - start
    print(f"  Verified existing schema edges: {verified}")
    print(f"  Created missing schema edges: {missing}")
    print(f"  Total synapses: {len(brain.synapses):,}")
    print(f"  Time: {elapsed:.1f}s")

    return brain


# ── FASE 4: Prune + Consolidate ─────────────────────────────────
def consolidate(brain, min_usage=2):
    """Prune noise and normalize costs."""
    print(f"\n--- FASE 4: Pruning + Consolidation ---")
    start = time.time()

    # Prune synapses used less than min_usage times
    removed = brain.prune(min_usage=min_usage)
    print(f"  Pruned {removed} unused synapses (min_usage={min_usage})")

    # Normalize costs: compute average cost and adjust
    if brain.synapses:
        total_cost = sum(s.cost for s in brain.synapses.values())
        avg_cost = total_cost / max(len(brain.synapses), 1)
        print(f"  Avg cost (before normalization): {avg_cost:.4f}")

        if avg_cost > 0:
            for s in brain.synapses.values():
                s.cost = s.cost / avg_cost

        total_cost_after = sum(s.cost for s in brain.synapses.values())
        avg_cost_after = total_cost_after / max(len(brain.synapses), 1)
        print(f"  Avg cost (after normalization): {avg_cost_after:.4f}")

    # Report final stats
    rel_counts = Counter(s.relation for s in brain.synapses.values())
    print(f"\n  Final state:")
    print(f"  Cells: {len(brain.cells):,}")
    print(f"  Synapses: {len(brain.synapses):,}")
    print(f"  Relation distribution:")
    for rel, cnt in rel_counts.most_common(10):
        print(f"    {rel}: {cnt:,}")

    elapsed = time.time() - start
    print(f"  Time: {elapsed:.1f}s")

    return brain


# ── Main ──────────────────────────────────────────────────────────
def main():
    import argparse
    parser = argparse.ArgumentParser(description="MaBrain massive training loop")
    parser.add_argument("--layer1", required=True, help="Capa 1 triplets JSON")
    parser.add_argument("--layer2", required=True, help="Capa 2 reasoning paths JSON")
    parser.add_argument("--layer3", default=None, help="Capa 3 schemas JSON (optional)")
    parser.add_argument("--output", "-o", default="data/brain_states/brain_massive_v1.json",
                        help="Output brain state path")
    parser.add_argument("--epochs", type=int, default=3,
                        help="Training epochs over Capa 2 (default: 3)")
    parser.add_argument("--min-usage", type=int, default=2,
                        help="Min synapse usage to survive pruning (default: 2)")
    parser.add_argument("--context-bias", type=float, default=0.2,
                        help="Default context_bias_weight for trained brain (default: 0.2)")
    args = parser.parse_args()

    print("=" * 60)
    print("  MaBrain Massive Training (bio-inspired graph reasoning)")
    print("=" * 60)

    # Initialize brain
    brain = Brain(context_bias_weight=args.context_bias)
    total_start = time.time()

    # FASE 1: Layer 1
    triplets = load_json(args.layer1).get("triplets", [])
    print(f"  Loaded {len(triplets):,} triplets from {args.layer1}")
    load_layer1(brain, triplets)

    # FASE 2: Layer 2
    paths = load_json(args.layer2).get("paths", [])
    print(f"  Loaded {len(paths):,} paths from {args.layer2}")
    load_layer2(brain, paths, epochs=args.epochs)

    # FASE 3: Layer 3 (optional)
    if args.layer3 and os.path.exists(args.layer3):
        schemas = load_json(args.layer3).get("schemas", [])
        print(f"  Loaded {len(schemas):,} schemas from {args.layer3}")
        load_layer3(brain, schemas)
    else:
        print("\n  --- FASE 3: Skipped (no Layer 3 provided) ---")

    # FASE 4: Prune + Consolidate
    consolidate(brain, min_usage=args.min_usage)

    # Save
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    brain.save(args.output)
    print(f"\n  Saved brain state to: {args.output}")
    print(f"  File size: {os.path.getsize(args.output)/1024/1024:.1f} MB")

    elapsed = time.time() - total_start
    print(f"\n  Total training time: {elapsed/60:.1f} min")
    print("  Done.")


if __name__ == "__main__":
    main()

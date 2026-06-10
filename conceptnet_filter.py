"""Stream through 10 GB ConceptNet CSV, keep only EN pairs where both words in vocab."""
import json
import os
import sys

# ── Relation mapping CN → MaBrain ──────────────────────────────
CN_TO_MB = {
    "/r/IsA": "IS_A",
    "/r/UsedFor": "USED_FOR",
    "/r/Causes": "CAUSE",
    "/r/CausesDesire": "CAUSE",
    "/r/PartOf": "PART_OF",
    "/r/LocatedIn": "LOCATED_IN",
    "/r/AtLocation": "LOCATED_IN",
    "/r/CapableOf": "FUNCTION",
    "/r/HasProperty": "HAS_PROPERTY",
    "/r/HasA": "PART_OF",
    "/r/Antonym": "OPPOSITE",
    "/r/RelatedTo": None,  # skip generic
    "/r/Synonym": None,
    "/r/DerivedFrom": None,
    "/r/EtymologicallyRelatedTo": None,
    "/r/FormOf": None,
    "/r/NotDesires": None,
    "/r/Desires": None,
    "/r/ReceivesAction": None,
    "/r/MotivatedByGoal": None,
    "/r/ObstructedBy": None,
    "/r/CreatedBy": None,
    "/r/HasContext": None,
    "/r/HasFirstSubevent": None,
    "/r/HasLastSubevent": None,
    "/r/HasPrerequisite": None,
    "/r/HasSubevent": None,
    "/r/InstanceOf": None,
    "/r/MadeOf": None,
    "/r/NotCapableOf": None,
    "/r/NotHasProperty": None,
    "/r/PlacesFor": None,
    "/r/SymbolOf": None,
    "/r/UsedFor": "USED_FOR",
}

# Inverse mapping for creating reverse edges in dataset
INV_REL = {
    "IS_A": "IS_A",          # symmetric in reverse edge (subcategory → supercategory treated same)
    "USED_FOR": "FUNCTION",
    "CAUSE": "CAUSE",        # cause is asymmetric but we keep forward
    "PART_OF": "CONTAINS",
    "LOCATED_IN": "CONTAINED_IN",
    "FUNCTION": "USED_FOR",
    "HAS_PROPERTY": "PROPERTY_OF",
    "OPPOSITE": "OPPOSITE",
}


def load_vocab(path="current_vocab.json"):
    with open(path, "r", encoding="utf-8") as f:
        return set(json.load(f))


def parse_cn_line(line):
    """Parse a ConceptNet CSV tab-separated line.
    Returns (cn_rel, source_word, target_word, weight) or None."""
    parts = line.rstrip("\n").split("\t")
    if len(parts) < 5:
        return None
    cn_rel = parts[1]
    start = parts[2]
    end = parts[3]
    if not (start.startswith("/c/en/") and end.startswith("/c/en/")):
        return None
    s_word = start.split("/")[3]
    e_word = end.split("/")[3]
    try:
        meta = json.loads(parts[4])
        weight = meta.get("weight", 0.0)
    except json.JSONDecodeError:
        weight = 0.0
    return cn_rel, s_word, e_word, weight


def process_conceptnet(csv_path, vocab, output_path, min_weight=2.0):
    """Stream through CSV and collect valid pairs."""
    mb_relations = {r for r in CN_TO_MB.values() if r is not None}
    pairs = []
    line_count = 0
    en_count = 0
    kept_count = 0

    size_gb = os.path.getsize(csv_path) / (1024**3)
    print(f"  Scanning {csv_path} ({size_gb:.1f} GB)...")
    print(f"  Vocab size: {len(vocab)} words")
    print(f"  Min weight: {min_weight}")

    with open(csv_path, "r", encoding="utf-8") as f:
        for line in f:
            line_count += 1
            if line_count % 10000000 == 0:
                print(f"    Processed {line_count//1000000}M lines...")

            parsed = parse_cn_line(line)
            if parsed is None:
                continue
            cn_rel, s_word, e_word, weight = parsed
            en_count += 1

            if weight < min_weight:
                continue

            mb_rel = CN_TO_MB.get(cn_rel)
            if mb_rel is None:
                continue

            if s_word in vocab and e_word in vocab:
                pairs.append((s_word, e_word, mb_rel, weight))
                kept_count += 1
            elif s_word in vocab and e_word not in vocab:
                pass  # forward not possible
            elif s_word not in vocab and e_word in vocab:
                pass  # forward not possible

    print(f"  Total lines: {line_count:,} | EN: {en_count:,} | Kept: {kept_count:,}")
    return pairs


def save_pairs(pairs, output_path):
    """Save pairs as sequences with relations in dataset format."""
    sequences = []
    for s_word, e_word, rel, w in pairs:
        sequences.append({
            "sequence": [s_word, e_word],
            "relations": [rel],
            "weight": w,
        })

    desc = f"ConceptNet filtered pairs: {len(sequences)} seqs, min_weight=2.0"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({"description": desc, "sequences": sequences}, f, indent=1)
    print(f"  Saved {len(sequences)} pairs to {output_path}")


def main():
    csv_path = r"D:\ma_brain_data\conceptnet-assertions-5.7.0.csv"
    vocab_path = "current_vocab.json"
    output_path = r"D:\ma_brain_data\dataset_cn_filtered.json"

    vocab = load_vocab(vocab_path)
    pairs = process_conceptnet(csv_path, vocab, output_path, min_weight=2.0)
    save_pairs(pairs, output_path)

    # Show relation breakdown
    from collections import Counter
    rel_counts = Counter(rel for _, _, rel, _ in pairs)
    print(f"\n  Relation breakdown:")
    for rel, count in rel_counts.most_common():
        print(f"    {rel}: {count}")


if __name__ == "__main__":
    main()

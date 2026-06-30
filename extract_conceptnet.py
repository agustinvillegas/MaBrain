"""
Stream through ConceptNet 5.7 assertions CSV, extract English pairs,
map to MaBrain relations, output JSONL for canonicalization pipeline.

Usage:
    python extract_conceptnet.py data/raw/conceptnet-assertions-5.7.0.csv \\
        --min-weight 2.0 --output data/processed/cn_triplets.jsonl
"""
import json
import os
import sys
import time
from collections import Counter

# ── Relation mapping ConceptNet → MaBrain ────────────────────────
CN_TO_MB = {
    "/r/IsA":           "IS_A",
    "/r/UsedFor":       "USED_FOR",
    "/r/Causes":        "CAUSE",
    "/r/CausesDesire":  "CAUSE",
    "/r/PartOf":        "PART_OF",
    "/r/LocatedIn":     "LOCATED_IN",
    "/r/AtLocation":    "LOCATED_IN",
    "/r/CapableOf":     "FUNCTION",
    "/r/HasProperty":   "HAS_PROPERTY",
    "/r/HasA":          "PART_OF",
    "/r/Antonym":       "OPPOSITE",
    "/r/MadeOf":        "MADE_OF",
    "/r/SymbolOf":      "SYMBOL_OF",
    "/r/RelatedTo":     None,   # skip generic
    "/r/Synonym":       None,
    "/r/DerivedFrom":   None,
    "/r/FormOf":        None,
    "/r/Desires":       None,
    "/r/NotDesires":    None,
    "/r/ReceivesAction": None,
    "/r/MotivatedByGoal": None,
    "/r/ObstructedBy":  None,
    "/r/CreatedBy":     None,
    "/r/HasContext":    None,
    "/r/HasFirstSubevent": None,
    "/r/HasLastSubevent":  None,
    "/r/HasPrerequisite":  None,
    "/r/HasSubevent":   None,
    "/r/InstanceOf":    None,
    "/r/NotCapableOf":  None,
    "/r/NotHasProperty": None,
    "/r/EtymologicallyRelatedTo": None,
    "/r/EtymologicallyDerivedFrom": None,
}

# Relations that have clear inverse mapping
INV_REL = {
    "IS_A":         "HAS_INVERSE",
    "USED_FOR":     "FUNCTION",
    "CAUSE":        "EFFECT_OF",
    "PART_OF":      "CONTAINS",
    "LOCATED_IN":   "CONTAINED_IN",
    "FUNCTION":     "USED_FOR",
    "HAS_PROPERTY": "PROPERTY_OF",
    "OPPOSITE":     "OPPOSITE",
    "MADE_OF":      "MADE_OF",
    "SYMBOL_OF":    "SYMBOLIZED_BY",
}


def parse_cn_line(line):
    """Parse a ConceptNet CSV tab-separated line.
    Returns (cn_rel, source_word, target_word, weight, source_dataset) or None."""
    parts = line.rstrip("\n").split("\t")
    if len(parts) < 5:
        return None
    cn_rel = parts[1]
    start = parts[2]
    end = parts[3]
    # Only English
    if not (start.startswith("/c/en/") and end.startswith("/c/en/")):
        return None
    s_word = start.split("/")[3]
    e_word = end.split("/")[3]
    # Skip single-char words, punctuation
    if len(s_word) <= 1 or len(e_word) <= 1:
        return None
    # Skip words with special characters (not pure alpha)
    if not s_word.replace("_", "").isalpha() or not e_word.replace("_", "").isalpha():
        return None
    try:
        meta = json.loads(parts[4])
        weight = float(meta.get("weight", 0.0))
        source_dataset = meta.get("dataset", "/d/unknown")
    except (json.JSONDecodeError, ValueError):
        weight = 0.0
        source_dataset = "/d/unknown"
    return {
        "cn_rel": cn_rel,
        "from": s_word.lower().replace(" ", "_"),
        "to": e_word.lower().replace(" ", "_"),
        "weight": weight,
        "source": source_dataset,
    }


def process_stream(csv_path, min_weight=2.0, output_path=None, report_every=1000000):
    """Stream through CSV and write JSONL with English MaBrain relations."""
    mb_relations = {r for r in CN_TO_MB.values() if r is not None}

    line_count = 0
    en_count = 0
    kept_count = 0
    rel_counter = Counter()
    start_time = time.time()
    last_report = start_time

    # Determine output
    f_out = open(output_path, "w", encoding="utf-8") if output_path else sys.stdout

    csv_size = os.path.getsize(csv_path) if os.path.exists(csv_path) else 0
    print(f"  File: {csv_path} ({csv_size/1024**3:.1f} GB)")
    print(f"  Min weight: {min_weight}")
    print(f"  Output: {output_path or 'stdout'}")
    print()

    with open(csv_path, "r", encoding="utf-8") as f:
        for line in f:
            line_count += 1
            if line_count % report_every == 0:
                elapsed = time.time() - start_time
                rate = line_count / elapsed
                print(f"  [{elapsed/60:.1f}min] {line_count//1000000}M lines | "
                      f"EN: {en_count//1000}K | kept: {kept_count//1000}K | "
                      f"{rate/1000000:.1f}M lines/min", flush=True)

            parsed = parse_cn_line(line)
            if parsed is None:
                continue
            en_count += 1

            if parsed["weight"] < min_weight:
                continue

            mb_rel = CN_TO_MB.get(parsed["cn_rel"])
            if mb_rel is None:
                continue

            # Write JSONL
            record = {
                "from": parsed["from"],
                "to": parsed["to"],
                "relation": mb_rel,
                "weight": parsed["weight"],
                "source": parsed["source"],
            }
            f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
            kept_count += 1
            rel_counter[mb_rel] += 1

    elapsed_total = time.time() - start_time
    print()
    print(f"  === SUMMARY ===")
    print(f"  Total lines:   {line_count:,}")
    print(f"  EN lines:      {en_count:,}")
    print(f"  Kept (≥{min_weight}):  {kept_count:,}  ({(kept_count/en_count*100) if en_count else 0:.1f}% of EN)")
    print(f"  Time:          {elapsed_total/60:.1f} min")
    print(f"  Rate:          {line_count/elapsed_total/1000000:.1f}M lines/min")
    print()
    print("  Relations:")
    for rel, cnt in rel_counter.most_common():
        print(f"    {rel}: {cnt}")

    f_out.close()
    return kept_count


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Extract English MaBrain triplets from ConceptNet 5.7 CSV")
    parser.add_argument("csv_path", help="Path to conceptnet-assertions-5.7.0.csv")
    parser.add_argument("--min-weight", type=float, default=2.0,
                        help="Minimum edge weight (default: 2.0)")
    parser.add_argument("--output", "-o", default="data/processed/cn_triplets.jsonl",
                        help="Output JSONL path (default: data/processed/cn_triplets.jsonl)")
    parser.add_argument("--report-every", type=int, default=1000000,
                        help="Lines between progress reports (default: 1M)")
    args = parser.parse_args()

    if not os.path.exists(args.csv_path):
        print(f"ERROR: File not found: {args.csv_path}")
        sys.exit(1)

    process_stream(
        args.csv_path,
        min_weight=args.min_weight,
        output_path=args.output,
        report_every=args.report_every,
    )


if __name__ == "__main__":
    main()

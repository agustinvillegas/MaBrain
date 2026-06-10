"""Merge multiple dataset JSON files into one combined dataset."""
import json
import sys


def load_seqs(path):
    """Load sequences from a dataset file (handles both old dict format and list format)."""
    d = json.load(open(path, "r", encoding="utf-8"))
    if isinstance(d, dict):
        return d.get("sequences", [])
    return d


def merge_datasets(paths, output_path, description="Merged dataset"):
    """Merge multiple datasets, deduplicating sequences."""
    all_seqs = []
    seen = set()
    for path in paths:
        seqs = load_seqs(path)
        for s in seqs:
            seq_tuple = tuple(s["sequence"])
            rel_tuple = tuple(s.get("relations", [None] * (len(s["sequence"]) - 1)))
            key = (seq_tuple, rel_tuple)
            if key not in seen:
                seen.add(key)
                all_seqs.append(s)
        print(f"  {path}: {len(seqs)} seqs -> {len(all_seqs)} unique so far")

    # Count unique words
    words = set()
    for s in all_seqs:
        for w in s["sequence"]:
            words.add(w)
    
    desc = f"{description}: {len(all_seqs)} seqs, {len(words)} words"
    result = {"description": desc, "sequences": all_seqs}
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=1)
    print(f"\n  Saved: {len(all_seqs)} sequences, {len(words)} words -> {output_path}")
    return result


if __name__ == "__main__":
    paths = sys.argv[1:-2] if len(sys.argv) > 3 else []
    if not paths:
        print("Usage: python merge_datasets.py <input1.json> <input2.json> ... <output.json> <description>")
        sys.exit(1)
    output_path = sys.argv[-2]
    description = sys.argv[-1]
    merge_datasets(paths, output_path, description)

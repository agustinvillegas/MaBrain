"""
Extract MaBrain triplets from Wikidata via SPARQL.

Relations extracted:
  - IS_A:      wdt:P31 (instance of) + wdt:P279 (subclass of)
  - PART_OF:   wdt:P361 (part of)
  - LOCATED_IN: wdt:P131 (located in administrative entity)

Usage:
    python extract_wikidata.py --output data/processed/wd_triplets.jsonl
    python extract_wikidata.py --resume  # continue from cache
"""
import json
import os
import sys
import time
import urllib.request
import urllib.error
import urllib.parse

SPARQL_ENDPOINT = "https://query.wikidata.org/sparql"
USER_AGENT = "MaBrain/1.0 (extraction pipeline)"
CACHE_DIR = "data/raw/wikidata_cache"

# Batch size per SPARQL query (CONSTRUCT with LIMIT)
BATCH_SIZE = 50000
DELAY_BETWEEN_BATCHES = 1.0  # seconds to respect rate limits

# ── SPARQL queries ──────────────────────────────────────────────
# We query English labels and edges for common concepts (~40k most used)
# Each query gets items with English labels, limited to BATCH_SIZE

SPARQL_TEMPLATES = {
    "IS_A": """
SELECT DISTINCT ?item ?itemLabel ?class ?classLabel WHERE {
  VALUES ?rel { wdt:P31 wdt:P279 }
  ?item ?rel ?class .
  ?item rdfs:label ?itemLabel .
  ?class rdfs:label ?classLabel .
  FILTER(LANG(?itemLabel) = "en" && LANG(?classLabel) = "en")
  FILTER(?item != ?class)
}
LIMIT {limit}
OFFSET {offset}
""",
    "PART_OF": """
SELECT DISTINCT ?item ?itemLabel ?whole ?wholeLabel WHERE {
  ?item wdt:P361 ?whole .
  ?item rdfs:label ?itemLabel .
  ?whole rdfs:label ?wholeLabel .
  FILTER(LANG(?itemLabel) = "en" && LANG(?wholeLabel) = "en")
  FILTER(?item != ?whole)
}
LIMIT {limit}
OFFSET {offset}
""",
    "LOCATED_IN": """
SELECT DISTINCT ?item ?itemLabel ?location ?locationLabel WHERE {
  ?item wdt:P131 ?location .
  ?item rdfs:label ?itemLabel .
  ?location rdfs:label ?locationLabel .
  FILTER(LANG(?itemLabel) = "en" && LANG(?locationLabel) = "en")
  FILTER(?item != ?location)
}
LIMIT {limit}
OFFSET {offset}
""",
}


def clean_label(label):
    """Normalize label: lowercase, replace spaces with underscores."""
    return label.lower().strip().replace(" ", "_").replace("-", "_")


def query_sparql(query, timeout=60):
    """Execute SPARQL query via Wikidata public endpoint, return JSON results."""
    url = SPARQL_ENDPOINT + "?format=json&query=" + urllib.parse.quote(query)
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return data.get("results", {}).get("bindings", [])
    except urllib.error.HTTPError as e:
        print(f"    HTTP Error {e.code}: {e.reason}")
        if e.code == 429:
            print("    Rate limited. Sleeping 30s...")
            time.sleep(30)
            return None
        return None
    except Exception as e:
        print(f"    Error: {e}")
        return None


def extract_relation(rel_name, output_path, cache=True):
    """Extract all triplets for a given relation, writing to JSONL."""
    template = SPARQL_TEMPLATES[rel_name]
    offset = 0
    total = 0
    batch_num = 0
    empty_batches = 0
    f_out = open(output_path, "a", encoding="utf-8")

    while empty_batches < 3:
        query = template.format(limit=BATCH_SIZE, offset=offset)
        results = query_sparql(query)
        if results is None:
            time.sleep(5)
            continue

        if len(results) == 0:
            empty_batches += 1
            offset += BATCH_SIZE
            continue
        empty_batches = 0

        for bind in results:
            item_label = clean_label(bind.get("itemLabel", {}).get("value", ""))
            if rel_name == "IS_A":
                class_label = clean_label(bind.get("classLabel", {}).get("value", ""))
                record = {
                    "from": item_label,
                    "relation": "IS_A",
                    "to": class_label,
                    "source": "wikidata",
                }
            elif rel_name == "PART_OF":
                whole_label = clean_label(bind.get("wholeLabel", {}).get("value", ""))
                record = {
                    "from": item_label,
                    "relation": "PART_OF",
                    "to": whole_label,
                    "source": "wikidata",
                }
            elif rel_name == "LOCATED_IN":
                loc_label = clean_label(bind.get("locationLabel", {}).get("value", ""))
                record = {
                    "from": item_label,
                    "relation": "LOCATED_IN",
                    "to": loc_label,
                    "source": "wikidata",
                }
            # Sanity: both terms must be at least 2 chars
            if len(record["from"]) >= 2 and len(record["to"]) >= 2:
                f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
                total += 1

        batch_num += 1
        offset += BATCH_SIZE
        print(f"    [{rel_name}] batch {batch_num}: {total:,} triplets (offset {offset:,})", flush=True)

        if batch_num > 0 and total == 0:
            break

        # Rate limiting
        time.sleep(DELAY_BETWEEN_BATCHES)

    f_out.close()
    return total


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Extract MaBrain triplets from Wikidata SPARQL")
    parser.add_argument("--output-dir", default="data/processed",
                        help="Output directory (default: data/processed)")
    parser.add_argument("--relations", nargs="+", default=["IS_A", "PART_OF", "LOCATED_IN"],
                        help="Relations to extract (default: all)")
    parser.add_argument("--no-cache", action="store_true",
                        help="Overwrite existing files instead of appending")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)

    for rel in args.relations:
        if rel not in SPARQL_TEMPLATES:
            print(f"  Unknown relation: {rel}. Skipping.")
            continue

        output_path = os.path.join(args.output_dir, f"wd_{rel.lower()}.jsonl")
        mode = "w" if args.no_cache else "a"
        # Clear file if overwriting
        if args.no_cache:
            with open(output_path, "w", encoding="utf-8") as f:
                pass

        print(f"\nExtracting {rel} -> {output_path}")
        total = extract_relation(rel, output_path)
        print(f"  Done: {total:,} {rel} triplets")

    # Merge all wd_*.jsonl into one
    print("\n--- Merging Wikidata sources ---")
    all_path = os.path.join(args.output_dir, "wd_triplets.jsonl")
    with open(all_path, "w", encoding="utf-8") as f_out:
        total_all = 0
        for rel in args.relations:
            src_path = os.path.join(args.output_dir, f"wd_{rel.lower()}.jsonl")
            if os.path.exists(src_path):
                with open(src_path, "r", encoding="utf-8") as f_in:
                    for line in f_in:
                        f_out.write(line)
                        total_all += 1
                print(f"    Merged wd_{rel.lower()}.jsonl")
        print(f"  Total: {total_all:,} Wikidata triplets -> {all_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()

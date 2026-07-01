"""
Filter schemas for human validation. Produces a compact JSONL file
where each line is a schema candidate ready for ✓/✗ validation.

Output format:
    {"schema_id": "...", "relation_sequence": ["...", "..."],
     "roles": ["...", "..."], "domain": "...", "instances": [...],
     "valid": null}  # ← rellenas con true/false

Usage:
    python filter_schemas_for_validation.py \\
        --layer3 data/processed/layer3_schemas.json \\
        --output data/validation/schemas_to_validate.jsonl \\
        --max-schemas 100
"""
import json
import os
import sys
import random
from collections import Counter, defaultdict


# ── Domain balance ─────────────────────────────────────────────
TARGET_DOMAINS = [
    "biology", "physics", "society", "technology",
    "psychology", "geography", "culture", "general",
]


def load_schemas(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("schemas", [])


def filter_and_sample(schemas, max_schemas=100, min_instances=3, max_instances_show=5):
    """Sample diverse schemas for validation, with domain balance."""

    # Remove schemas with too few instances
    schemas = [s for s in schemas if s.get("num_instances", 0) >= min_instances]
    print(f"  After min_instances filter ({min_instances}): {len(schemas)} schemas")

    # Remove very long sequences (hard to validate)
    schemas = [s for s in schemas if len(s.get("relation_sequence", [])) <= 4]
    print(f"  After max_length filter (<=4): {len(schemas)} schemas")

    # Remove schemas with no instances
    schemas = [s for s in schemas if s.get("instances")]
    print(f"  After removing empty: {len(schemas)} schemas")

    # Group by domain
    by_domain = defaultdict(list)
    for s in schemas:
        domain = s.get("domain", "general")
        by_domain[domain].append(s)

    # Sort by instance count within each domain
    for domain in by_domain:
        by_domain[domain].sort(key=lambda s: -s["num_instances"])

    # Sample: give each domain a proportional share
    n_domains = len([d for d in TARGET_DOMAINS if by_domain.get(d)])
    if n_domains == 0:
        n_domains = 1
    per_domain = max(1, max_schemas // n_domains)

    selected = []
    seen_by_rel = Counter()

    # Interleave domains to get diversity
    domain_queues = {}
    for d in TARGET_DOMAINS:
        if by_domain.get(d):
            domain_queues[d] = iter(by_domain[d])

    while len(selected) < max_schemas:
        added_this_pass = 0
        for d in TARGET_DOMAINS:
            if d not in domain_queues:
                continue
            it = domain_queues[d]
            try:
                schema = next(it)
                rel_key = tuple(schema.get("relation_sequence", []))
                # Avoid duplicate relation sequences
                if seen_by_rel[rel_key] >= 3:
                    continue
                selected.append(schema)
                seen_by_rel[rel_key] += 1
                added_this_pass += 1
            except StopIteration:
                del domain_queues[d]
        if added_this_pass == 0:
            break

    # Shuffle for unbiased validation
    random.shuffle(selected)

    return selected[:max_schemas]


def prepare_validation(schemas):
    """Convert to validation-friendly format."""
    validation_items = []
    for s in schemas:
        # Limit shown instances to keep each item compact
        instances_shown = s.get("instances", [])[:5]
        # Format instances as readable tuples
        instance_tuples = []
        for inst in instances_shown:
            roles = s.get("roles", [])
            if roles and len(roles) == len(inst):
                instance_tuples.append(inst)
            else:
                # Fallback: show raw word list
                instance_tuples.append(inst)

        item = {
            "schema_id": s["schema_id"],
            "description": s.get("description", ""),
            "relation_sequence": s.get("relation_sequence", []),
            "roles": s.get("roles", []),
            "domain": s.get("domain", "general"),
            "difficulty": s.get("difficulty", 1),
            "instances": instance_tuples,
            "num_instances_total": s.get("num_instances", 0),
            "valid": None,  # ← validación humana: true/false
            "notes": "",    # ← opcional: por qué inválido
        }
        validation_items.append(item)

    return validation_items


def save_validation(items, output_path):
    """Save as JSONL, one schema per line."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"  Saved {len(items)} validation items to: {output_path}")


def print_validation_preview(items):
    """Print a human-readable summary of what to validate."""
    domain_counts = Counter(i["domain"] for i in items)
    rel_counts = Counter(tuple(i["relation_sequence"]) for i in items)

    print(f"\n  === Validation Preview ===")
    print(f"  Total schemas to validate: {len(items)}")
    print(f"  Domains: {dict(domain_counts.most_common())}")
    print(f"  Unique relation patterns: {len(rel_counts)}")
    print()
    print(f"  Examples:")
    for i, item in enumerate(items[:5]):
        print(f"\n  --- [{i+1}] {item['schema_id']} ---")
        print(f"  Description: {item['description']}")
        print(f"  Relations: {' -> '.join(item['relation_sequence'])}")
        print(f"  Roles: {item['roles']}")
        print(f"  Domain: {item['domain']}")
        print(f"  Total instances: {item['num_instances_total']}")

        for j, inst in enumerate(item["instances"][:3]):
            roles = item["roles"]
            if len(roles) == len(inst):
                pair_str = ", ".join(f"{r}={w}" for r, w in zip(roles, inst))
            else:
                pair_str = str(inst)
            print(f"    {j+1}. {pair_str}")
        if item["num_instances_total"] > 3:
            print(f"    ... +{item['num_instances_total'] - 3} more")
        print(f"  Valid? [ ] yes  [ ] no  Notes: ______________")


# ── Main ──────────────────────────────────────────────────────────
def main():
    import argparse
    parser = argparse.ArgumentParser(description="Filter Capa 3 schemas for human validation")
    parser.add_argument("--layer3", required=True,
                        help="Path to layer3_schemas.json")
    parser.add_argument("--output", "-o", default="data/gold/schemas_to_validate.jsonl",
                        help="Output JSONL path")
    parser.add_argument("--max-schemas", type=int, default=100,
                        help="Maximum schemas to validate (default: 100)")
    parser.add_argument("--min-instances", type=int, default=3,
                        help="Minimum instances per schema (default: 3)")
    parser.add_argument("--preview", action="store_true",
                        help="Print preview only, don't save")
    args = parser.parse_args()

    print("=== Filter Schemas for Human Validation ===\n")

    # Load
    schemas = load_schemas(args.layer3)
    print(f"  Loaded {len(schemas)} schemas from {args.layer3}")

    # Filter + sample
    selected = filter_and_sample(schemas, max_schemas=args.max_schemas,
                                 min_instances=args.min_instances)
    print(f"  Selected {len(selected)} schemas for validation")

    # Prepare
    items = prepare_validation(selected)

    # Preview or save
    print_validation_preview(items)

    if not args.preview:
        save_validation(items, args.output)
        print(f"\n  To validate, edit the JSONL and set \"valid\": true/false")
        print(f"  You can also add \"notes\": \"\" for invalid ones.")

    print("\nDone.")


if __name__ == "__main__":
    main()

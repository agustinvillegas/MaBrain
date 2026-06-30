"""
Mine relational schemas (Capa 3) from reasoning paths.

Groups paths with identical relation sequences into schemas,
then extracts instance tuples for each schema.

Usage:
    python mine_schemas.py --layer2 data/processed/layer2_paths.json \\
        --output data/processed/layer3_schemas.json
"""
import json
import os
import sys
import time
from collections import defaultdict, Counter


def load_paths(path):
    """Load Capa 2 paths."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("paths", [])


def mine_schemas(paths, min_support=3):
    """Group paths by relation sequence → extract schemas with instances.

    A schema is defined by a unique sequence of relations.
    Instances are the concept tuples that follow that sequence.
    """
    # Group by relation sequence
    rel_seq_to_paths = defaultdict(list)
    for p in paths:
        rels = tuple(p.get("relations", []))
        if len(rels) >= 1:
            rel_seq_to_paths[rels].append(p)

    schemas = []
    schema_id_counter = 0

    for rel_seq, group_paths in sorted(
        rel_seq_to_paths.items(),
        key=lambda x: -len(x[1]),
    ):
        if len(group_paths) < min_support:
            continue

        schema_id_counter += 1
        rel_list = list(rel_seq)

        # Determine schema type from dominant domain
        domains = Counter(p.get("domain", "general") for p in group_paths)
        primary_domain = domains.most_common(1)[0][0]

        # Build role labels based on relation sequence
        roles = infer_role_labels(rel_list)

        # Build instances
        instances = []
        instance_words_seen = set()
        for p in group_paths:
            words = p.get("path", [])
            if len(words) != len(rel_list) + 1:
                # Path length should be n_relations + 1
                continue

            # Build instance dict keyed by role
            instance = {}
            for i, (role, word) in enumerate(zip(roles, words)):
                instance[role] = word

            # Dedup instances by the word tuple
            inst_key = tuple(words)
            if inst_key not in instance_words_seen:
                instance_words_seen.add(inst_key)
                instances.append(instance)

        # Generate schema description
        description = generate_description(rel_list)

        schemas.append({
            "schema_id": f"SCHEMA_{schema_id_counter:04d}",
            "description": description,
            "relation_sequence": rel_list,
            "roles": roles,
            "domain": primary_domain,
            "difficulty": max(1, len(rel_list) - 1),
            "instances": instances,
            "num_instances": len(instances),
            "num_raw_paths": len(group_paths),
        })

    return schemas


# ── Role / Description inference ─────────────────────────────────
ROLE_TEMPLATES = {
    "IS_A":        ["entity", "class"],
    "HAS_INVERSE": ["class", "entity"],
    "FUNCTION":    ["agent", "action"],
    "FUNCTION_OF": ["action", "agent"],
    "USED_FOR":    ["tool", "action"],
    "CAUSE":       ["cause", "effect"],
    "EFFECT_OF":   ["effect", "cause"],
    "PART_OF":     ["part", "whole"],
    "CONTAINS":    ["whole", "part"],
    "HAS_PROPERTY": ["entity", "property"],
    "PROPERTY_OF":  ["property", "entity"],
    "LOCATED_IN":   ["entity", "location"],
    "CONTAINED_IN": ["location", "entity"],
    "OPPOSITE":     ["concept_a", "concept_b"],
    "SIMILAR_TO":   ["concept_a", "concept_b"],
    "MADE_OF":      ["entity", "material"],
    "SYMBOL_OF":    ["symbol", "meaning"],
}

DESCRIPTION_TEMPLATES = {
    ("IS_A",): "Una entidad pertenece a una clase",
    ("IS_A", "IS_A"): "Una entidad pertenece a una clase que a su vez es subclase de otra",
    ("IS_A", "IS_A", "IS_A"): "Jerarquía taxonómica de 3 niveles (entity → class → superclass → supersuperclass)",
    ("FUNCTION",): "Un agente realiza una acción",
    ("FUNCTION", "IS_A"): "Un agente realiza una acción que es un tipo de actividad",
    ("CAUSE",): "Una causa produce un efecto",
    ("CAUSE", "CAUSE"): "Cadena causal de 2 pasos",
    ("CAUSE", "CAUSE", "CAUSE"): "Cadena causal de 3 pasos",
    ("PART_OF",): "Una parte pertenece a un todo",
    ("PART_OF", "PART_OF"): "Anidación de partes: parte de parte de todo",
    ("HAS_PROPERTY",): "Una entidad tiene una propiedad",
    ("LOCATED_IN",): "Una entidad está localizada en un lugar",
    ("OPPOSITE",): "Dos conceptos son opuestos",
    ("SIMILAR_TO",): "Dos conceptos son similares",
    ("USED_FOR",): "Una herramienta se usa para una acción",
    ("IS_A", "FUNCTION"): "Una entidad de cierto tipo realiza una función específica",
    ("FUNCTION", "CAUSE"): "Una acción causa un efecto",
}


def infer_role_labels(rel_list):
    """Generate role labels for each position in the relation sequence."""
    roles = []

    # For first position, use the template for first relation
    first_rel = rel_list[0]
    template = ROLE_TEMPLATES.get(first_rel, ["concept_x", "concept_y"])
    roles.append(template[0])  # role for first word

    # For each relation step, add the target role
    for i, rel in enumerate(rel_list):
        template = ROLE_TEMPLATES.get(rel, ["source", "target"])
        # First step already has its source role, add the target role
        if i == len(roles) - 1:
            roles.append(template[1])
        else:
            # For middle positions in chains, the target of previous = source of current
            pass  # Already added by previous iteration

    # Ensure all positions have roles
    while len(roles) < len(rel_list) + 1:
        roles.append(f"concept_{len(roles)}")

    return roles


def generate_description(rel_list):
    """Generate a human-readable description of this schema."""
    key = tuple(rel_list)
    if key in DESCRIPTION_TEMPLATES:
        return DESCRIPTION_TEMPLATES[key]

    # Auto-generate
    if len(rel_list) == 1:
        return f"Relación simple: {rel_list[0]}"
    rel_types = Counter(rel_list)
    if len(rel_types) == 1:
        rel = list(rel_types.keys())[0]
        return f"Cadena de {len(rel_list)} pasos de {rel}"
    return f"Secuencia mixta: {' → '.join(rel_list)}"


# ── Stats ─────────────────────────────────────────────────────────
def print_schema_stats(schemas):
    print(f"\n  --- Capa 3 Statistics ---")
    print(f"  Total schemas: {len(schemas):,}")
    print(f"  Instances total: {sum(s['num_instances'] for s in schemas):,}")

    domain_counts = Counter(s["domain"] for s in schemas)
    print(f"  Domains:")
    for d, c in domain_counts.most_common():
        print(f"    {d}: {c}")

    length_dist = Counter(len(s["relation_sequence"]) for s in schemas)
    print(f"  Length distribution:")
    for l, c in sorted(length_dist.items()):
        print(f"    {l} relations: {c} schemas")

    # Top schemas by instance count
    print(f"\n  Top 10 schemas:")
    for s in schemas[:10]:
        print(f"    {s['schema_id']}: {s['description']} ({s['num_instances']} instances)")


# ── Export ────────────────────────────────────────────────────────
def export_schemas(schemas, output_path, top_k=None):
    """Export schemas in Capa 3 format."""
    if top_k is not None and top_k < len(schemas):
        exported = schemas[:top_k]
    else:
        exported = schemas

    # Remove raw path data from instances for cleaner output
    for s in exported:
        s.pop("num_raw_paths", None)

    dataset = {
        "version": "v1.0",
        "description": f"Capa 3: {len(exported):,} schemas, {sum(s['num_instances'] for s in exported):,} instances",
        "schemas": exported,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)
    print(f"\nSaved: {output_path} ({os.path.getsize(output_path)/1024/1024:.1f} MB)")


# ── Main ──────────────────────────────────────────────────────────
def main():
    import argparse
    parser = argparse.ArgumentParser(description="Mine relational schemas from Capa 2 paths (Capa 3)")
    parser.add_argument("--layer2", required=True,
                        help="Path to layer2_paths.json")
    parser.add_argument("--output", "-o", default="data/processed/layer3_schemas.json",
                        help="Output path for Capa 3 schemas")
    parser.add_argument("--min-support", type=int, default=3,
                        help="Minimum paths to form a schema (default: 3)")
    parser.add_argument("--top-k", type=int, default=500,
                        help="Export only top K schemas by instance count (default: 500)")
    args = parser.parse_args()

    print("=== Schema Mining (Capa 3) ===\n")

    # Load Capa 2
    print(f"Loading Capa 2 from: {args.layer2}")
    start_time = time.time()
    paths = load_paths(args.layer2)
    print(f"  Loaded {len(paths):,} paths")

    # Mine schemas
    print(f"Mining schemas (min support: {args.min_support})...")
    schemas = mine_schemas(paths, min_support=args.min_support)
    print(f"  Found {len(schemas):,} unique schemas")

    # Sort by instance count descending
    schemas.sort(key=lambda s: -s["num_instances"])

    # Stats
    print_schema_stats(schemas)

    # Export
    export_schemas(schemas, args.output, top_k=args.top_k)

    elapsed = time.time() - start_time
    print(f"\nTotal time: {elapsed/60:.1f} min")


if __name__ == "__main__":
    main()

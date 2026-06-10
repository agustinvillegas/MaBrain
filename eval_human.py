"""
Human evaluation protocol for MaBrain Dialogue Manager.
Usage:
    python eval_human.py --run       # Generate CSV with both variants
    python eval_human.py --score     # Open CSV for annotation
"""

import argparse
import copy
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import dialogue_manager
from braincell import Brain

BRAIN_PATH = r"D:\ma_brain_data\brain_state_v8_v8_pruned_evaled.json"
OUTPUT_CSV = "eval_human_results.csv"

# 8 multi-turn dialogues covering all 8 intents + greeting (compact for speed)
DIALOGUES = [
    {  # 1: dog — definition → property → function → followup
        "id": 1,
        "turns": [
            ("definition", "What is a dog?"),
            ("property", "What is a dog like?"),
            ("function", "What does a dog do?"),
            ("followup", "Tell me more."),
        ],
    },
    {  # 2: cat — definition → function → property → analogy
        "id": 2,
        "turns": [
            ("definition", "What is a cat?"),
            ("function", "What does a cat do?"),
            ("property", "What is a cat like?"),
            ("analogy", "How is a cat similar to a dog?"),
        ],
    },
    {  # 3: fire — definition → cause → property → followup
        "id": 3,
        "turns": [
            ("definition", "What is fire?"),
            ("cause", "What causes fire?"),
            ("property", "What is fire like?"),
            ("followup", "What else?"),
        ],
    },
    {  # 4: water — property → location → cause → analogy
        "id": 4,
        "turns": [
            ("property", "What is water like?"),
            ("location", "Where is water?"),
            ("cause", "What does water cause?"),
            ("analogy", "How is water similar to fire?"),
        ],
    },
    {  # 5: rain → greeting → definition → followup → function
        "id": 5,
        "turns": [
            ("greeting", "Hello"),
            ("definition", "What is rain?"),
            ("followup", "Tell me more."),
            ("function", "What does rain do?"),
        ],
    },
    {  # 6: fish — location → definition → property → followup
        "id": 6,
        "turns": [
            ("location", "Where is a fish?"),
            ("definition", "What is a fish?"),
            ("property", "What is a fish like?"),
            ("followup", "Continue."),
        ],
    },
    {  # 7: tree — definition → property → function → followup
        "id": 7,
        "turns": [
            ("definition", "What is a tree?"),
            ("property", "What is a tree like?"),
            ("function", "What does a tree do?"),
            ("followup", "Tell me more."),
        ],
    },
    {  # 8: bird — definition → analogy → function → followup
        "id": 8,
        "turns": [
            ("definition", "What is a bird?"),
            ("analogy", "How is a bird similar to a fish?"),
            ("function", "What does a bird do?"),
            ("followup", "What else?"),
        ],
    },
]


def set_all_ew(value):
    """Set embedding_weight on all strategies."""
    for v in dialogue_manager.STRATEGY_CONFIG.values():
        v["ew"] = value


def run_eval():
    """Run all dialogues on both variants and produce CSV."""
    # Backup module-level config
    original_config = {
        k: dict(v) for k, v in dialogue_manager.STRATEGY_CONFIG.items()
    }

    # Reduce steps for slow strategies during eval
    dialogue_manager.STRATEGY_CONFIG["CONTINUE_TOPIC"]["steps"] = 4
    dialogue_manager.STRATEGY_CONFIG["EXPLORE"]["steps"] = 4

    # Load brain once (reuse for both variants)
    brain = Brain()
    brain.load(BRAIN_PATH)

    from working_memory import WorkingMemory

    rows = []

    for variant in ("A", "B"):
        if variant == "A":
            set_all_ew(0.0)
            variant_label = "ew=0.0"
        else:
            # Restore original config
            dialogue_manager.STRATEGY_CONFIG.clear()
            dialogue_manager.STRATEGY_CONFIG.update(
                {k: dict(v) for k, v in original_config.items()}
            )
            dialogue_manager.STRATEGY_CONFIG["CONTINUE_TOPIC"]["steps"] = 4
            dialogue_manager.STRATEGY_CONFIG["EXPLORE"]["steps"] = 4
            variant_label = "ew=0.1*"

        print(f"Running variant {variant} ({variant_label})...")
        for dlg in DIALOGUES:
            did = dlg["id"]
            # Reset WM for each dialogue
            brain.wm = WorkingMemory()
            for turn_idx, (intent, prompt) in enumerate(dlg["turns"]):
                resp = brain.get_response(prompt)
                rows.append({
                    "dialogue_id": did,
                    "turn": turn_idx + 1,
                    "intent": intent,
                    "prompt": prompt,
                    "variant": variant,
                    "response": resp,
                    "coherence": "",
                    "informativeness": "",
                    "naturalness": "",
                })

    fieldnames = [
        "dialogue_id", "turn", "intent", "prompt",
        "variant", "response", "coherence", "informativeness", "naturalness",
    ]
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved {len(rows)} rows to {OUTPUT_CSV}")
    print(f"Variants: {len(set(r['variant'] for r in rows))} ({sorted(set(r['variant'] for r in rows))})")
    print(f"Dialogues: {len(DIALOGUES)}")
    total_turns = sum(len(d["turns"]) for d in DIALOGUES)
    print(f"Turns per dialogue: {total_turns}")
    print(f"\nScoring legend (enter in CSV):")
    print(f"  1 = poor, 2 = fair, 3 = good, 4 = very good, 5 = excellent")
    print(f"  Coherence:       Does the response logically follow context?")
    print(f"  Informativeness: Does it add relevant knowledge?")
    print(f"  Naturalness:     Does it sound human-like?")
    print(f"\nComposite = 0.45*C + 0.35*I + 0.20*N")


def main():
    parser = argparse.ArgumentParser(description="MaBrain Human Evaluation Protocol")
    parser.add_argument("--run", action="store_true", help="Run all dialogues and generate CSV")
    parser.add_argument("--score", action="store_true", help="Open CSV for scoring")
    args = parser.parse_args()

    if args.run:
        run_eval()
    elif args.score:
        abs_path = os.path.abspath(OUTPUT_CSV)
        if not os.path.exists(abs_path):
            print(f"CSV not found at {abs_path}. Run --run first.")
            return
        print(f"Open {abs_path} in your spreadsheet editor.")
        print("Fill in coherence, informativeness, naturalness (1-5) for each row.")
        print("Save the file when done.")
        os.startfile(abs_path) if sys.platform == "win32" else None
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

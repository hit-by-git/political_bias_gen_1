#!/usr/bin/env python3
"""
src/split_dataset.py

Splits the synthetic dataset into train, validation, and test sets.
Guarantees strict event-level atomicity: all variants of an event (neutral,
hard_negative, misinfo) remain strictly in the same split to prevent data leakage.

Stratification is performed jointly over (language, tactic) combinations.
"""

import os
import json
import random
import argparse
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Any


def load_dataset_by_events(input_path: str) -> Dict[Any, List[Dict[str, Any]]]:
    """Load jsonl file and group records by event_id."""
    events = defaultdict(list)
    total_records = 0
    with open(input_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"⚠️ Warning: Skipped invalid JSON at line {line_num}: {e}")
                continue
            
            event_id = record.get("event_id")
            if event_id is None:
                raise ValueError(f"Record at line {line_num} missing 'event_id'!")
            events[event_id].append(record)
            total_records += 1

    print(f"📥 Loaded {total_records} records across {len(events)} unique events from {input_path}")
    return events


def get_event_stratum(posts: List[Dict[str, Any]]) -> Tuple[str, str]:
    """Determine the stratification key (language, tactic) for an event."""
    lang = posts[0].get("language", "Unknown")
    misinfo_posts = [p for p in posts if p.get("variant") == "misinfo"]
    tactic = misinfo_posts[0].get("tactic", "none") if misinfo_posts else "none"
    return (lang, tactic)


def stratified_event_split(
    events: Dict[Any, List[Dict[str, Any]]],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42
) -> Tuple[List[Any], List[Any], List[Any]]:
    """
    Split event IDs into train, val, and test partitions using joint
    stratification over (language, tactic).
    """
    assert abs((train_ratio + val_ratio + test_ratio) - 1.0) < 1e-5, "Ratios must sum to 1.0"
    rng = random.Random(seed)

    # Group events by stratum
    strata = defaultdict(list)
    for event_id, posts in events.items():
        stratum = get_event_stratum(posts)
        strata[stratum].append(event_id)

    train_events, val_events, test_events = [], [], []

    for stratum, ev_ids in sorted(strata.items()):
        rng.shuffle(ev_ids)
        n = len(ev_ids)
        
        if n == 1:
            # Single-event stratum goes to training
            train_events.extend(ev_ids)
            continue
        elif n == 2:
            # Split between train and val
            train_events.append(ev_ids[0])
            val_events.append(ev_ids[1])
            continue

        n_train = max(1, int(round(n * train_ratio)))
        n_val = max(1, int(round(n * val_ratio))) if n >= 3 else 1
        
        # Adjust test count so sum equals n
        n_test = n - n_train - n_val
        if n_test < 0:
            n_train = n - n_val - 1
            n_test = 1
        elif n_test == 0 and n >= 4:
            n_train -= 1
            n_test = 1

        train_part = ev_ids[:n_train]
        val_part = ev_ids[n_train:n_train + n_val]
        test_part = ev_ids[n_train + n_val:]

        train_events.extend(train_part)
        val_events.extend(val_part)
        test_events.extend(test_part)

    # Verification: check no overlap
    set_train, set_val, set_test = set(train_events), set(val_events), set(test_events)
    assert len(set_train & set_val) == 0, "Data leakage detected: train & val overlap!"
    assert len(set_train & set_test) == 0, "Data leakage detected: train & test overlap!"
    assert len(set_val & set_test) == 0, "Data leakage detected: val & test overlap!"
    assert len(set_train | set_val | set_test) == len(events), "Mismatch in total partitioned events!"

    return train_events, val_events, test_events


def save_split(
    output_path: str,
    event_ids: List[Any],
    events: Dict[Any, List[Dict[str, Any]]],
    split_name: str
) -> int:
    """Write records for given event IDs to jsonl with updated 'split' field."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    count = 0
    with open(output_path, "w", encoding="utf-8") as f:
        for ev_id in event_ids:
            for post in events[ev_id]:
                # Update split metadata
                record = dict(post)
                record["split"] = split_name
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                count += 1
    return count


def print_split_statistics(
    split_name: str,
    event_ids: List[Any],
    events: Dict[Any, List[Dict[str, Any]]]
) -> Dict[str, Any]:
    """Calculate and print statistics for a split."""
    posts = [p for ev_id in event_ids for p in events[ev_id]]
    variant_counts = Counter(p.get("variant") for p in posts)
    lang_counts = Counter(p.get("language") for p in posts)
    tactic_counts = Counter(p.get("tactic") for p in posts if p.get("variant") == "misinfo")

    print(f"\n{'=' * 60}")
    print(f"📊 SPLIT: {split_name.upper()} ({len(event_ids)} events, {len(posts)} posts)")
    print(f"{'=' * 60}")
    print(f"  Variants:  {dict(variant_counts)}")
    print(f"  Languages: {dict(lang_counts)}")
    print(f"  Tactics:   {dict(tactic_counts)}")

    return {
        "num_events": len(event_ids),
        "num_posts": len(posts),
        "variants": dict(variant_counts),
        "languages": dict(lang_counts),
        "tactics": dict(tactic_counts)
    }


def main():
    parser = argparse.ArgumentParser(description="Stratified event-level train/val/test splitting.")
    parser.add_argument(
        "--input",
        type=str,
        default="output/output/synthetic_dataset.jsonl",
        help="Path to synthetic_dataset.jsonl"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/splits",
        help="Directory to save train.jsonl, val.jsonl, test.jsonl"
    )
    parser.add_argument("--train_ratio", type=float, default=0.70)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--test_ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Fallback if default not found
    if not os.path.exists(args.input):
        fallback = "output/synthetic_dataset.jsonl"
        if os.path.exists(fallback):
            print(f"Input {args.input} not found, falling back to {fallback}")
            args.input = fallback
        else:
            raise FileNotFoundError(f"Neither {args.input} nor {fallback} exists!")

    events = load_dataset_by_events(args.input)
    train_ids, val_ids, test_ids = stratified_event_split(
        events,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed
    )

    out_train = os.path.join(args.output_dir, "train.jsonl")
    out_val = os.path.join(args.output_dir, "val.jsonl")
    out_test = os.path.join(args.output_dir, "test.jsonl")

    n_train = save_split(out_train, train_ids, events, "train")
    n_val = save_split(out_val, val_ids, events, "val")
    n_test = save_split(out_test, test_ids, events, "test")

    summary = {
        "train": print_split_statistics("train", train_ids, events),
        "val": print_split_statistics("val", val_ids, events),
        "test": print_split_statistics("test", test_ids, events),
    }

    summary_file = os.path.join(args.output_dir, "split_summary.json")
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n✅ All splits successfully generated in `{args.output_dir}/`")
    print(f"   Train: {n_train} posts | Val: {n_val} posts | Test: {n_test} posts")
    print(f"   Summary saved to: {summary_file}")


if __name__ == "__main__":
    main()

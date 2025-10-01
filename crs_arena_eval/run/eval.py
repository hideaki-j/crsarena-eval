#!/usr/bin/env python3
"""Evaluate FACE run file against CRS Arena gold annotations."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
from scipy.stats import pearsonr, spearmanr


DEFAULT_EVAL_PATH = Path(__file__).resolve().parent.parent / "crs_arena_eval.json"
TURN_ASPECTS = ["relevance", "interestingness"]
DIALOGUE_ASPECTS = ["understanding", "task_completion", "interest_arousal", "efficiency", "dialogue_overall"]
DATASET_ORDER = ("redial", "opendialkg")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute CRS Arena evaluation metrics from a combined run file.")
    parser.add_argument("--run_file", type=Path, required=True, help="Path to combined run JSON file (face_run.json)")
    parser.add_argument("--eval_file", type=Path, default=DEFAULT_EVAL_PATH, help="Path to evaluation gold JSON (default ../crs_arena_eval.json)")
    return parser.parse_args()


def load_gold(gold_path: Path) -> tuple[Dict[Tuple[str, int], Dict[str, float]], Dict[str, Dict[str, float]]]:
    with gold_path.open() as f:
        gold_data = json.load(f)

    turn_gold: Dict[Tuple[str, int], Dict[str, float]] = {}
    dial_gold: Dict[str, Dict[str, float]] = {}

    for dialog in gold_data:
        conv_id = dialog["conv_id"]
        dial_gold[conv_id] = dialog.get("dial_level_aggregated", {})
        for turn in dialog["dialogue"]:
            if turn.get("role") != "ASST":
                continue
            turn_gold[(conv_id, turn["turn_ind"])] = turn.get("turn_level_aggregated", {})

    return turn_gold, dial_gold


def dataset_from_conv_id(conv_id: str) -> str:
    try:
        return conv_id.split("_")[1]
    except IndexError:
        raise ValueError(f"Unexpected conv_id format: {conv_id}")


def load_run_predictions(path: Path) -> tuple[Dict[Tuple[str, int], Dict[str, float]], Dict[str, Dict[str, float]]]:
    with path.open() as f:
        run_data = json.load(f)

    turn_preds: Dict[Tuple[str, int], Dict[str, float]] = {}
    dial_preds: Dict[str, Dict[str, float]] = {}

    for dialog in run_data:
        conv_id = dialog["conv_id"]
        turn_list = dialog.get("turns", [])
        for turn in turn_list:
            turn_ind = int(turn["turn_ind"])
            turn_preds[(conv_id, turn_ind)] = {
                aspect: float(value)
                for aspect, value in turn.get("turn_level_pred", {}).items()
                if aspect in TURN_ASPECTS
            }
        dial_preds[conv_id] = {
            aspect: float(dialog.get("dial_level_pred", {}).get(aspect))
            for aspect in DIALOGUE_ASPECTS
            if aspect in dialog.get("dial_level_pred", {})
        }

    return turn_preds, dial_preds


def compute_metrics(records: Iterable[Tuple[str, float, float]]) -> Dict[str, Dict[str, float]]:
    by_dataset: Dict[str, Dict[str, list]] = defaultdict(lambda: {"pred": [], "gold": []})
    for dataset, pred, gold in records:
        by_dataset[dataset]["pred"].append(pred)
        by_dataset[dataset]["gold"].append(gold)

    metrics: Dict[str, Dict[str, float]] = {}
    for dataset in DATASET_ORDER:
        preds = by_dataset.get(dataset, {}).get("pred", [])
        golds = by_dataset.get(dataset, {}).get("gold", [])
        if not preds:
            metrics[dataset] = {"pearson": float("nan"), "spearman": float("nan")}
            continue
        preds_arr = np.array(preds, dtype=float)
        golds_arr = np.array(golds, dtype=float)
        pearson = pearsonr(preds_arr, golds_arr)[0]
        spearman = spearmanr(preds_arr, golds_arr)[0]
        metrics[dataset] = {"pearson": pearson, "spearman": spearman}
    return metrics


def evaluate_turn_level(turn_preds, turn_gold):
    results = {}
    for aspect in TURN_ASPECTS:
        records = []
        for key, gold_aspects in turn_gold.items():
            if aspect not in gold_aspects:
                continue
            if key not in turn_preds or aspect not in turn_preds[key]:
                continue
            dataset = dataset_from_conv_id(key[0])
            records.append((dataset, turn_preds[key][aspect], gold_aspects[aspect]))
        results[aspect] = compute_metrics(records)
    return results


def evaluate_dialogue_level(dial_preds, dial_gold):
    results = {}
    for aspect in DIALOGUE_ASPECTS:
        records = []
        for conv_id, gold_aspects in dial_gold.items():
            if aspect not in gold_aspects:
                continue
            if conv_id not in dial_preds or aspect not in dial_preds[conv_id]:
                continue
            dataset = dataset_from_conv_id(conv_id)
            records.append((dataset, dial_preds[conv_id][aspect], gold_aspects[aspect]))
        results[aspect] = compute_metrics(records)
    return results


def format_table(title: str, aspects: Iterable[str], metrics: Dict[str, Dict[str, Dict[str, float]]]):
    header = f"{title}\n" + "=" * len(title)
    print(header)
    print(f"{'Aspect':<18} {'ReDial (P/S)':>18} {'OpenDialKG (P/S)':>22}")
    for aspect in aspects:
        stats = metrics.get(aspect, {})
        red = stats.get("redial", {"pearson": float("nan"), "spearman": float("nan")})
        odk = stats.get("opendialkg", {"pearson": float("nan"), "spearman": float("nan")})
        red_str = f"{red['pearson']:.3f}/{red['spearman']:.3f}"
        odk_str = f"{odk['pearson']:.3f}/{odk['spearman']:.3f}"
        print(f"{aspect:<18} {red_str:>18} {odk_str:>22}")
    print()


def main() -> None:
    args = parse_args()
    turn_gold, dial_gold = load_gold(args.eval_file)
    turn_preds, dial_preds = load_run_predictions(args.run_file)

    turn_results = evaluate_turn_level(turn_preds, turn_gold)
    dial_results = evaluate_dialogue_level(dial_preds, dial_gold)

    format_table("Turn-level Aspects", TURN_ASPECTS, turn_results)
    format_table("Dialogue-level Aspects", DIALOGUE_ASPECTS, dial_results)


if __name__ == "__main__":
    main()

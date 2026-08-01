"""
Pipeline Orchestrator for Taxor Bill Extraction.

Main entry point that coordinates the end-to-end execution flow:
1. Load environment variables (.env)
2. Read bill images from `data/bills/`
3. Load ground truth from `data/ground_truth.json`
4. Invoke model extractors (Gemini, Claude, OpenRouter)
5. Save raw extractions to `results/raw_extractions.json`
6. Score extractions against ground truth in `data/ground_truth.json`
7. Calculate input/output token costs
8. Save summary matrices (`results/scores.csv`, `results/cost_summary.csv`)
9. (Optional) Push verified extractions to Zoho Books via `ZohoBooksClient`
"""

import argparse
import os
import sys
import json
import pandas as pd
from glob import glob
from typing import Dict, Any, List

# Ensure project root directory is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from dotenv import load_dotenv
from src.models.gemini_extractor import GeminiExtractor
from src.models.claude_extractor import ClaudeExtractor
from src.models.openrouter_extractor import OpenRouterExtractor
from src.eval.scorer import FieldScorer
from src.eval.cost_tracker import CostTracker, PRICING_TABLE

MODEL_MAP = {
    "gemini": GeminiExtractor,
    "claude": ClaudeExtractor,
    "openrouter": OpenRouterExtractor,
}

MODEL_PRICING_MAP = {
    "gemini": "gemini-3.5-flash-lite",
    "claude": "claude-sonnet-4-6",
    "openrouter": "google/gemma-4-26b-a4b-it:free",
}

ESTIMATED_INPUT_TOKENS_PER_BILL = 1200
ESTIMATED_OUTPUT_TOKENS_PER_BILL = 300
SAFETY_COST_CEILING_USD = 2.00


def load_ground_truth() -> Dict[str, Any]:
    """Load ground truth JSON dictionary."""
    gt_path = os.path.join(PROJECT_ROOT, "data", "ground_truth.json")
    if not os.path.exists(gt_path):
        return {}
    with open(gt_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list) and len(data) > 0:
        return data[0]
    return data


def estimate_pipeline_cost(models: list, bill_count: int) -> float:
    """Estimate total run cost across specified models and bill count."""
    total_cost = 0.0
    for model_key in models:
        model_name = MODEL_PRICING_MAP.get(model_key.lower(), model_key)
        pricing = PRICING_TABLE.get(model_name, {"input_per_m": 0.0, "output_per_m": 0.0})
        model_cost = bill_count * (
            (ESTIMATED_INPUT_TOKENS_PER_BILL / 1_000_000) * pricing["input_per_m"]
            + (ESTIMATED_OUTPUT_TOKENS_PER_BILL / 1_000_000) * pricing["output_per_m"]
        )
        total_cost += model_cost
    return round(total_cost, 4)


def main():
    load_dotenv()
    parser = argparse.ArgumentParser(description="Taxor Bill Extraction & Evaluation Pipeline")
    parser.add_argument("--models", nargs="+", default=["gemini", "openrouter"], help="Models to run")
    parser.add_argument("--push-to-zoho", action="store_true", help="Push extractions to Zoho Books")
    parser.add_argument("--dry-run", action="store_true", help="Estimate tokens & costs without calling APIs")
    parser.add_argument("--confirm", action="store_true", help="Confirm execution if estimated cost exceeds $2.00")
    args = parser.parse_args()

    # Discover bill image files
    bills_dir = os.path.join(PROJECT_ROOT, "data", "bills")
    bill_files = sorted(
        glob(os.path.join(bills_dir, "*.jpg"))
        + glob(os.path.join(bills_dir, "*.jpeg"))
        + glob(os.path.join(bills_dir, "*.png"))
    )
    bill_count = len(bill_files) if len(bill_files) > 0 else 15

    estimated_cost = estimate_pipeline_cost(args.models, bill_count)
    summary_msg = f"Estimated total cost: ${estimated_cost:.2f} across {len(args.models)} models for {bill_count} bills."

    if args.dry_run:
        print(f"[DRY-RUN] {summary_msg}")
        print("[DRY-RUN] No API calls were made. Dry run complete.")
        sys.exit(0)

    # Cost Safety Guardrail
    if estimated_cost > SAFETY_COST_CEILING_USD and not args.confirm:
        print(f"[WARNING] {summary_msg}")
        print(f"[ERROR] Estimated cost exceeds safety ceiling of ${SAFETY_COST_CEILING_USD:.2f}.")
        print("Run with '--confirm' flag to proceed with real API execution, or use '--dry-run' to inspect details.")
        sys.exit(1)

    print("=" * 80)
    print(f"STARTING PIPELINE RUN FOR MODELS: {args.models}")
    print(f"Total Bills Found: {len(bill_files)}")
    print(f"Estimated Cost: ${estimated_cost:.2f}")
    print("=" * 80)

    ground_truth_map = load_ground_truth()
    scorer = FieldScorer()

    raw_extractions: Dict[str, Dict[str, Any]] = {}
    all_score_rows: List[Dict[str, Any]] = []
    usage_records: List[Dict[str, Any]] = []

    for model_key in args.models:
        key_lower = model_key.lower()
        if key_lower not in MODEL_MAP:
            print(f"[WARNING] Skipping unknown model key: {model_key}")
            continue

        extractor_cls = MODEL_MAP[key_lower]
        extractor = extractor_cls()
        model_name = extractor.model_name
        print(f"\n>>> Running Extractor: {key_lower} (Model ID: {model_name})")

        raw_extractions[key_lower] = {}

        for bill_path in bill_files:
            filename = os.path.basename(bill_path)
            print(f"  Extracting {filename}...", end="", flush=True)

            result = extractor.extract(bill_path)
            raw_extractions[key_lower][filename] = result.to_dict()

            gt_record = ground_truth_map.get(filename, {})
            scores = scorer.score_extraction(result.to_dict(), gt_record)

            actual_model_used = result.actual_model or model_name
            cost = CostTracker.calculate_cost(actual_model_used, result.input_tokens, result.output_tokens)

            # Record detailed field scores for scores.csv
            score_row = {
                "model": key_lower,
                "model_id": actual_model_used,
                "bill_image": filename,
                "vendor": scores["vendor"],
                "bill_number": scores["bill_number"],
                "date": scores["date"],
                "amount": scores["amount"],
                "currency": scores["currency"],
                "tax_details": scores["tax_details"],
                "overall_accuracy": scores["overall_accuracy"],
                "is_success": result.is_success,
            }
            all_score_rows.append(score_row)

            # Record usage details for cost_summary.csv
            usage_records.append({
                "model": key_lower,
                "model_id": actual_model_used,
                "input_tokens": result.input_tokens,
                "output_tokens": result.output_tokens,
                "latency_seconds": result.latency_seconds,
                "cost": cost,
                "overall_accuracy": scores["overall_accuracy"],
                "is_success": result.is_success,
            })

            status_str = "OK" if result.is_success else "FAIL"
            print(f" [{status_str}] Acc: {scores['overall_accuracy']*100:.0f}% | {result.latency_seconds}s | {result.input_tokens}/{result.output_tokens} tok")

    # Ensure results directory exists
    results_dir = os.path.join(PROJECT_ROOT, "results")
    os.makedirs(results_dir, exist_ok=True)

    # 1. Save raw_extractions.json
    raw_path = os.path.join(results_dir, "raw_extractions.json")
    with open(raw_path, "w", encoding="utf-8") as f:
        json.dump(raw_extractions, f, indent=2)
    print(f"\n[OUTPUT] Saved raw extractions to {raw_path}")

    # 2. Save scores.csv
    scores_df = pd.DataFrame(all_score_rows)
    scores_path = os.path.join(results_dir, "scores.csv")
    scores_df.to_csv(scores_path, index=False)
    print(f"[OUTPUT] Saved detailed scores to {scores_path}")

    # 3. Save cost_summary.csv
    summary_data = CostTracker.summarize_costs(usage_records)
    # Merge aggregate accuracy into cost summary
    for s in summary_data:
        m = s["model"]
        model_scores = [r["overall_accuracy"] for r in all_score_rows if r["model"] == m]
        avg_acc = round(sum(model_scores) / len(model_scores), 4) if model_scores else 0.0
        s["overall_accuracy"] = avg_acc

    summary_df = pd.DataFrame(summary_data)
    cost_path = os.path.join(results_dir, "cost_summary.csv")
    summary_df.to_csv(cost_path, index=False)
    print(f"[OUTPUT] Saved cost summary to {cost_path}")

    # (Optional) Zoho Books Push
    if args.push_to_zoho:
        print("\n[ZOHO] --push-to-zoho flag set. Initializing ZohoBooksClient...")
        from src.zoho.zoho_client import ZohoBooksClient
        zoho = ZohoBooksClient()
        print("[ZOHO] Zoho Books integration pending OAuth configuration.")

    print("\n" + "=" * 80)
    print("PIPELINE EXECUTION COMPLETE!")
    print("=" * 80)


if __name__ == "__main__":
    main()

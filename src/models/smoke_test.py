"""
Smoke Test Script for Taxor Bill Extractors.

Runs Gemini, Claude, and OpenAI extractors on a single bill image to verify
that API calls, image encoding, token parsing, and JSON extraction work end-to-end.
"""

import sys
import os
import glob
from pprint import pprint

# Ensure project root directory is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from dotenv import load_dotenv
from src.models.gemini_extractor import GeminiExtractor
from src.models.claude_extractor import ClaudeExtractor
from src.models.openrouter_extractor import OpenRouterExtractor


def run_smoke_test(image_path: str):
    """
    Run all three extractors on a single bill image and print results.
    """
    load_dotenv()
    print("=" * 80)
    print(f"RUNNING SMOKE TEST ON BILL IMAGE: {image_path}")
    print("=" * 80)

    if not os.path.exists(image_path):
        print(f"[ERROR] Specified bill image not found: {image_path}")
        sys.exit(1)

    extractors = [
        ("Google Gemini", GeminiExtractor()),
        ("Anthropic Claude", ClaudeExtractor()),
        ("OpenRouter", OpenRouterExtractor()),
    ]

    for name, extractor in extractors:
        print(f"\n--- Testing {name} Extractor (Model: {extractor.model_name}) ---")
        try:
            result = extractor.extract(image_path)
            if result.is_success:
                print(f"[STATUS] SUCCESS (Latency: {result.latency_seconds}s)")
                print(f"[MODEL] Requested: {extractor.model_name} | Actual Served: {result.actual_model or extractor.model_name}")
                print(f"[USAGE] Input Tokens: {result.input_tokens} | Output Tokens: {result.output_tokens}")
                print("[EXTRACTED FIELDS]")
                print(f"  Vendor      : {result.vendor}")
                print(f"  Bill Number : {result.bill_number}")
                print(f"  Date        : {result.date}")
                print(f"  Amount      : {result.amount} {result.currency}")
                print(f"  Tax Details : {result.tax_details}")
            else:
                print(f"[STATUS] FAILED (Reason: {result.error or 'API_ERROR encountered'})")
                print(f"[USAGE] Input Tokens: {result.input_tokens} | Output Tokens: {result.output_tokens}")
            print(f"[RAW RESPONSE]\n{result.raw_response}\n")
        except Exception as e:
            print(f"[STATUS] FAILED (Exception: {e})")


def main():
    # If image path provided as CLI argument, use it; otherwise search data/bills/
    if len(sys.argv) > 1:
        target_image = sys.argv[1]
    else:
        # Search for first image in data/bills/
        images = glob.glob("data/bills/*.jpg*") + glob.glob("data/bills/*.png*")
        if not images:
            print("[ERROR] No bill images found in data/bills/")
            sys.exit(1)
        target_image = images[0]

    run_smoke_test(target_image)


if __name__ == "__main__":
    main()

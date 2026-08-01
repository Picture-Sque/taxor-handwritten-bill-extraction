"""
Push Sample Verified Extractions to Zoho Books.

Reads results/raw_extractions.json and pushes OpenRouter (Gemma-4) extractions
for bills 01, 04, 05, and 08 (100% accuracy clean test cases) to Zoho Books as real Expenses.
"""

import os
import sys
import json
import argparse
from typing import Dict, Any

# Ensure project root directory is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.zoho.zoho_client import ZohoBooksClient

SAMPLE_BILLS = ["bill_01.jpg", "bill_04.jpg", "bill_05.jpg", "bill_08.jpg"]


def load_sample_extractions() -> Dict[str, Dict[str, Any]]:
    """Load OpenRouter extractions for the target sample bills."""
    raw_path = os.path.join(PROJECT_ROOT, "results", "raw_extractions.json")
    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"Raw extractions file not found at {raw_path}. Run pipeline first.")

    with open(raw_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    openrouter_data = data.get("openrouter", {})
    samples = {}
    for bill_name in SAMPLE_BILLS:
        if bill_name in openrouter_data:
            samples[bill_name] = openrouter_data[bill_name]
        else:
            print(f"[WARNING] {bill_name} not found in openrouter extractions.")

    return samples


def main():
    parser = argparse.ArgumentParser(description="Push Sample Extractions to Zoho Books")
    parser.add_argument("--first-only", action="store_true", help="Push only bill_01.jpg and show raw request/response details")
    args = parser.parse_args()

    zoho = ZohoBooksClient()
    samples = load_sample_extractions()

    if not samples:
        print("[ERROR] No sample extractions available to push.")
        sys.exit(1)

    print("=" * 80)
    print("ZOHO BOOKS EXPENSE CREATION - SAMPLE PUSH")
    print(f"Target Bills: {list(samples.keys())}")
    print(f"Target Organization ID: {zoho.organization_id}")
    print(f"Target Expense Account ID: {zoho.expense_account_id}")
    print("=" * 80)

    # Process first bill first for confirmation
    target_list = SAMPLE_BILLS if not args.first_only else SAMPLE_BILLS[:1]
    created_expenses = []

    for bill_name in target_list:
        if bill_name not in samples:
            continue

        rec = samples[bill_name]
        vendor = rec.get("vendor") or "Unknown Vendor"
        bill_num = rec.get("bill_number") or "N/A"
        date = rec.get("date") or "2026-07-15"
        amount = rec.get("amount") or 0.0
        currency = rec.get("currency", "INR") or "INR"

        desc = f"Bill Extraction: {vendor} (Bill #{bill_num})"

        print(f"\n>>> Creating Expense for {bill_name}:")
        print(f"    Vendor      : {vendor}")
        print(f"    Bill Number : {bill_num}")
        print(f"    Date        : {date}")
        print(f"    Amount      : {amount} {currency}")
        print(f"    Description : {desc}")

        # Execute creation
        try:
            res_json = zoho.create_expense(
                vendor=vendor,
                date=date,
                amount=amount,
                currency=currency,
                description=desc,
            )

            expense_record = res_json.get("expense", {})
            expense_id = expense_record.get("expense_id") or res_json.get("expense_id")

            print(f"\n[EXACT API RESPONSE JSON]:")
            print(json.dumps(res_json, indent=2))
            print(f"\n[SUCCESS] Created Zoho Expense ID: {expense_id}")

            created_expenses.append({
                "bill_image": bill_name,
                "vendor": vendor,
                "amount": amount,
                "expense_id": expense_id,
            })

        except Exception as e:
            print(f"[FAILED] Could not create expense for {bill_name}: {e}")

    print("\n" + "=" * 80)
    print("SUMMARY OF CREATED ZOHO EXPENSES:")
    for item in created_expenses:
        print(f"  {item['bill_image']} -> Expense ID: {item['expense_id']} ({item['vendor']}, {item['amount']} INR)")
    print("=" * 80)


if __name__ == "__main__":
    main()

"""
Field-Level Scorer and String Distance Evaluation.

Compares extracted bill dictionaries against ground truth annotations.
- Numerical fields (amount): exact match or within 5% tolerance (e.g., for smudged digits like bill_15.jpg)
- Categorical/date/bill_number/currency: normalized exact match or null-equivalence
- Vendor name & tax details: fuzzy string similarity using RapidFuzz
"""

from typing import Dict, Any, List
from rapidfuzz import fuzz


class FieldScorer:
    """
    Evaluates field-level extraction accuracy against ground truth annotations.
    """

    def __init__(self, fuzzy_threshold: float = 80.0, amount_tolerance_pct: float = 0.05):
        """
        Args:
            fuzzy_threshold (float): RapidFuzz similarity threshold (0-100) for textual fields.
            amount_tolerance_pct (float): Numerical tolerance percentage for amount matching (0.05 = 5%).
        """
        self.fuzzy_threshold = fuzzy_threshold
        self.amount_tolerance_pct = amount_tolerance_pct

    def score_field(self, field_name: str, extracted_val: Any, gt_val: Any) -> float:
        """
        Calculate binary/proportional score (0.0 to 1.0) for a single field.
        """
        # Both null / None / empty
        if (extracted_val is None or str(extracted_val).strip().lower() in ("", "null", "none")) and (
            gt_val is None or str(gt_val).strip().lower() in ("", "null", "none")
        ):
            return 1.0

        # One null, other not
        if (extracted_val is None or str(extracted_val).strip().lower() in ("", "null", "none")) != (
            gt_val is None or str(gt_val).strip().lower() in ("", "null", "none")
        ):
            return 0.0

        # Amount evaluation with tolerance
        if field_name == "amount":
            try:
                ext_amt = float(extracted_val)
                gt_amt = float(gt_val)
                if abs(ext_amt - gt_amt) < 1e-4:
                    return 1.0
                # Check percentage tolerance (e.g., 5% tolerance)
                if gt_amt > 0 and (abs(ext_amt - gt_amt) / gt_amt) <= self.amount_tolerance_pct:
                    return 1.0
                return 0.0
            except (ValueError, TypeError):
                return 0.0

        # Bill Number: strip all internal whitespace & punctuation spacing, then apply character-level fuzz.ratio
        if field_name == "bill_number":
            s1_clean = "".join(str(extracted_val).split()).lower()
            s2_clean = "".join(str(gt_val).split()).lower()
            if s1_clean == s2_clean:
                return 1.0
            ratio = fuzz.ratio(s1_clean, s2_clean)
            return 1.0 if ratio >= self.fuzzy_threshold else round(ratio / 100.0, 2)

        # Vendor & tax_details: token-based fuzzy matching
        if field_name in ("vendor", "tax_details"):
            s1 = str(extracted_val).strip().lower()
            s2 = str(gt_val).strip().lower()
            ratio = fuzz.token_set_ratio(s1, s2)
            return 1.0 if ratio >= self.fuzzy_threshold else round(ratio / 100.0, 2)

        # Exact normalized string matching for date, currency
        s1 = str(extracted_val).strip().lower()
        s2 = str(gt_val).strip().lower()
        return 1.0 if s1 == s2 else 0.0

    def score_extraction(self, extracted: Dict[str, Any], ground_truth: Dict[str, Any]) -> Dict[str, float]:
        """
        Compare single extracted record against ground truth record.

        Returns:
            Dict[str, float]: Field scores + overall average accuracy.
        """
        fields = ["vendor", "bill_number", "date", "amount", "currency", "tax_details"]
        scores = {}
        total = 0.0

        for f in fields:
            score = self.score_field(f, extracted.get(f), ground_truth.get(f))
            scores[f] = score
            total += score

        scores["overall_accuracy"] = round(total / len(fields), 4)
        return scores

    def aggregate_scores(self, bill_scores: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Aggregate list of bill scores for a model into aggregate averages per field.
        """
        if not bill_scores:
            return {}

        fields = ["vendor", "bill_number", "date", "amount", "currency", "tax_details", "overall_accuracy"]
        aggregates = {f: 0.0 for f in fields}

        for score_dict in bill_scores:
            for f in fields:
                aggregates[f] += score_dict.get(f, 0.0)

        count = len(bill_scores)
        return {f: round(aggregates[f] / count, 4) for f in fields}

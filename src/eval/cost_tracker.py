"""
API Token Cost Tracker.

This module tracks and calculates financial costs of extraction runs based on
provider token pricing models (input tokens, output tokens) for Gemini, Claude, and OpenRouter APIs.
"""

from typing import Dict, Any, List

# Standard pricing table per 1M tokens (USD) - subject to provider updates
PRICING_TABLE = {
    "gemini-3.5-flash-lite": {"input_per_m": 0.30, "output_per_m": 2.50},
    "gemini-3.6-flash": {"input_per_m": 1.50, "output_per_m": 7.50},
    "claude-sonnet-4-6": {"input_per_m": 3.00, "output_per_m": 15.00},
    "google/gemma-4-26b-a4b-it:free": {"input_per_m": 0.00, "output_per_m": 0.00},
    "openrouter/free": {"input_per_m": 0.00, "output_per_m": 0.00},
}


class CostTracker:
    """
    Calculates cost per model based on input/output token usage.
    """

    @staticmethod
    def calculate_cost(model_name: str, input_tokens: int, output_tokens: int) -> float:
        """
        Calculate total USD cost for an extraction request.

        Args:
            model_name (str): Identifier of the LLM model.
            input_tokens (int): Count of input tokens (text + image tokens).
            output_tokens (int): Count of generated output tokens.

        Returns:
            float: Cost in USD rounded to 6 decimal places.
        """
        pricing = PRICING_TABLE.get(model_name, {"input_per_m": 0.0, "output_per_m": 0.0})
        input_cost = (input_tokens / 1_000_000.0) * pricing["input_per_m"]
        output_cost = (output_tokens / 1_000_000.0) * pricing["output_per_m"]
        return round(input_cost + output_cost, 6)

    @staticmethod
    def summarize_costs(usage_records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Summarize total token usage, average latency, and cost per model across evaluation runs.

        Args:
            usage_records (List[Dict[str, Any]]): List of dicts containing:
                - model (str)
                - input_tokens (int)
                - output_tokens (int)
                - latency_seconds (float)
                - cost (float)

        Returns:
            List[Dict[str, Any]]: Summary table rows per model.
        """
        summary_by_model: Dict[str, Dict[str, Any]] = {}

        for rec in usage_records:
            model = rec["model"]
            if model not in summary_by_model:
                summary_by_model[model] = {
                    "model": model,
                    "total_runs": 0,
                    "successful_runs": 0,
                    "total_input_tokens": 0,
                    "total_output_tokens": 0,
                    "total_latency_seconds": 0.0,
                    "total_cost_usd": 0.0,
                }

            s = summary_by_model[model]
            s["total_runs"] += 1
            if rec.get("is_success", True):
                s["successful_runs"] += 1
            s["total_input_tokens"] += rec.get("input_tokens", 0)
            s["total_output_tokens"] += rec.get("output_tokens", 0)
            s["total_latency_seconds"] += rec.get("latency_seconds", 0.0)
            s["total_cost_usd"] += rec.get("cost", 0.0)

        results = []
        for model, s in summary_by_model.items():
            runs = s["total_runs"] or 1
            results.append({
                "model": model,
                "total_runs": s["total_runs"],
                "successful_runs": s["successful_runs"],
                "total_input_tokens": s["total_input_tokens"],
                "total_output_tokens": s["total_output_tokens"],
                "avg_latency_seconds": round(s["total_latency_seconds"] / runs, 3),
                "total_cost_usd": round(s["total_cost_usd"], 6),
            })

        return results

"""
Abstract Base Extractor interface and shared extraction prompt for Vision LLMs.

DESIGN DECISION NOTE:
All model wrappers (Gemini, Claude, OpenAI) import and use the EXACT SAME extraction prompt
(`EXTRACTION_PROMPT`). Using an identical system prompt across all three vision models is critical
to ensure a fair, unbiased benchmark comparison of extraction accuracy and cost efficiency.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any
import json
import re

# Shared extraction prompt across all providers for fair benchmark comparison
EXTRACTION_PROMPT = """You are extracting structured data from a photo of a handwritten Indian bill/receipt for expense tracking. Read the image carefully, including handwritten text which may be unclear. Extract the following fields and respond ONLY with valid JSON, no other text:
{
  "vendor": "shop/vendor name if visible, else null",
  "bill_number": "invoice/bill number if visible, else null",
  "date": "date in YYYY-MM-DD format if visible, else null",
  "amount": "total amount as a number (no currency symbol) if visible, else null",
  "currency": "currency code, default 'INR' unless another currency is clearly shown",
  "tax_details": "any GST/tax amount or rate visible as a string, else null"
}
If a field is illegible or absent, use null rather than guessing."""


@dataclass
class ExtractionResult:
    """
    Standardized result data structure across all extractors.
    """
    vendor: Optional[str] = None
    bill_number: Optional[str] = None
    date: Optional[str] = None
    amount: Optional[float] = None
    currency: Optional[str] = "INR"
    tax_details: Optional[str] = None
    raw_response: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    latency_seconds: float = 0.0
    error: Optional[str] = None
    actual_model: Optional[str] = None

    @property
    def is_success(self) -> bool:
        """
        Returns True if extraction succeeded without API errors and returned genuine content.
        """
        if self.error is not None:
            return False
        if self.raw_response and self.raw_response.startswith("API_ERROR"):
            return False
        return True

    def to_dict(self) -> Dict[str, Any]:
        """Convert result dataclass to dictionary."""
        d = asdict(self)
        d["is_success"] = self.is_success
        return d


class BillExtractor(ABC):
    """
    Abstract base class for vision-capable LLM bill extractors.
    """

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key

    @abstractmethod
    def extract(self, image_path: str) -> ExtractionResult:
        """
        Extract structured fields from a handwritten bill image.

        Args:
            image_path (str): File path to bill image (JPG/PNG).

        Returns:
            ExtractionResult: Standardized dataclass result with extracted fields,
                              raw text response, token usage, and latency.
        """
        pass

    # Alias for backward compatibility
    def extract_bill_data(self, image_path: str) -> Dict[str, Any]:
        return self.extract(image_path).to_dict()


# Alias for backward compatibility
BaseExtractor = BillExtractor


def parse_json_response(raw_text: str) -> Dict[str, Any]:
    """
    Helper utility to clean and parse JSON responses from LLM outputs.
    Handles Markdown code blocks (e.g. ```json ... ```), conversational text before/after,
    and regex bracket matching fallback.
    """
    cleaned = raw_text.strip()

    # 1. Try stripping markdown code fences if present
    match_fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, re.DOTALL)
    if match_fence:
        try:
            return json.loads(match_fence.group(1).strip())
        except Exception:
            pass

    # 2. Try direct json.loads
    try:
        return json.loads(cleaned)
    except Exception:
        pass

    # 3. Fallback regex to extract first valid JSON object block {...} amidst surrounding text
    json_block_match = re.search(r"(\{[\s\S]*\})", raw_text)
    if json_block_match:
        candidate = json_block_match.group(1).strip()
        try:
            return json.loads(candidate)
        except Exception:
            pass

    # 4. Outermost brace slice fallback
    start = raw_text.find("{")
    end = raw_text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(raw_text[start : end + 1])
        except Exception:
            pass

    raise ValueError(f"Failed to parse valid JSON from model response: {raw_text}")

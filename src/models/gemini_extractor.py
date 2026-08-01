"""
Google Gemini Vision Model Extractor.

Per-1M-token pricing constants listed below are subject to provider updates.
NOTE: Double-check these figures against Google AI Studio's live pricing page:
https://ai.google.dev/pricing
"""

import os
import time
import logging
from typing import Optional
from PIL import Image

from src.models.base import (
    BillExtractor,
    ExtractionResult,
    EXTRACTION_PROMPT,
    parse_json_response,
)

# Published per-1M-token pricing (USD) for gemini-3.5-flash-lite
# (Live pricing page: https://ai.google.dev/pricing)
GEMINI_INPUT_COST_PER_M = 0.30
GEMINI_OUTPUT_COST_PER_M = 2.50

logger = logging.getLogger(__name__)


class GeminiExtractor(BillExtractor):
    """
    Extractor implementation for Google Gemini vision models using google-genai / google-generativeai SDK.
    """

    def __init__(
        self,
        model_name: str = "gemini-3.5-flash-lite",
        api_key: Optional[str] = None,
    ):
        super().__init__(api_key=api_key or os.getenv("GOOGLE_API_KEY"))
        self.model_name = model_name

    def extract(self, image_path: str) -> ExtractionResult:
        """
        Extract structured bill data from an image file using Gemini API.
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Bill image file not found: {image_path}")

        start_time = time.time()
        raw_response_text = ""
        input_tokens = 0
        output_tokens = 0

        error_msg = None
        try:
            # Try new google-genai SDK first, fallback to google-generativeai
            try:
                from google import genai

                client = genai.Client(api_key=self.api_key)
                img = Image.open(image_path)
                response = client.models.generate_content(
                    model=self.model_name,
                    contents=[img, EXTRACTION_PROMPT],
                )
                raw_response_text = response.text or ""
                if hasattr(response, "usage_metadata") and response.usage_metadata:
                    input_tokens = getattr(response.usage_metadata, "prompt_token_count", 0) or 0
                    output_tokens = getattr(response.usage_metadata, "candidates_token_count", 0) or 0

            except (ImportError, Exception) as genai_err:
                logger.debug(f"google-genai SDK fallback to google-generativeai: {genai_err}")
                import google.generativeai as genai_legacy

                if self.api_key:
                    genai_legacy.configure(api_key=self.api_key)
                model = genai_legacy.GenerativeModel(self.model_name)
                img = Image.open(image_path)
                response = model.generate_content([EXTRACTION_PROMPT, img])
                raw_response_text = response.text or ""
                if hasattr(response, "usage_metadata") and response.usage_metadata:
                    input_tokens = getattr(response.usage_metadata, "prompt_token_count", 0) or 0
                    output_tokens = getattr(response.usage_metadata, "candidates_token_count", 0) or 0

        except Exception as e:
            logger.error(f"Gemini API call failed for {image_path}: {e}")
            error_msg = str(e)
            raw_response_text = f"API_ERROR: {str(e)}"

        latency = round(time.time() - start_time, 3)

        # Parse JSON output gracefully
        result_data = {}
        if not error_msg:
            try:
                result_data = parse_json_response(raw_response_text)
            except Exception as parse_err:
                logger.warning(f"Malformed JSON from Gemini for {image_path}: {parse_err}. Raw: {raw_response_text}")

        # Safely extract amount float
        amount_val = result_data.get("amount")
        if amount_val is not None:
            try:
                amount_val = float(amount_val)
            except (ValueError, TypeError):
                amount_val = None

        return ExtractionResult(
            vendor=result_data.get("vendor"),
            bill_number=result_data.get("bill_number"),
            date=result_data.get("date"),
            amount=amount_val,
            currency=result_data.get("currency", "INR") or "INR",
            tax_details=result_data.get("tax_details"),
            raw_response=raw_response_text,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_seconds=latency,
            error=error_msg,
            actual_model=self.model_name,
        )

"""
Anthropic Claude Vision Model Extractor.

Per-1M-token pricing constants listed below are subject to provider updates.
NOTE: Double-check these figures against Anthropic's live pricing page:
https://www.anthropic.com/pricing
"""

import os
import time
import base64
import logging
import mimetypes
from typing import Optional

from src.models.base import (
    BillExtractor,
    ExtractionResult,
    EXTRACTION_PROMPT,
    parse_json_response,
)

# Published per-1M-token pricing (USD) for claude-sonnet-4-6
# (Double-check against live pricing page: https://www.anthropic.com/pricing)
CLAUDE_INPUT_COST_PER_M = 3.00
CLAUDE_OUTPUT_COST_PER_M = 15.00

logger = logging.getLogger(__name__)


class ClaudeExtractor(BillExtractor):
    """
    Extractor implementation for Anthropic Claude vision models using anthropic SDK.
    """

    def __init__(
        self,
        model_name: str = "claude-sonnet-4-6",
        api_key: Optional[str] = None,
    ):
        super().__init__(api_key=api_key or os.getenv("ANTHROPIC_API_KEY"))
        self.model_name = model_name

    def extract(self, image_path: str) -> ExtractionResult:
        """
        Extract structured bill data from an image file using Claude API.
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Bill image file not found: {image_path}")

        start_time = time.time()
        raw_response_text = ""
        input_tokens = 0
        output_tokens = 0

        # Determine media type (image/jpeg, image/png, etc.)
        media_type, _ = mimetypes.guess_type(image_path)
        if not media_type or not media_type.startswith("image/"):
            media_type = "image/jpeg"

        error_msg = None
        try:
            import anthropic

            # Encode image file to base64
            with open(image_path, "rb") as img_file:
                b64_image = base64.b64encode(img_file.read()).decode("utf-8")

            client = anthropic.Anthropic(api_key=self.api_key)
            response = client.messages.create(
                model=self.model_name,
                max_tokens=1024,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": media_type,
                                    "data": b64_image,
                                },
                            },
                            {
                                "type": "text",
                                "text": EXTRACTION_PROMPT,
                            },
                        ],
                    }
                ],
            )

            # Extract output text and usage metrics
            if response.content and len(response.content) > 0:
                raw_response_text = response.content[0].text or ""

            if hasattr(response, "usage") and response.usage:
                input_tokens = getattr(response.usage, "input_tokens", 0) or 0
                output_tokens = getattr(response.usage, "output_tokens", 0) or 0

        except Exception as e:
            logger.error(f"Claude API call failed for {image_path}: {e}")
            error_msg = str(e)
            raw_response_text = f"API_ERROR: {str(e)}"

        latency = round(time.time() - start_time, 3)

        # Parse JSON output gracefully
        result_data = {}
        if not error_msg:
            try:
                result_data = parse_json_response(raw_response_text)
            except Exception as parse_err:
                logger.warning(f"Malformed JSON from Claude for {image_path}: {parse_err}. Raw: {raw_response_text}")

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

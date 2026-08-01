"""
OpenRouter Vision Model Extractor implementation.

Per-1M-token pricing constants listed below represent OpenRouter's rate-limited free tier.
NOTE: OpenRouter free models have zero monetary cost ($0.00 / 1M tokens), but are subject
to rate limits (e.g. 20 requests per minute).
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

# Published per-1M-token pricing (USD) for OpenRouter free tier models
# (Rate-limited free tier - zero out-of-pocket cost)
OPENROUTER_INPUT_COST_PER_M = 0.00
OPENROUTER_OUTPUT_COST_PER_M = 0.00

logger = logging.getLogger(__name__)


class OpenRouterExtractor(BillExtractor):
    """
    Extractor implementation for vision-capable models via OpenRouter's OpenAI-compatible API.
    """

    def __init__(
        self,
        model_name: str = "google/gemma-4-26b-a4b-it:free",
        api_key: Optional[str] = None,
    ):
        super().__init__(api_key=api_key or os.getenv("OPENROUTER_API_KEY"))
        self.model_name = model_name

    def extract(self, image_path: str) -> ExtractionResult:
        """
        Extract structured bill data from an image file using OpenRouter's API endpoint.
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

        actual_model_name = self.model_name
        error_msg = None
        try:
            import openai

            # Encode image file to base64 data URL
            with open(image_path, "rb") as img_file:
                b64_image = base64.b64encode(img_file.read()).decode("utf-8")
            data_url = f"data:{media_type};base64,{b64_image}"

            client = openai.OpenAI(
                api_key=self.api_key or "NO_KEY_PROVIDED",
                base_url="https://openrouter.ai/api/v1",
            )
            response = client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": EXTRACTION_PROMPT},
                            {
                                "type": "image_url",
                                "image_url": {"url": data_url},
                            },
                        ],
                    }
                ],
                temperature=0.0,
            )

            # Capture actual model returned in response header/object
            if hasattr(response, "model") and response.model:
                actual_model_name = response.model

            # Extract output text and usage metrics
            if response.choices and len(response.choices) > 0:
                raw_response_text = response.choices[0].message.content or ""

            if hasattr(response, "usage") and response.usage:
                input_tokens = getattr(response.usage, "prompt_tokens", 0) or 0
                output_tokens = getattr(response.usage, "completion_tokens", 0) or 0

        except Exception as e:
            logger.error(f"OpenRouter API call failed for {image_path}: {e}")
            error_msg = str(e)
            raw_response_text = f"API_ERROR: {str(e)}"

        latency = round(time.time() - start_time, 3)

        # Parse JSON output gracefully
        result_data = {}
        if not error_msg:
            try:
                result_data = parse_json_response(raw_response_text)
            except Exception as parse_err:
                logger.warning(f"Malformed JSON from OpenRouter for {image_path}: {parse_err}. Raw: {raw_response_text}")

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
            actual_model=actual_model_name,
        )

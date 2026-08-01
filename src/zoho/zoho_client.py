"""
Zoho Books API Client.

Handles OAuth 2.0 authentication (using India data center endpoints: accounts.zoho.in & www.zohoapis.in)
and creates expense records in Zoho Books.
"""

import os
import time
import json
import logging
import requests
from typing import Dict, Any, Optional
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

ACCOUNTS_URL = "https://accounts.zoho.in/oauth/v2/token"
BOOKS_API_BASE = "https://www.zohoapis.in/books/v3"


class ZohoBooksClient:
    """
    Handles OAuth 2.0 refresh flow and expense operations for Zoho Books (India region).
    """

    def __init__(
        self,
        client_id: Optional[str] = None,
        client_secret: Optional[str] = None,
        refresh_token: Optional[str] = None,
        organization_id: Optional[str] = None,
        expense_account_id: Optional[str] = None,
    ):
        load_dotenv()
        self.client_id = client_id or os.getenv("ZOHO_CLIENT_ID")
        self.client_secret = client_secret or os.getenv("ZOHO_CLIENT_SECRET")
        self.refresh_token = refresh_token or os.getenv("ZOHO_REFRESH_TOKEN")
        self.organization_id = organization_id or os.getenv("ZOHO_ORGANIZATION_ID")
        self.expense_account_id = expense_account_id or os.getenv("ZOHO_EXPENSE_ACCOUNT_ID")

        self.access_token: Optional[str] = None
        self.token_expiry: float = 0.0

    def refresh_access_token(self) -> str:
        """
        Exchanges refresh token for a new access token via https://accounts.zoho.in/oauth/v2/token.
        """
        if not self.refresh_token or not self.client_id or not self.client_secret:
            raise ValueError("Missing Zoho credentials in environment variables or arguments.")

        params = {
            "refresh_token": self.refresh_token,
            "client_id": self.client_id,
            "client_secret": self.client_secret,
            "grant_type": "refresh_token",
        }

        logger.info(f"Refreshing Zoho OAuth access token at {ACCOUNTS_URL}...")
        res = requests.post(ACCOUNTS_URL, params=params)

        if res.status_code != 200:
            print(f"\n[ZOHO TOKEN ERROR] Status Code: {res.status_code}")
            print(f"[ZOHO TOKEN ERROR Response]: {res.text}\n")
            res.raise_for_status()

        data = res.json()
        if "access_token" not in data:
            print(f"\n[ZOHO TOKEN ERROR] 'access_token' missing in response: {data}\n")
            raise RuntimeError(f"Zoho OAuth failed: {data.get('error', 'Unknown error')}")

        self.access_token = data["access_token"]
        expires_in = data.get("expires_in", 3600)
        self.token_expiry = time.time() + expires_in - 60  # 60s safety buffer
        logger.info("Successfully obtained new Zoho access token.")
        return self.access_token

    def _ensure_access_token(self) -> str:
        """Helper to ensure valid access token exists before API call."""
        if not self.access_token or time.time() >= self.token_expiry:
            return self.refresh_access_token()
        return self.access_token

    def create_expense(
        self,
        vendor: Optional[str],
        date: str,
        amount: float,
        currency: str = "INR",
        description: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Create a new expense entry in Zoho Books.

        Args:
            vendor (str): Vendor name (referenced in description).
            date (str): Date string in YYYY-MM-DD format.
            amount (float): Numerical expense amount.
            currency (str): Currency code (default INR).
            description (str): Optional expense description.

        Returns:
            Dict[str, Any]: Zoho Books API response JSON containing expense details and expense_id.
        """
        token = self._ensure_access_token()
        url = f"{BOOKS_API_BASE}/expenses?organization_id={self.organization_id}"
        headers = {"Authorization": f"Zoho-oauthtoken {token}"}

        desc = description or (f"Bill extraction payment to {vendor}" if vendor else "Bill extraction expense")

        payload = {
            "account_id": self.expense_account_id,
            "date": date,
            "amount": round(float(amount), 2),
            "description": desc,
        }

        logger.info(f"POSTing expense to Zoho Books: {url}")
        
        # Zoho Books v3 API accepts JSON payload or JSONString form parameter
        res = requests.post(url, headers=headers, json=payload)

        # Retry once on 401 Unauthorized (token expired)
        if res.status_code == 401:
            logger.warning("Zoho token expired (401). Refreshing token and retrying request...")
            token = self.refresh_access_token()
            headers["Authorization"] = f"Zoho-oauthtoken {token}"
            res = requests.post(url, headers=headers, json=payload)

        # Surfaced clear error handling
        if res.status_code not in (200, 201):
            print(f"\n[ZOHO API ERROR] HTTP Status Code: {res.status_code}")
            try:
                err_json = res.json()
                print(f"[ZOHO API ERROR JSON]: {json.dumps(err_json, indent=2)}")
            except Exception:
                print(f"[ZOHO API ERROR Response Body]: {res.text}")
            res.raise_for_status()

        res_data = res.json()
        if res_data.get("code") != 0:
            print(f"\n[ZOHO API ERROR CODE {res_data.get('code')}]: {res_data.get('message')}")
            print(f"[FULL RESPONSE]: {json.dumps(res_data, indent=2)}")
            raise RuntimeError(f"Zoho Books API error: {res_data.get('message')}")

        return res_data

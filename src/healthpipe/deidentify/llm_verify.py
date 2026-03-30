"""LLM-based verification pass for residual PHI.

After NER and pattern-matching have done the heavy lifting, this module
sends text through an LLM with a carefully crafted prompt asking it to
identify any remaining Protected Health Information.  This catches
context-dependent identifiers that rule-based systems miss -- e.g. a
doctor's name embedded in a narrative, or a location mentioned only by
nickname.

Supports Anthropic (Claude) and OpenAI-compatible APIs via ``httpx``.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

import httpx

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are a HIPAA compliance auditor. Your task is to identify any remaining
Protected Health Information (PHI) in the following text that may have been
missed by automated de-identification.

PHI includes the 18 HIPAA Safe Harbor identifiers:
1. Names
2. Geographic data smaller than a state
3. Dates (except year) related to an individual
4. Phone numbers
5. Fax numbers
6. Email addresses
7. Social Security Numbers
8. Medical record numbers
9. Health plan beneficiary numbers
10. Account numbers
11. Certificate/license numbers
12. Vehicle identifiers and serial numbers
13. Device identifiers and serial numbers
14. Web URLs
15. IP addresses
16. Biometric identifiers
17. Full-face photographs
18. Any other unique identifying number or code

Respond ONLY with a JSON array of objects. Each object must have:
- "text": the PHI string found
- "category": which of the 18 identifiers it falls under
- "start": character offset where the PHI starts
- "confidence": your confidence (0.0 to 1.0)

If no PHI is found, respond with an empty array: []
"""


@dataclass
class LLMFinding:
    """A single PHI finding from the LLM verification pass."""

    text: str
    category: str
    start: int
    confidence: float


@dataclass
class LLMVerifier:
    """LLM-powered final verification for residual PHI.

    Args:
        model: Model identifier (e.g. ``"claude-haiku-4-5"``).
        api_key: API key for the LLM provider.
        api_url: Base URL for the API.
        provider: ``"anthropic"`` or ``"openai"``.
        timeout: HTTP timeout in seconds.
    """

    model: str = "claude-haiku-4-5"
    api_key: str = ""
    api_url: str = "https://api.anthropic.com"
    provider: str = "anthropic"
    timeout: float = 30.0
    _client: httpx.AsyncClient | None = field(init=False, default=None, repr=False)
    last_status: str = field(init=False, default="idle", repr=False)
    last_error: str | None = field(init=False, default=None, repr=False)

    async def verify(self, text: str) -> list[LLMFinding]:
        """Send *text* to the LLM and parse its PHI findings.

        Returns:
            List of ``LLMFinding`` objects.  Empty if no PHI detected
            or if the LLM call fails (failures are logged, not raised).
        """
        self.last_error = None
        if not self.api_key:
            self.last_status = "skipped"
            logger.warning(
                "LLM verification skipped: no API key configured. "
                "Set api_key to enable the LLM verification layer."
            )
            return []

        try:
            response_text = await self._call_llm(text)
            findings = self._parse_response(response_text)
            self.last_status = "findings" if findings else "clean"
            return findings
        except (httpx.HTTPError, json.JSONDecodeError, KeyError) as exc:
            self.last_status = "error"
            self.last_error = str(exc)
            logger.error("LLM verification failed: %s", exc)
            return []

    async def _call_llm(self, text: str) -> str:
        """Make the API call and return the response text."""
        headers = self._build_headers()
        payload = self._build_payload(text)

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            if self.provider == "anthropic":
                url = f"{self.api_url.rstrip('/')}/v1/messages"
            else:
                url = f"{self.api_url.rstrip('/')}/v1/chat/completions"

            resp = await client.post(url, headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()

        return self._extract_text(data)

    def _build_headers(self) -> dict[str, str]:
        """Build HTTP headers for the LLM API."""
        if self.provider == "anthropic":
            return {
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            }
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def _build_payload(self, text: str) -> dict[str, Any]:
        """Build the request body for the LLM API."""
        if self.provider == "anthropic":
            return {
                "model": self.model,
                "max_tokens": 1024,
                "system": _SYSTEM_PROMPT,
                "messages": [
                    {"role": "user", "content": f"Scan this text for PHI:\n\n{text}"}
                ],
            }
        return {
            "model": self.model,
            "messages": [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": f"Scan this text for PHI:\n\n{text}",
                },
            ],
            "max_tokens": 1024,
        }

    def _extract_text(self, data: dict[str, Any]) -> str:
        """Extract the text content from an API response."""
        if self.provider == "anthropic":
            content = data.get("content", [])
            if isinstance(content, list) and content and isinstance(content[0], dict):
                text_value = content[0].get("text")
                if isinstance(text_value, str):
                    return text_value
            return ""

        choices = data.get("choices", [])
        if (
            isinstance(choices, list)
            and choices
            and isinstance(choices[0], dict)
            and isinstance(choices[0].get("message"), dict)
        ):
            message = choices[0]["message"]
            content_value = message.get("content")
            if isinstance(content_value, str):
                return content_value
        return ""

    @staticmethod
    def _parse_response(response_text: str) -> list[LLMFinding]:
        """Parse the LLM JSON response into LLMFinding objects."""
        # Find the JSON array in the response (LLMs sometimes add prose)
        start = response_text.find("[")
        end = response_text.rfind("]")
        if start == -1 or end == -1:
            return []

        raw = json.loads(response_text[start : end + 1])
        findings: list[LLMFinding] = []
        for item in raw:
            findings.append(
                LLMFinding(
                    text=item.get("text", ""),
                    category=item.get("category", "UNKNOWN"),
                    start=item.get("start", -1),
                    confidence=float(item.get("confidence", 0.5)),
                )
            )
        return findings

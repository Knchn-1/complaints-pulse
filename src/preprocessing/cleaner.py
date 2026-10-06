"""
ComplaintsPulse — Text Preprocessing & PII Redaction

Provides production-grade text sanitization, PII masking, and validation
for incoming consumer complaints before model ingestion.
"""

import re
from typing import Optional, Tuple
import pandas as pd


# Regex patterns for PII and web artifacts
EMAIL_PATTERN = re.compile(
    r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
)
PHONE_PATTERN = re.compile(
    r"(?:\+?\d{1,3}[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?)?\d{3}[-.\s]?\d{4}\b"
)
SSN_PATTERN = re.compile(
    r"\b\d{3}[-.\s]?\d{2}[-.\s]?\d{4}\b"
)
CREDIT_CARD_PATTERN = re.compile(
    r"\b(?:\d{4}[-\s]?){3}\d{4}\b|\b\d{15,16}\b"
)
URL_PATTERN = re.compile(
    r"https?://(?:www\.)?[-a-zA-Z0-9@:%._+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b(?:[-a-zA-Z0-9()@:%_+.~#?&/=]*)"
)
HTML_TAG_PATTERN = re.compile(r"<[^>]+>")

# CFPB specific redaction token patterns: 'XXXX', 'XX/XX/XXXX', 'xxxx xxxx'
CFPB_REDACTION_PATTERN = re.compile(r"\b(?:x{2,}[/-]?)+\b", re.IGNORECASE)

# Repeated punctuation, e.g. "!!!!!" -> "!"
REPEATED_PUNCT_PATTERN = re.compile(r"([!?.,])\1+")

# Whitespace collapsing
WHITESPACE_PATTERN = re.compile(r"\s+")


class ComplaintCleaner:
    """
    Robust text cleaner for financial complaints.
    
    Handles:
    - PII detection and redaction (emails, phones, SSNs, credit cards)
    - CFPB synthetic redaction marker normalization
    - HTML entity / tag removal
    - URL redaction
    - Whitespace and punctuation normalization
    - Configurable lowercasing
    """

    def __init__(
        self,
        redact_pii: bool = True,
        lowercase: bool = True,
        strip_urls: bool = True,
        normalize_cfpb_markers: bool = True,
    ):
        self.redact_pii = redact_pii
        self.lowercase = lowercase
        self.strip_urls = strip_urls
        self.normalize_cfpb_markers = normalize_cfpb_markers

    def clean(self, text: Optional[str]) -> str:
        """
        Sanitize and normalize a single complaint text string.
        Returns a clean string. Returns empty string if input is null or invalid.
        """
        if text is None or not isinstance(text, str):
            return ""

        # Remove HTML tags if present
        text = HTML_TAG_PATTERN.sub(" ", text)

        # URL handling
        if self.strip_urls:
            text = URL_PATTERN.sub(" ", text)

        # PII Redaction
        if self.redact_pii:
            text = EMAIL_PATTERN.sub(" ", text)
            text = SSN_PATTERN.sub(" ", text)
            text = CREDIT_CARD_PATTERN.sub(" ", text)
            text = PHONE_PATTERN.sub(" ", text)

        # CFPB Redaction tokens (XXXX, xx/xx/xxxx)
        if self.normalize_cfpb_markers:
            text = CFPB_REDACTION_PATTERN.sub(" ", text)

        # Normalize repeated punctuation
        text = REPEATED_PUNCT_PATTERN.sub(r"\1 ", text)

        # Lowercase
        if self.lowercase:
            text = text.lower()

        # Normalize whitespace (replaces newlines, tabs, and multi-spaces)
        text = WHITESPACE_PATTERN.sub(" ", text).strip()

        return text

    def validate(self, text: Optional[str], min_words: int = 3) -> Tuple[bool, str]:
        """
        Validate if the input text contains sufficient narrative content for processing.
        
        Returns:
            (is_valid, reason_message)
        """
        if text is None or not isinstance(text, str):
            return False, "Input is null or not a string."
        
        cleaned = self.clean(text)
        if not cleaned:
            return False, "Complaint text is empty or contains only whitespace/PII."

        words = cleaned.split()
        if len(words) < min_words:
            return (
                False,
                f"Complaint text is too short ({len(words)} words; minimum required is {min_words})."
            )

        return True, "Valid complaint text."


# Module-level convenience functions
_default_cleaner = ComplaintCleaner()


def clean_text(text: Optional[str], **kwargs) -> str:
    """Convenience helper using ComplaintCleaner."""
    if kwargs:
        cleaner = ComplaintCleaner(**kwargs)
        return cleaner.clean(text)
    return _default_cleaner.clean(text)


def validate_text(text: Optional[str], min_words: int = 3) -> Tuple[bool, str]:
    """Convenience helper to validate a complaint string."""
    return _default_cleaner.validate(text, min_words=min_words)


def clean_series(series: pd.Series, **kwargs) -> pd.Series:
    """Vectorized / batch cleaning for a pandas Series of complaint texts."""
    cleaner = ComplaintCleaner(**kwargs) if kwargs else _default_cleaner
    return series.fillna("").astype(str).map(cleaner.clean)

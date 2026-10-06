"""
Unit tests for text preprocessing, PII redaction, and input validation.
"""

import pytest
from src.preprocessing.cleaner import ComplaintCleaner, clean_text, validate_text


def test_cleaner_pii_redaction():
    cleaner = ComplaintCleaner()
    
    # Test email
    text_email = "Contact me at john.doe@example.com regarding my credit account."
    cleaned = cleaner.clean(text_email)
    assert "john.doe@example.com" not in cleaned
    assert "credit account" in cleaned

    # Test phone number
    text_phone = "Call me at 123-456-7890 or (800) 555-0199 immediately."
    cleaned = cleaner.clean(text_phone)
    assert "123-456-7890" not in cleaned
    assert "555-0199" not in cleaned
    assert "call me at" in cleaned

    # Test SSN
    text_ssn = "My SSN is 123-45-6789 and it was stolen."
    cleaned = cleaner.clean(text_ssn)
    assert "123-45-6789" not in cleaned
    assert "stolen" in cleaned

    # Test Credit Card number
    text_card = "My card 4111 2222 3333 4444 was charged fraudulently."
    cleaned = cleaner.clean(text_card)
    assert "4111" not in cleaned
    assert "fraudulently" in cleaned


def test_cleaner_cfpb_markers():
    cleaner = ComplaintCleaner()
    text = "On XX/XX/XXXX I visited XXXX branch and spoke with XXXX."
    cleaned = cleaner.clean(text)
    assert "xxxx" not in cleaned
    assert "visited branch and spoke with" in cleaned


def test_cleaner_urls_and_html():
    cleaner = ComplaintCleaner()
    text = "<div>Please see https://bank.example.com/dispute for details</div>"
    cleaned = cleaner.clean(text)
    assert "<div>" not in cleaned
    assert "</div>" not in cleaned
    assert "https://" not in cleaned
    assert "bank.example.com" not in cleaned
    assert "please see for details" in cleaned


def test_cleaner_whitespace_and_punctuation():
    cleaner = ComplaintCleaner()
    text = "Error!!!   Too   many    spaces...  \n\n  New line."
    cleaned = cleaner.clean(text)
    assert cleaned == "error! too many spaces. new line."


def test_cleaner_validation():
    # Empty string
    valid, msg = validate_text("")
    assert not valid
    assert "empty" in msg.lower()

    # None
    valid, msg = validate_text(None)
    assert not valid
    assert "null" in msg.lower()

    # Whitespace only
    valid, msg = validate_text("     ")
    assert not valid

    # Too short
    valid, msg = validate_text("Bad bank", min_words=3)
    assert not valid
    assert "too short" in msg.lower()

    # Valid complaint
    valid, msg = validate_text("My credit card was charged twice without authorization.")
    assert valid
    assert "valid" in msg.lower()

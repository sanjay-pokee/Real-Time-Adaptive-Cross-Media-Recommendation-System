"""Text cleaning and hashing utilities for the content catalog pipeline."""

import ast
import hashlib
import re
from typing import Iterable

import pandas as pd


WHITESPACE_PATTERN = re.compile(r"\s+")


def clean_text(value: object) -> str:
    """Normalize a value to a clean string; returns empty string for null/blank."""
    if pd.isna(value):
        return ""

    text = str(value).strip()
    text = WHITESPACE_PATTERN.sub(" ", text)
    return text


def parse_name_list(value: object) -> list[str]:
    """
    Parse a stringified list (e.g. JSON-like ``[{'name': 'Action'}, ...]``)
    or a plain delimited string into a list of clean name strings.
    """
    text = clean_text(value)
    if not text:
        return []

    try:
        parsed = ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return split_delimited_text(text)

    if isinstance(parsed, list):
        names = []
        for item in parsed:
            if isinstance(item, dict) and "name" in item:
                names.append(clean_text(item["name"]))
            else:
                names.append(clean_text(item))
        return [name for name in names if name]

    return split_delimited_text(text)


def split_delimited_text(value: object) -> list[str]:
    """Split a comma/pipe/semicolon-delimited string into clean parts."""
    text = clean_text(value)
    if not text:
        return []

    parts = re.split(r"[,|;/]+", text)
    return [clean_text(part) for part in parts if clean_text(part)]


def join_non_empty(values: Iterable[object], separator: str = " ") -> str:
    """Join non-blank values with the given separator."""
    cleaned_values = [clean_text(value) for value in values]
    return separator.join(value for value in cleaned_values if value)


def make_text_hash(text: str) -> str:
    """Return the SHA-256 hex digest of the given text (UTF-8 encoded)."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

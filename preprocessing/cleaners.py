import ast
import re
from typing import Iterable

import pandas as pd


WHITESPACE_PATTERN = re.compile(r"\s+")


def clean_text(value: object) -> str:
    if pd.isna(value):
        return ""

    text = str(value).strip()
    text = WHITESPACE_PATTERN.sub(" ", text)
    return text


def parse_name_list(value: object) -> list[str]:
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
    text = clean_text(value)
    if not text:
        return []

    parts = re.split(r"[,|;/]+", text)
    return [clean_text(part) for part in parts if clean_text(part)]


def join_non_empty(values: Iterable[object], separator: str = " ") -> str:
    cleaned_values = [clean_text(value) for value in values]
    return separator.join(value for value in cleaned_values if value)

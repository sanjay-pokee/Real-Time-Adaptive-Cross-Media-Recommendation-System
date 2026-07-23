"""Shared schema definitions for the unified content catalog."""

from dataclasses import dataclass
from typing import Optional


# Ordered list of all columns in content_catalog.csv.
CONTENT_COLUMNS = [
    "global_id",
    "content_type",
    "source",
    "source_id",
    "title",
    "description",
    "creators",
    "categories",
    "release_date",
    "popularity",
    "rating",
    "metadata_text",
    "embedding_text",
    "text_hash",
]

# Columns that must always be present and non-empty for a valid row.
REQUIRED_NON_EMPTY = ["global_id", "title", "embedding_text", "text_hash"]


def make_global_id(content_type: str, source: str, source_id: str) -> str:
    """Build the stable primary key: {content_type}:{source}:{source_id}."""
    return f"{content_type}:{source}:{source_id}"


@dataclass(frozen=True)
class ContentRecord:
    global_id: str
    content_type: str
    source: str
    source_id: str
    title: str
    description: str
    creators: str
    categories: str
    release_date: Optional[str]
    popularity: Optional[float]
    rating: Optional[float]
    metadata_text: str
    embedding_text: str
    text_hash: str

    def to_dict(self) -> dict[str, object]:
        return {
            "global_id": self.global_id,
            "content_type": self.content_type,
            "source": self.source,
            "source_id": self.source_id,
            "title": self.title,
            "description": self.description,
            "creators": self.creators,
            "categories": self.categories,
            "release_date": self.release_date,
            "popularity": self.popularity,
            "rating": self.rating,
            "metadata_text": self.metadata_text,
            "embedding_text": self.embedding_text,
            "text_hash": self.text_hash,
        }

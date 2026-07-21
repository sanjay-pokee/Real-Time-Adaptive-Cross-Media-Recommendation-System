from dataclasses import dataclass
from typing import Optional


CONTENT_COLUMNS = [
    "content_id",
    "content_type",
    "title",
    "description",
    "categories",
    "creators",
    "release_date",
    "popularity",
    "rating",
    "metadata_text",
    "source",
]


@dataclass(frozen=True)
class ContentRecord:
    content_id: str
    content_type: str
    title: str
    description: str
    categories: str
    creators: str
    release_date: Optional[str]
    popularity: Optional[float]
    rating: Optional[float]
    metadata_text: str
    source: str

    def to_dict(self) -> dict[str, object]:
        return {
            "content_id": self.content_id,
            "content_type": self.content_type,
            "title": self.title,
            "description": self.description,
            "categories": self.categories,
            "creators": self.creators,
            "release_date": self.release_date,
            "popularity": self.popularity,
            "rating": self.rating,
            "metadata_text": self.metadata_text,
            "source": self.source,
        }

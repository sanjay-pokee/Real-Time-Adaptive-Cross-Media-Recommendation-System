"""Shared test setup."""

import pytest

from backend.qdrant_recommender import QdrantRecommender


@pytest.fixture(autouse=True)
def no_local_plot_index(monkeypatch):
    """Tests never read the synopsis index built on this machine (embeddings/plot_*).

    A test that wants one sets ``rec._plot_index_cache`` itself.
    """
    monkeypatch.setattr(QdrantRecommender, "_plot_index", lambda self: self.__dict__.get("_plot_index_cache"))

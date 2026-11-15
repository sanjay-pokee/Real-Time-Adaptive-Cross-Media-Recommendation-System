"""MySQLStore shares one connection pool and degrades to direct connections."""

from types import SimpleNamespace

import pytest

from backend import mysql_store
from backend.mysql_store import MySQLStore


class PoolError(Exception):
    pass


class FakePool:
    created = 0

    def __init__(self, capacity=2, **params):
        FakePool.created += 1
        self.free = capacity
        self.params = params

    def get_connection(self):
        if self.free == 0:
            raise PoolError("pool exhausted")
        self.free -= 1
        return "pooled"


def _fake_connector(pool_factory):
    direct = []
    connector = SimpleNamespace(
        pooling=SimpleNamespace(MySQLConnectionPool=pool_factory),
        errors=SimpleNamespace(PoolError=PoolError),
        connect=lambda **params: direct.append(params) or "direct",
    )
    return connector, direct


@pytest.fixture(autouse=True)
def fresh_pools(monkeypatch):
    monkeypatch.setattr(mysql_store, "_POOLS", {})
    FakePool.created = 0


def _store(monkeypatch, connector):
    store = MySQLStore()
    monkeypatch.setattr(store, "_connector", lambda: connector, raising=False)
    return store


def test_every_store_instance_shares_one_pool(monkeypatch):
    connector, direct = _fake_connector(lambda **kw: FakePool(capacity=5, **kw))

    first = _store(monkeypatch, connector)._connect_database()
    second = _store(monkeypatch, connector)._connect_database()

    assert (first, second) == ("pooled", "pooled")
    assert FakePool.created == 1
    assert direct == []


def test_an_exhausted_pool_falls_back_to_a_direct_connection(monkeypatch):
    connector, direct = _fake_connector(lambda **kw: FakePool(capacity=1, **kw))
    store = _store(monkeypatch, connector)

    assert store._connect_database() == "pooled"
    assert store._connect_database() == "direct"
    assert len(direct) == 1


def test_a_pool_that_cannot_be_built_is_retried_next_time(monkeypatch):
    attempts = []

    def flaky_pool(**kw):
        attempts.append(1)
        if len(attempts) == 1:
            raise RuntimeError("database does not exist yet")
        return FakePool(**kw)

    connector, direct = _fake_connector(flaky_pool)
    store = _store(monkeypatch, connector)

    assert store._connect_database() == "direct"
    assert store._connect_database() == "pooled"
    assert len(attempts) == 2

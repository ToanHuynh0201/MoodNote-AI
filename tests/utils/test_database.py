"""Tests for the MongoDB client/collection singleton helpers.

MongoClient() connects lazily, so these tests exercise the singleton/env-var
plumbing without needing a real MongoDB instance running.
"""

import pytest

from src.utils import database


@pytest.fixture(autouse=True)
def _reset_client():
    database.close_client()
    yield
    database.close_client()


def test_get_client_returns_singleton():
    client1 = database.get_client()
    client2 = database.get_client()
    assert client1 is client2


def test_get_training_collection_uses_env_db_name(monkeypatch):
    monkeypatch.setenv("MONGODB_DB_NAME", "custom_db")
    collection = database.get_training_collection()
    assert collection.database.name == "custom_db"
    assert collection.name == "training_samples"


def test_close_client_resets_singleton():
    client1 = database.get_client()
    database.close_client()
    client2 = database.get_client()
    assert client1 is not client2

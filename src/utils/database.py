from pymongo import MongoClient
from pymongo.collection import Collection
import os

_client: MongoClient | None = None


def get_client() -> MongoClient:
    global _client
    if _client is None:
        _client = MongoClient(os.getenv("MONGODB_URL", "mongodb://localhost:27017"))
    return _client


def get_training_collection() -> Collection:
    db_name = os.getenv("MONGODB_DB_NAME", "moodnote_training")
    return get_client()[db_name]["training_samples"]


def close_client():
    global _client
    if _client is not None:
        _client.close()
        _client = None

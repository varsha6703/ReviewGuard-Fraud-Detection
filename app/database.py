
from pymongo import MongoClient

MONGO_URL = "mongodb://localhost:27017/"

client = MongoClient(MONGO_URL)

db = client["ReviewGuardDB"]

reviews_collection = db["reviews"]

def test_connection():
    return client.admin.command("ping")

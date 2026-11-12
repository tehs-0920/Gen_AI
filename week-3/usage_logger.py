
import os
import time
import certifi

from datetime import datetime, timezone
from dotenv import load_dotenv
from pymongo import MongoClient
from pymongo.errors import PyMongoError


class UsageLogger:

    def __init__(self):
        load_dotenv(override=True)

        mongo_uri = os.getenv("MONGODB_URI")

        if not mongo_uri:
            raise ValueError("MONGODB_URI not found in .env")

        self.client = MongoClient(
            mongo_uri,
            tlsCAFile=certifi.where(),
            serverSelectionTimeoutMS=10000,
            connectTimeoutMS=5000,
            socketTimeoutMS=10000,
            retryWrites=True
        )

        # Check that MongoDB is reachable
        self.client.admin.command("ping")

        db = self.client["llm_monitoring"]
        self.collection = db["api_usage"]

        print("MongoDB connected successfully!")

    def log_call(self, record):

        # Work on a copy so the caller's dictionary is not changed
        document = dict(record)

        # Add UTC timestamp
        document["timestamp"] = datetime.now(timezone.utc)

        # Retry a failed database write up to three times
        for attempt in range(3):
            try:
                self.collection.insert_one(document)

                print("Usage details saved to MongoDB.")
                return True

            except PyMongoError as error:

                print(
                    f"MongoDB write attempt {attempt + 1} failed: {error}"
                )

                if attempt < 2:
                    time.sleep(2 ** attempt)

        print("Warning: usage record could not be saved.")
        return False

    def close(self):
        self.client.close()
        print("MongoDB connection closed.")

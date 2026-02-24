# scripts/reset_pinecone.py
import os
import time
from dotenv import load_dotenv
from pinecone import Pinecone

load_dotenv()

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index_name = "documind-index"

# Delete old index
try:
    pc.delete_index(index_name)
    print("✅ Old index deleted")
    time.sleep(5)
except:
    print("ℹ️  No existing index")

# Create new index (3072 dimensions)
pc.create_index(
    name=index_name,
    dimension=3072,
    metric="cosine",
    spec={"serverless": {"cloud": "aws", "region": "us-east-1"}}
)
print("✅ New index created (3072 dimensions)")
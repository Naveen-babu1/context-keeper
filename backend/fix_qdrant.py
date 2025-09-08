# Save as final_fix.py
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer
import requests
import uuid

print("🔧 Final Qdrant Fix...")

# Initialize
client = QdrantClient(host="localhost", port=6333)
model = SentenceTransformer('all-MiniLM-L6-v2')

# Create collection properly
collection_name = "context_events"

try:
    # Delete if exists
    client.delete_collection(collection_name)
except:
    pass

# Create with proper settings
client.create_collection(
    collection_name=collection_name,
    vectors_config=VectorParams(
        size=384,
        distance=Distance.COSINE
    )
)
print("✅ Collection created")

# Get events from API
response = requests.get("http://localhost:8000/api/timeline?limit=100")
events = response.json()['timeline']['events']
print(f"📦 Processing {len(events)} events...")

# Batch process
batch_size = 10
points = []

for i, event in enumerate(events):
    # Create embedding
    text = f"{event.get('title', '')} {event.get('message', '')}"
    vector = model.encode(text).tolist()
    
    # Create point with unique ID
    point = PointStruct(
        id=str(uuid.uuid4()),  # Use UUID instead of integer
        vector=vector,
        payload={
            "message": event.get('title', ''),
            "type": event.get('type', 'git_commit'),
            "timestamp": event.get('timestamp', ''),
            "idx": i
        }
    )
    points.append(point)
    
    # Upload in batches
    if len(points) >= batch_size:
        client.upsert(collection_name=collection_name, points=points)
        points = []
        print(f"✓ Uploaded {i+1}/{len(events)}")

# Upload remaining
if points:
    client.upsert(collection_name=collection_name, points=points)

# Verify
info = client.get_collection(collection_name)
print(f"\n✅ Success! {info.vectors_count} vectors stored")

# Test search
query = "version changes"
query_vector = model.encode(query).tolist()
results = client.search(
    collection_name=collection_name,
    query_vector=query_vector,
    limit=5
)

print(f"\n🔍 Search for '{query}':")
for r in results:
    print(f"  - Score: {r.score:.3f} | {r.payload['message'][:50]}")

print("\n✅ Qdrant is fully operational!")
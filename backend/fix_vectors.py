"""Fix vector storage in Qdrant"""
import requests
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer

print("🔧 Fixing vector storage...")

# Initialize
client = QdrantClient(host="localhost", port=6333)
model = SentenceTransformer('all-MiniLM-L6-v2')

# Check API
try:
    resp = requests.get("http://localhost:8000/api/stats")
    stats = resp.json()
    print(f"✅ API running: {stats['events']['in_memory']} events in memory")
except:
    print("❌ API not running! Start it first.")
    exit(1)

# Setup collection
try:
    client.delete_collection("context_events")
except:
    pass

client.create_collection(
    collection_name="context_events",
    vectors_config=VectorParams(size=384, distance=Distance.COSINE)
)
print("✅ Collection created")

# Get events
response = requests.get("http://localhost:8000/api/timeline?limit=200")
events = response.json()['timeline']['events']
print(f"📦 Processing {len(events)} events...")

# Create vectors
points = []
for i, event in enumerate(events):
    text = f"{event.get('title', '')} {event.get('message', '')}"
    vector = model.encode(text).tolist()
    
    points.append(PointStruct(
        id=i,
        vector=vector,
        payload={
            "message": event.get('title', event.get('message', '')),
            "type": event.get('type', 'git_commit'),
            "timestamp": event.get('timestamp', ''),
            "author": event.get('author', '')
        }
    ))
    
    if (i + 1) % 20 == 0:
        print(f"  Processed {i + 1}/{len(events)}")

# Upload
client.upsert(collection_name="context_events", points=points)

# Verify
info = client.get_collection("context_events")
print(f"✅ Stored {info.vectors_count} vectors in Qdrant!")

# Test search
test_queries = ["version 1.2", "LanguageService", "refactor"]
for query in test_queries:
    vector = model.encode(query).tolist()
    results = client.search("context_events", vector, limit=2)
    print(f"\n🔍 '{query}':")
    for r in results:
        msg = r.payload.get('message', '')[:60]
        print(f"  - {r.score:.3f}: {msg}")

print("\n✅ Vector storage fixed! Context Keeper is fully operational!")

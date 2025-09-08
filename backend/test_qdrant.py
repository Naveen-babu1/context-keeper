# Save as test_qdrant.py
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

try:
    # Connect
    client = QdrantClient(host="localhost", port=6333)
    print("✅ Connected to Qdrant")
    
    # List collections
    collections = client.get_collections()
    print(f"Collections: {collections}")
    
    # Create test collection
    client.recreate_collection(
        collection_name="test",
        vectors_config=VectorParams(size=4, distance=Distance.DOT)
    )
    print("✅ Created test collection")
    
    # Delete test
    client.delete_collection("test")
    print("✅ Qdrant is working properly!")
    
except Exception as e:
    print(f"❌ Error: {e}")
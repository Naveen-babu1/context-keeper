# Save this as backend/app/main.py (replacing the previous version)
"""
Context Keeper API Server - Full Version with Docker Services
"""
from contextlib import asynccontextmanager
import logging
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
import uvicorn
import os
from pathlib import Path

# Vector database imports
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer

# For LLM
import httpx

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Global services
qdrant_client = None
embedding_model = None
events_collection = "context_events"

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize services on startup"""
    global qdrant_client, embedding_model
    
    logger.info("🚀 Starting Context Keeper with Docker services...")
    
    # Initialize Qdrant
    try:
        qdrant_client = QdrantClient(host="localhost", port=6333)
        logger.info("✅ Connected to Qdrant")
        
        # Create collection if it doesn't exist
        collections = qdrant_client.get_collections().collections
        if not any(c.name == events_collection for c in collections):
            qdrant_client.create_collection(
                collection_name=events_collection,
                vectors_config=VectorParams(size=384, distance=Distance.COSINE)
            )
            logger.info(f"✅ Created collection: {events_collection}")
    except Exception as e:
        logger.warning(f"⚠️ Qdrant not available: {e}")
        logger.info("Running in degraded mode without vector search")
    
    # Initialize embedding model
    try:
        embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        logger.info("✅ Loaded embedding model")
    except Exception as e:
        logger.warning(f"⚠️ Could not load embedding model: {e}")
    
    # Create data directory
    Path("./data").mkdir(exist_ok=True)
    
    logger.info("✅ Context Keeper is ready!")
    
    yield
    
    logger.info("👋 Shutting down Context Keeper...")

# Create FastAPI app
app = FastAPI(
    title="Context Keeper",
    description="AI Memory Layer for Development Teams",
    version="0.2.0",
    lifespan=lifespan,
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============= Models =============
class QueryRequest(BaseModel):
    query: str
    limit: Optional[int] = 10
    include_graph: Optional[bool] = False

class QueryResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]]
    confidence: float
    
# In-memory storage (backup for when Docker is down)
git_events = []
context_events = []
next_id = 1

# ============= Helper Functions =============
def create_embedding(text: str) -> List[float]:
    """Create embedding for text"""
    if embedding_model:
        try:
            return embedding_model.encode(text).tolist()
        except:
            pass
    # Return dummy embedding if model not available
    return [0.0] * 384

async def store_in_qdrant(event: Dict[str, Any], event_id: str):
    """Store event in Qdrant vector database"""
    if not qdrant_client or not embedding_model:
        return False
    
    try:
        # Create text representation for embedding
        text = f"{event.get('message', '')} {event.get('description', '')} {' '.join(event.get('files_changed', []))}"
        embedding = create_embedding(text)
        
        # Store in Qdrant
        qdrant_client.upsert(
            collection_name=events_collection,
            points=[
                PointStruct(
                    id=event_id,
                    vector=embedding,
                    payload=event
                )
            ]
        )
        return True
    except Exception as e:
        logger.error(f"Failed to store in Qdrant: {e}")
        return False

async def search_similar(query: str, limit: int = 10) -> List[Dict[str, Any]]:
    if not qdrant_client or not embedding_model:
        return []
    
    try:
        query_embedding = create_embedding(query)
        results = qdrant_client.search(
            collection_name=events_collection,
            query_vector=query_embedding,
            limit=limit
        )
        
        # Remove duplicates based on message content
        seen = set()
        unique_results = []
        for hit in results:
            msg = hit.payload.get('message', '')
            if msg not in seen:
                seen.add(msg)
                unique_results.append(hit.payload)
        
        return unique_results
    except Exception as e:
        logger.error(f"Search failed: {e}")
        return []

async def query_llm(prompt: str) -> str:
    """Query local LLM using Ollama"""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": "mistral:7b-instruct",
                    "prompt": prompt,
                    "stream": False
                },
                timeout=30.0
            )
            if response.status_code == 200:
                return response.json().get("response", "")
    except Exception as e:
        logger.warning(f"LLM query failed: {e}")
    return ""

# ============= Endpoints =============
@app.get("/")
async def root():
    return {
        "message": "Context Keeper API",
        "version": "0.2.0",
        "status": "running",
        "docs": "/docs",
        "services": {
            "qdrant": qdrant_client is not None,
            "embeddings": embedding_model is not None
        }
    }

@app.get("/health")
async def health_check():
    services = {}
    
    # Check Qdrant
    try:
        if qdrant_client:
            qdrant_client.get_collections()
            services["qdrant"] = "healthy"
    except:
        services["qdrant"] = "unavailable"
    
    # Check Ollama
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get("http://localhost:11434/api/tags", timeout=2.0)
            services["ollama"] = "healthy" if response.status_code == 200 else "unavailable"
    except:
        services["ollama"] = "unavailable"
    
    services["embeddings"] = "healthy" if embedding_model else "unavailable"
    
    return {
        "status": "healthy" if any(s == "healthy" for s in services.values()) else "degraded",
        "services": services,
        "events_stored": len(git_events) + len(context_events)
    }

@app.post("/api/ingest/git")
async def ingest_git_event(event: Dict[str, Any], background_tasks: BackgroundTasks):
    """Ingest git event and store in vector database"""
    global next_id
    
    try:
        # Add metadata
        event["timestamp"] = event.get("timestamp", datetime.now().isoformat())
        event["type"] = "git_commit"
        event_id = str(next_id)
        next_id += 1
        
        # Store in memory
        git_events.append(event)
        
        # Store in Qdrant (async)
        if qdrant_client:
            background_tasks.add_task(store_in_qdrant, event, event_id)
        
        logger.info(f"📥 Ingested git event: {event.get('commit_hash', '')[:8]}")
        
        return {
            "status": "success",
            "message": "Git event ingested",
            "event_id": event_id,
            "commit": event.get("commit_hash", "")[:8],
            "vector_storage": qdrant_client is not None
        }
    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/query", response_model=QueryResponse)
async def query_context(request: QueryRequest):
    """Query context using vector search and LLM"""
    try:
        # Search for similar events
        similar_events = await search_similar(request.query, request.limit)
        
        # Fallback to simple search if vector search unavailable
        if not similar_events:
            for event in git_events[-20:]:
                if request.query.lower() in str(event).lower():
                    similar_events.append(event)
        
        # Generate answer using LLM if available
        answer = ""
        if similar_events and await query_llm("test"):
            context = "\n".join([
                f"- {e.get('message', e.get('description', ''))}" 
                for e in similar_events[:5]
            ])
            
            prompt = f"""Based on the following context, answer this question: {request.query}

Context:
{context}

Answer:"""
            
            llm_response = await query_llm(prompt)
            if llm_response:
                answer = llm_response
            else:
                answer = f"Found {len(similar_events)} relevant events for: {request.query}"
        else:
            answer = f"Found {len(similar_events)} relevant events" if similar_events else "No relevant events found"
        
        # Format sources
        sources = []
        for event in similar_events[:request.limit]:
            sources.append({
                "type": event.get("type", "unknown"),
                "content": event.get("message", event.get("description", "")),
                "timestamp": event.get("timestamp", ""),
                "author": event.get("author", ""),
                "commit": event.get("commit_hash", "")[:8] if "commit_hash" in event else ""
            })
        
        return QueryResponse(
            answer=answer,
            sources=sources,
            confidence=0.85 if similar_events else 0.25
        )
        
    except Exception as e:
        logger.error(f"Query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/timeline")
async def get_timeline(days: int = 7, limit: int = 50):
    """Get timeline of all events"""
    all_events = []
    
    # Get events from Qdrant if available
    if qdrant_client:
        try:
            # Scroll through all points
            result = qdrant_client.scroll(
                collection_name=events_collection,
                limit=limit
            )
            for point in result[0]:
                event = point.payload
                all_events.append({
                    "id": str(point.id),
                    "timestamp": event.get("timestamp", ""),
                    "type": event.get("type", ""),
                    "title": event.get("message", "")[:100] if "message" in event else event.get("title", ""),
                    "author": event.get("author", ""),
                })
        except Exception as e:
            logger.error(f"Failed to get timeline from Qdrant: {e}")
    
    # Add from memory if Qdrant failed
    if not all_events:
        for event in (git_events + context_events)[-limit:]:
            all_events.append({
                "id": f"mem_{len(all_events)}",
                "timestamp": event.get("timestamp", ""),
                "type": event.get("type", ""),
                "title": event.get("message", "")[:100] if "message" in event else event.get("title", ""),
                "author": event.get("author", ""),
            })
    
    return {
        "timeline": {
            "events": all_events,
            "total": len(all_events),
            "source": "qdrant" if qdrant_client else "memory"
        }
    }

@app.get("/api/stats")
async def get_statistics():
    stats = {
        "events": {
            "in_memory": len(git_events) + len(context_events),
            "in_qdrant": 0
        }
    }
    
    if qdrant_client:
        try:
            collection_info = qdrant_client.get_collection(events_collection)
            # Use points_count instead of vectors_count
            stats["events"]["in_qdrant"] = collection_info.points_count
        except:
            pass
    
    return stats

if __name__ == "__main__":
    print("\n" + "="*60)
    print("🚀 Context Keeper - Full Version")
    print("="*60)
    print("\n📦 Using Docker services:")
    print("  - Qdrant (Vector DB): http://localhost:6333")
    print("  - Neo4j (Graph DB): http://localhost:7474")
    print("  - Ollama (LLM): http://localhost:11434")
    print("  - Redis (Cache): http://localhost:6379")
    print("\n" + "="*60 + "\n")
    
    # FIX: Change this line
    uvicorn.run(
        app,  # Use app directly, not "app.main:app"
        host="0.0.0.0",
        port=8000,
        log_level="info"
        # Remove reload=True
    )
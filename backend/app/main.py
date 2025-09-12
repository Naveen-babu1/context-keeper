# Save this as backend/app/main.py (replacing the previous version)
"""
Context Keeper API Server - Full Version with Docker Services
"""
from contextlib import asynccontextmanager
import logging
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
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

ingestion_status = {
    "is_ingesting": False,
    "current_repo": None,
    "progress": 0,
    "total": 0,
    "last_ingested": None
}

# Global services
qdrant_client = None
embedding_model = None
events_collection = "context_events"
indexed_repos = set()

static_dir = os.path.join(os.path.dirname(__file__), "static")
os.makedirs(static_dir, exist_ok=True)

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
        
        # Store in Qdrant - use UUID string or convert to proper format
        import uuid
        
        # Generate a UUID if the event_id is just a number
        if event_id.isdigit():
            point_id = str(uuid.uuid4())
        else:
            point_id = event_id
            
        qdrant_client.upsert(
            collection_name=events_collection,
            points=[
                PointStruct(
                    id=point_id,  # Use UUID string instead of integer
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
    
async def generate_intelligent_answer(query: str, events: List[Dict]) -> str:
    """Generate intelligent summary of events"""
    if not events:
        return "No relevant events found."
    
    query_lower = query.lower()
    
    # Special handling for list/show all queries
    if any(word in query_lower for word in ["all", "list all", "show all", "every"]):
        commit_list = []
        for e in events[:20]:  # Show first 20
            commit_hash = e.get('commit_hash', '')
            msg = e.get('message', 'No message')
            author = e.get('author', 'Unknown')
            commit_str = f"• {msg} [{commit_hash[:8] if commit_hash else 'no-hash'}] by {author}"
            commit_list.append(commit_str)
        
        return f"Found {len(events)} commits. Here are the most relevant:\n\n" + "\n".join(commit_list)
    
    # Create context from events
    context = "\n".join([
        f"- {e.get('message', '')} (commit: {e.get('commit_hash', '')[:8] if e.get('commit_hash') else 'no-hash'})"
        for e in events[:10]
    ])
    
    # Try to use LLM if available
    if await query_llm("test"):
        prompt = f"""Based on these commits, provide a helpful summary for: {query}
        
Commits:
{context}

Summary:"""
        response = await query_llm(prompt)
        if response:
            return response
    
    # Intelligent fallback based on query type
    if "version" in query_lower:
        versions = [e for e in events if 'version' in e.get('message', '').lower()]
        if versions:
            version_list = [v.get('message', '') for v in versions[:5]]
            return f"Found {len(versions)} version updates:\n" + "\n".join(f"• {v}" for v in version_list)
    
    elif "fix" in query_lower or "bug" in query_lower:
        fixes = [e for e in events if any(word in e.get('message', '').lower() for word in ['fix', 'bug', 'patch'])]
        if fixes:
            return f"Found {len(fixes)} bug fixes/patches:\n" + "\n".join(f"• {f.get('message', '')}" for f in fixes[:5])
    
    elif "feature" in query_lower:
        features = [e for e in events if any(word in e.get('message', '').lower() for word in ['feat', 'add', 'implement'])]
        if features:
            return f"Found {len(features)} feature additions:\n" + "\n".join(f"• {f.get('message', '')}" for f in features[:5])
    
    # Default response with more detail
    return f"Found {len(events)} relevant commits for '{query}':\n" + "\n".join(
        f"• {e.get('message', '')} by {e.get('author', 'Unknown')}" 
        for e in events[:5]
    )

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
async def serve_ui():
    html_path = os.path.join(static_dir, "index.html")
    if os.path.exists(html_path):
        return FileResponse(html_path)
    else:
        # Fallback if HTML doesn't exist
        return {"message": "Context Keeper API", "docs": "/docs"}

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

    commit_hash = event.get("commit_hash", "")
    if commit_hash:
        # Check if this commit already exists
        for existing in git_events:
            if existing.get("commit_hash") == commit_hash:
                logger.info(f"⚠️ Duplicate commit skipped: {commit_hash[:8]}")
                return {
                    "status": "duplicate",
                    "message": "Commit already indexed",
                    "commit": commit_hash[:8] if commit_hash else ""
                }
    
    try:
        # Add metadata
        event["commit_hash"] = event.get("commit_hash", "")
        event["timestamp"] = event.get("timestamp", datetime.now().isoformat())
        event["type"] = "git_commit"
        
        # Use UUID instead of integer
        import uuid
        event_id = str(uuid.uuid4())
        
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
        query_lower = request.query.lower()
        if any(word in query_lower for word in ["all", "list all", "show all", "every"]):
            request.limit = 100  # Show more when user asks for all
        # Search for similar events
        similar_events = await search_similar(request.query, request.limit)
        
        # Generate answer using LLM if available
        answer = await generate_intelligent_answer(request.query, similar_events)
#         if similar_events and await query_llm("test"):
#             context = "\n".join([
#                 f"- {e.get('message', e.get('description', ''))}" 
#                 for e in similar_events[:5]
#             ])
            
#             prompt = f"""Based on the following context, answer this question: {request.query}

# Context:
# {context}

# Answer:"""
            
#             llm_response = await query_llm(prompt)
#             if llm_response:
#                 answer = llm_response
#             else:
#                 answer = f"Found {len(similar_events)} relevant events for: {request.query}"
#         else:
#             answer = f"Found {len(similar_events)} relevant events" if similar_events else "No relevant events found"
        
        # Format sources
        sources = []
        for event in similar_events[:request.limit]:
            sources.append({
                "type": event.get("type", "unknown"),
                "content": event.get("message", event.get("description", "")),
                "timestamp": event.get("timestamp", ""),
                "author": event.get("author", ""),
                "commit": event.get("commit_hash", "")[:8]
            })
        
        return QueryResponse(
            answer=answer,
            sources=sources,
            confidence=0.85 if similar_events else 0.25
        )
        
    except Exception as e:
        logger.error(f"Query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Add these endpoints to your main.py

@app.delete("/api/clear")
async def clear_all_data():
    """Clear all data (use with caution!)"""
    global git_events, context_events, indexed_repos
    
    # Clear memory
    git_events.clear()
    context_events.clear()
    indexed_repos.clear()
    
    # Clear Qdrant
    if qdrant_client:
        try:
            qdrant_client.delete_collection(events_collection)
            qdrant_client.create_collection(
                collection_name=events_collection,
                vectors_config=VectorParams(size=384, distance=Distance.COSINE)
            )
            logger.info("✅ Cleared all data from Qdrant")
            return {"status": "success", "message": "All data cleared"}
        except Exception as e:
            logger.error(f"Failed to clear Qdrant: {e}")
            return {"status": "error", "message": str(e)}
    
    return {"status": "success", "message": "Memory cleared (no Qdrant)"}

@app.post("/api/cleanup")
async def cleanup_bad_entries():
    """Remove entries without commit hashes"""
    global git_events
    
    removed_count = 0
    
    # Clean memory
    git_events = [e for e in git_events if e.get("commit_hash")]
    
    # Clean Qdrant
    if qdrant_client:
        try:
            # Get all points
            results = qdrant_client.scroll(
                collection_name=events_collection,
                limit=1000,
                with_payload=True,
                with_vectors=False
            )[0]
            
            # Find points without commit_hash
            points_to_delete = []
            for point in results:
                if not point.payload.get("commit_hash"):
                    points_to_delete.append(point.id)
                    removed_count += 1
            
            # Delete bad points
            if points_to_delete:
                qdrant_client.delete(
                    collection_name=events_collection,
                    points_selector=points_to_delete
                )
                logger.info(f"✅ Removed {len(points_to_delete)} entries without commit hashes")
            
        except Exception as e:
            logger.error(f"Cleanup failed: {e}")
            return {"status": "error", "message": str(e)}
    
    return {
        "status": "success",
        "removed": removed_count,
        "message": f"Removed {removed_count} entries without commit hashes"
    }

@app.get("/api/debug/commits")
async def debug_commits():
    """Debug endpoint to see what's in the database"""
    debug_info = {
        "in_memory": [],
        "in_qdrant": []
    }
    
    # Check memory
    for event in git_events[:10]:  # First 10
        debug_info["in_memory"].append({
            "commit_hash": event.get("commit_hash", "MISSING"),
            "message": event.get("message", "")[:50]
        })
    
    # Check Qdrant
    if qdrant_client:
        try:
            results = qdrant_client.scroll(
                collection_name=events_collection,
                limit=10,
                with_payload=True,
                with_vectors=False
            )[0]
            
            for point in results:
                debug_info["in_qdrant"].append({
                    "id": str(point.id),
                    "commit_hash": point.payload.get("commit_hash", "MISSING"),
                    "message": point.payload.get("message", "")[:50]
                })
        except Exception as e:
            debug_info["qdrant_error"] = str(e)
    
    return debug_info

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
                limit=1000,
                with_payload=True,
                with_vector=False
            )[0]
            for point in result:  # Fix: result is already the list
                event = point.payload
                commit_hash = event.get("commit_hash", "")
                all_events.append({
                    "id": str(point.id),
                    "timestamp": event.get("timestamp", ""),
                    "type": event.get("type", ""),
                    "title": event.get("message", "")[:100] if "message" in event else event.get("title", ""),
                    "author": event.get("author", ""),
                    "commit": commit_hash[:8] if commit_hash else ""
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

@app.post("/api/ingest/start")
async def start_ingestion(request: dict):
    """Start ingesting a repository"""
    repo_path = request.get("repo_path")
    # max_commits = request.get("max_commits", 100)
    
    if not repo_path:
        raise HTTPException(status_code=400, detail="repo_path required")
    
    if repo_path in indexed_repos:
        return {"status": "already_indexed", "message": f"Repository {repo_path} is already indexed"}
    
    # Update status
    ingestion_status["is_ingesting"] = True
    ingestion_status["current_repo"] = repo_path
    ingestion_status["progress"] = 0
    ingestion_status["total"] = 999999
    
    # Run git collector in background
    import subprocess
    import threading
    import os
    
    def run_collector():
        try:
            # Fix the path to git_collector.py
            collector_path = os.path.join(
                os.path.dirname(os.path.dirname(__file__)),  # Go up from app to backend
                "collectors", "git", "git_collector.py"
            )
            
            # If collector doesn't exist in expected location, try alternative
            if not os.path.exists(collector_path):
                collector_path = "D:/projects/context-keeper/collectors/git/git_collector.py"
            
            subprocess.run([
                "python", collector_path,
                "--repo", repo_path,
                "--history", "999999",
                "--api-url", "http://localhost:8000"
            ])
            
            ingestion_status["is_ingesting"] = False
            ingestion_status["last_ingested"] = datetime.now().isoformat()
        except Exception as e:
            ingestion_status["is_ingesting"] = False
            ingestion_status["error"] = str(e)
            logger.error(f"Ingestion error: {e}")
    
    thread = threading.Thread(target=run_collector)
    thread.start()
    indexed_repos.add(repo_path)
    
    return {"status": "started", "repo": repo_path}

@app.get("/api/ingest/status")
async def get_ingestion_status():
    """Get current ingestion status"""
    return ingestion_status

@app.get("/api/repos")
async def get_indexed_repos():
    """Get list of indexed repositories"""
    repos = {}
    for event in git_events:
        repo = event.get("branch", "unknown")
        if repo not in repos:
            repos[repo] = {
                "commit_count": 0,
                "last_commit": None,
                "authors": set()
            }
        repos[repo]["commit_count"] += 1
        repos[repo]["last_commit"] = event.get("timestamp")
        repos[repo]["authors"].add(event.get("author", ""))
    
    # Convert sets to lists for JSON serialization
    for repo in repos:
        repos[repo]["authors"] = list(repos[repo]["authors"])
    
    return repos

@app.get("/api/info")  # Move the API info to a different endpoint
async def api_info():
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
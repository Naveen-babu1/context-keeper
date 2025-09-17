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
import json
from pathlib import Path


# Vector database imports
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer

from datetime import datetime
from collections import defaultdict

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
DATA_DIR = Path("./data")
DATA_DIR.mkdir(exist_ok=True)
REPOS_FILE = DATA_DIR / "repositories.json"
REPOS_FILE = Path("./data/repositories.json")
indexed_repositories = {} 


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
    load_repositories()
    # Initialize Qdrant
    try:
        qdrant_client = QdrantClient(host="localhost", port=6333)
        logger.info("✅ Connected to Qdrant")
        
        # Create collection if needed
        collections = qdrant_client.get_collections().collections
        if not any(c.name == events_collection for c in collections):
            qdrant_client.create_collection(
                collection_name=events_collection,
                vectors_config=VectorParams(size=384, distance=Distance.COSINE)
            )
            logger.info(f"✅ Created collection: {events_collection}")
        else:
            # Load existing events from Qdrant into memory
            try:
                results = qdrant_client.scroll(
                    collection_name=events_collection,
                    limit=1000,
                    with_payload=True,
                    with_vectors=False
                )[0]
                
                for point in results:
                    git_events.append(point.payload)
                
                logger.info(f"✅ Loaded {len(results)} existing events from Qdrant")
            except Exception as e:
                logger.error(f"Failed to load existing events: {e}")
                
    except Exception as e:
        logger.warning(f"⚠️ Qdrant not available: {e}")
    
    # Initialize embedding model
    try:
        embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        logger.info("✅ Loaded embedding model")
    except Exception as e:
        logger.warning(f"⚠️ Could not load embedding model: {e}")
    
    # Create data directory
    # Path("./data").mkdir(exist_ok=True)
    
    # logger.info("✅ Context Keeper is ready!")
    
    yield
    save_repositories()
    
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
    repository: Optional[str] = None 
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

def create_code_embedding(event: Dict[str, Any]) -> List[float]:
    """Create embedding that captures code context"""
    if not embedding_model:
        return [0.0] * 384
    
    # Combine different aspects of the commit
    text_parts = []
    
    # Commit message (intent)
    text_parts.append(f"Message: {event.get('message', '')}")
    
    # Files changed (scope)
    files = event.get('files_changed', [])
    if files:
        text_parts.append(f"Files: {', '.join(files[:10])}")
    
    # Code diffs (actual changes)
    diffs = event.get('code_diffs', [])
    for diff in diffs[:3]:  # First 3 diffs
        if 'diff' in diff:
            # Extract key lines from diff
            lines = diff['diff'].split('\n')
            important_lines = [l for l in lines if l.startswith('+') or l.startswith('-')][:10]
            text_parts.append(f"Changes in {diff['file']}: {' '.join(important_lines)}")
    
    # Combine and create embedding
    full_text = " ".join(text_parts)
    return embedding_model.encode(full_text).tolist()

def save_repositories():
    """Save repository list to file"""
    try:
        with open(REPOS_FILE, 'w') as f:
            json.dump(indexed_repositories, f, indent=2)
        logger.info(f"Saved {len(indexed_repositories)} repositories")
    except Exception as e:
        logger.error(f"Failed to save repositories: {e}")

def normalize_repo_path(path: str) -> str:
    """Normalize repository path to prevent duplicates"""
    # Convert to absolute path and normalize
    normalized = str(Path(path).resolve())
    # Remove trailing slashes
    normalized = normalized.rstrip('/\\')
    # Convert to lowercase for Windows compatibility
    if os.name == 'nt':  # Windows
        normalized = normalized.lower()
    return normalized

def load_repositories():
    """Load repository list from file"""
    global indexed_repositories
    if REPOS_FILE.exists():
        try:
            with open(REPOS_FILE, 'r') as f:
                indexed_repositories = json.load(f)
                logger.info(f"✅ Loaded {len(indexed_repositories)} repositories from storage")
        except Exception as e:
            logger.error(f"Failed to load repositories: {e}")
            indexed_repositories = {}
    else:
        indexed_repositories = {}

async def store_in_qdrant(event: Dict[str, Any], event_id: str):
    """Store event in Qdrant vector database"""
    if not qdrant_client or not embedding_model:
        return False
    
    try:
        # Create text representation for embedding
        text = f"{event.get('message', '')} {event.get('description', '')} {' '.join(event.get('files_changed', []))}"
        embedding = embedding_model.encode(text).tolist()
        
        # Store in Qdrant - use UUID string or convert to proper format
        import uuid
        point_id = str(uuid.uuid4())
        
            
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
        logger.info(f"Stored in Qdrant: {event.get('commit_hash', '')[:8]}")
        return True
    except Exception as e:
        logger.error(f"Failed to store in Qdrant: {e}")
        return False

async def search_similar(query: str, limit: int = 10, repository: str = None) -> List[Dict[str, Any]]:
    if not qdrant_client or not embedding_model:
        return []
    
    try:
        query_embedding = create_embedding(query)
        # Build filter for repository if specified
        search_filter = None
        if repository:
            normalized_repo = str(Path(repository).resolve())
            search_filter = {
                "must": [
                    {
                        "key": "repository",
                        "match": {"value": repository}
                    }
                ]
            }
        results = qdrant_client.search(
            collection_name=events_collection,
            query_vector=query_embedding,
            query_filter=None,
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
    """Enhanced intelligent answer generation with better commit handling"""
    if not events:
        return "No relevant events found."
    
    query_lower = query.lower()
    
    # Classify the query type for better handling
    query_type = classify_query_type(query_lower)
    
    # Handle specific query types
    if query_type == "repository_summary":
        return await generate_enhanced_repository_summary(events, query)
    elif query_type == "list_commits":
        return await list_all_commits(events, query)
    elif query_type == "specific_commit":
        return await analyze_specific_commit(events, query)
    elif query_type == "commit_comparison":
        return await compare_commits(events, query)
    elif query_type == "timeline":
        return await generate_detailed_timeline(events, query)
    elif query_type == "author_analysis":
        return await analyze_author_activity(events, query)
    
    # Try LLM for complex queries
    llm_response = await query_enhanced_llm(query, events)
    if llm_response:
        return llm_response
    
    # Fallback to pattern matching
    return await generate_fallback_response(query, events)

def classify_query_type(query_lower: str) -> str:
    """Enhanced query classification"""
    
    # Repository summary queries
    if any(word in query_lower for word in ["summarize", "summary", "overview", "describe repo", "tell me about", "what is this repo"]):
        return "repository_summary"
    
    # List all commits queries
    elif any(phrase in query_lower for phrase in ["show all commits", "list commits", "all the commits", "every commit", "commit list"]):
        return "list_commits"
    
    # Specific commit queries
    elif any(word in query_lower for word in ["commit", "hash"]) and any(word in query_lower for word in ["show", "explain", "what", "analyze", "details"]):
        return "specific_commit"
    
    # Commit comparison queries
    elif any(phrase in query_lower for phrase in ["compare", "difference", "changed from", "vs", "versus"]):
        return "commit_comparison"
    
    # Timeline queries
    elif any(word in query_lower for word in ["timeline", "history", "chronological", "when", "evolution"]):
        return "timeline"
    
    # Author analysis
    elif any(word in query_lower for word in ["who", "author", "contributor", "developer"]):
        return "author_analysis"
    
    return "general"

async def generate_enhanced_repository_summary(events: List[Dict], query: str) -> str:
    """Enhanced repository summary with comprehensive analysis"""
    if not events:
        return "No repository data available for summary."
    
    # Extract repository info
    repo_path = events[0].get("repository", "unknown")
    repo_name = repo_path.split('\\')[-1] if '\\' in repo_path else repo_path.split('/')[-1]
    
    # Comprehensive statistics
    stats = analyze_comprehensive_stats(events)
    
    # Build enhanced summary
    summary = f"""# {repo_name} - Repository Analysis

## 📊 Overview
**Total Commits:** {stats['total_commits']}
**Active Contributors:** {stats['contributors']}
**Files Modified:** {stats['files_touched']}
**Development Period:** {stats['date_range']}
**Primary Language:** {stats['primary_language']}

## 🏗️ Development Activity
{format_commit_breakdown(stats['commit_breakdown'])}

## 🔥 Recent Activity ({len(events[:10])} latest commits)
{format_recent_commits(events[:10])}

## 👥 Top Contributors
{format_top_contributors(stats['author_stats'])}

## 🎯 Development Focus
{analyze_development_focus(events)}
"""
    
    # Add LLM insights if available
    if len(events) >= 5:
        llm_insights = await get_llm_repository_insights(events[:20], repo_name)
        if llm_insights:
            summary += f"\n## 🤖 AI Insights\n{llm_insights}"
    
    return summary

async def list_all_commits(events: List[Dict], query: str) -> str:
    """List all commits with enhanced formatting"""
    if not events:
        return "No commits found."
    
    # Check if user wants a specific number
    limit = extract_number_from_query(query) or 20
    limited_events = events[:limit]
    
    commit_list = f"# All Commits ({len(limited_events)} shown)\n\n"
    
    for i, event in enumerate(limited_events, 1):
        commit_hash = event.get('commit_hash', 'unknown')[:8]
        message = event.get('message', 'No message')
        author = event.get('author', 'Unknown').split('<')[0].strip()
        timestamp = format_timestamp(event.get('timestamp', ''))
        files_count = len(event.get('files_changed', []))
        
        commit_list += f"**{i}. [{commit_hash}]** {message}\n"
        commit_list += f"   👤 {author} • 📅 {timestamp} • 📁 {files_count} files\n\n"
    
    if len(events) > limit:
        commit_list += f"\n*Showing {limit} of {len(events)} total commits*\n"
    
    return commit_list

async def analyze_specific_commit(events: List[Dict], query: str) -> str:
    """Analyze a specific commit with detailed explanation"""
    # Try to extract commit hash from query
    commit_hash = extract_commit_hash(query)
    
    target_commit = None
    if commit_hash:
        # Find by hash
        for event in events:
            if event.get('commit_hash', '').startswith(commit_hash):
                target_commit = event
                break
    else:
        # Use the most relevant commit (first in results)
        target_commit = events[0] if events else None
    
    if not target_commit:
        return "Could not find the specified commit."
    
    return await generate_detailed_commit_analysis(target_commit, events)

async def generate_detailed_commit_analysis(commit: Dict, all_events: List[Dict]) -> str:
    """Generate comprehensive commit analysis"""
    commit_hash = commit.get('commit_hash', 'unknown')
    message = commit.get('message', 'No message')
    author = commit.get('author', 'Unknown')
    timestamp = commit.get('timestamp', '')
    files_changed = commit.get('files_changed', [])
    
    analysis = f"""# Commit Analysis: {commit_hash[:8]}

## 📋 Basic Information
**Message:** {message}
**Author:** {author}
**Date:** {format_timestamp(timestamp)}
**Files Modified:** {len(files_changed)} files

## 📁 Files Changed
{format_files_changed(files_changed)}

## 🔍 Commit Context
{await analyze_commit_context(commit, all_events)}

## 🎯 Change Analysis
{classify_commit_impact(commit)}
"""
    
    # Add LLM explanation if available
    llm_explanation = await get_llm_commit_explanation(commit)
    if llm_explanation:
        analysis += f"\n## 🤖 Detailed Explanation\n{llm_explanation}"
    
    return analysis

async def get_llm_commit_explanation(commit: Dict) -> str:
    """Get LLM explanation for a specific commit"""
    commit_hash = commit.get('commit_hash', 'unknown')[:8]
    message = commit.get('message', '')
    files = commit.get('files_changed', [])[:10]  # Limit files
    author = commit.get('author', 'Unknown')
    
    prompt = f"""Analyze this git commit in detail:

Commit: {commit_hash}
Message: {message}
Author: {author}
Files changed: {', '.join(files)}

Please explain:
1. What this commit does (technical details)
2. Why this change was likely needed
3. The impact of these changes
4. Any patterns or development practices shown

Provide a clear, technical explanation suitable for developers."""
    
    return await query_llm(prompt)

async def query_enhanced_llm(query: str, events: List[Dict]) -> str:
    """Enhanced LLM query with better context"""
    if not events:
        return ""
    
    # Build rich context
    context = build_rich_context(events[:15], query)
    
    prompt = f"""You are analyzing a git repository. Here's the commit data:

{context}

User Question: {query}

Please provide a comprehensive answer that:
1. Directly addresses the user's question
2. Uses specific commit information from the data above
3. Explains technical details clearly
4. Provides actionable insights

Be specific and use actual commit hashes, author names, and file names from the data."""
    
    return await query_llm(prompt)

def build_rich_context(events: List[Dict], query: str) -> str:
    """Build rich context for LLM queries"""
    context = ""
    
    for i, event in enumerate(events, 1):
        commit_hash = event.get('commit_hash', 'unknown')[:8]
        message = event.get('message', '')
        author = event.get('author', 'Unknown').split('<')[0].strip()
        timestamp = event.get('timestamp', '')[:10]
        files = event.get('files_changed', [])[:5]
        
        context += f"Commit {i}:\n"
        context += f"  Hash: {commit_hash}\n"
        context += f"  Message: {message}\n"
        context += f"  Author: {author}\n"
        context += f"  Date: {timestamp}\n"
        context += f"  Files: {', '.join(files)}\n\n"
    
    return context

def analyze_comprehensive_stats(events: List[Dict]) -> dict:
    """Analyze comprehensive repository statistics"""
    stats = {
        'total_commits': len(events),
        'contributors': len(set(e.get('author', '') for e in events)),
        'files_touched': len(set(f for e in events for f in e.get('files_changed', []))),
        'commit_breakdown': {},
        'author_stats': {},
        'primary_language': 'Unknown',
        'date_range': 'Unknown'
    }
    
    # Analyze commit types
    commit_types = {
        'Features': 0, 'Fixes': 0, 'Refactors': 0, 
        'Documentation': 0, 'Tests': 0, 'Build': 0, 'Other': 0
    }
    
    # Analyze authors
    author_commits = {}
    
    # Analyze file extensions for language detection
    extensions = {}
    
    for event in events:
        # Classify commit
        message = event.get('message', '').lower()
        if any(word in message for word in ['feat', 'add', 'implement', 'create', 'new']):
            commit_types['Features'] += 1
        elif any(word in message for word in ['fix', 'bug', 'patch', 'resolve']):
            commit_types['Fixes'] += 1
        elif any(word in message for word in ['refactor', 'improve', 'optimize']):
            commit_types['Refactors'] += 1
        elif any(word in message for word in ['doc', 'readme', 'comment']):
            commit_types['Documentation'] += 1
        elif any(word in message for word in ['test', 'spec']):
            commit_types['Tests'] += 1
        elif any(word in message for word in ['build', 'deps', 'version']):
            commit_types['Build'] += 1
        else:
            commit_types['Other'] += 1
        
        # Count author commits
        author = event.get('author', 'Unknown').split('<')[0].strip()
        author_commits[author] = author_commits.get(author, 0) + 1
        
        # Analyze file extensions
        for file in event.get('files_changed', []):
            if '.' in file:
                ext = file.split('.')[-1].lower()
                extensions[ext] = extensions.get(ext, 0) + 1
    
    stats['commit_breakdown'] = commit_types
    stats['author_stats'] = sorted(author_commits.items(), key=lambda x: x[1], reverse=True)
    
    # Determine primary language
    if extensions:
        primary_ext = max(extensions.items(), key=lambda x: x[1])[0]
        lang_map = {
            'py': 'Python', 'js': 'JavaScript', 'ts': 'TypeScript',
            'java': 'Java', 'cpp': 'C++', 'cs': 'C#', 'go': 'Go',
            'rs': 'Rust', 'rb': 'Ruby', 'php': 'PHP'
        }
        stats['primary_language'] = lang_map.get(primary_ext, primary_ext.upper())
    
    # Date range
    if events:
        timestamps = [e.get('timestamp', '') for e in events if e.get('timestamp')]
        if timestamps:
            timestamps.sort()
            first = timestamps[-1][:10] if timestamps else ''
            last = timestamps[0][:10] if timestamps else ''
            stats['date_range'] = f"{first} to {last}"
    
    return stats

def format_commit_breakdown(breakdown: dict) -> str:
    """Format commit breakdown for display"""
    total = sum(breakdown.values())
    if total == 0:
        return "No commits to analyze"
    
    result = ""
    for commit_type, count in breakdown.items():
        if count > 0:
            percentage = (count * 100) // total
            emoji_map = {
                'Features': '🚀', 'Fixes': '🐛', 'Refactors': '🔧',
                'Documentation': '📚', 'Tests': '🧪', 'Build': '📦', 'Other': '📝'
            }
            emoji = emoji_map.get(commit_type, '📝')
            result += f"{emoji} **{commit_type}:** {count} commits ({percentage}%)\n"
    
    return result

def format_recent_commits(events: List[Dict]) -> str:
    """Format recent commits for display"""
    if not events:
        return "No recent commits"
    
    result = ""
    for event in events:
        commit_hash = event.get('commit_hash', 'unknown')[:8]
        message = event.get('message', '')[:60]
        author = event.get('author', 'Unknown').split('<')[0].strip()
        timestamp = format_timestamp(event.get('timestamp', ''))
        
        result += f"• **[{commit_hash}]** {message}\n"
        result += f"  👤 {author} • 📅 {timestamp}\n\n"
    
    return result

def format_top_contributors(author_stats: List[tuple]) -> str:
    """Format top contributors"""
    if not author_stats:
        return "No contributor data"
    
    result = ""
    for i, (author, commits) in enumerate(author_stats[:5], 1):
        result += f"{i}. **{author}** - {commits} commits\n"
    
    return result

def analyze_development_focus(events: List[Dict]) -> str:
    """Analyze current development focus"""
    if not events:
        return "No data to analyze"
    
    recent_events = events[:10]
    focus_areas = {}
    
    for event in recent_events:
        message = event.get('message', '').lower()
        if 'fix' in message:
            focus_areas['Bug Fixing'] = focus_areas.get('Bug Fixing', 0) + 1
        if any(word in message for word in ['feat', 'add', 'new']):
            focus_areas['Feature Development'] = focus_areas.get('Feature Development', 0) + 1
        if 'refactor' in message:
            focus_areas['Code Refactoring'] = focus_areas.get('Code Refactoring', 0) + 1
        if any(word in message for word in ['doc', 'readme']):
            focus_areas['Documentation'] = focus_areas.get('Documentation', 0) + 1
    
    if not focus_areas:
        return "Mixed development activities"
    
    top_focus = max(focus_areas.items(), key=lambda x: x[1])
    result = f"**Primary Focus:** {top_focus[0]}\n"
    
    if len(focus_areas) > 1:
        result += "**Other Activities:** " + ", ".join([area for area, _ in focus_areas.items() if area != top_focus[0]])
    
    return result

async def get_llm_repository_insights(events: List[Dict], repo_name: str) -> str:
    """Get LLM insights about the repository"""
    recent_commits = events[:10]
    context = ""
    
    for event in recent_commits:
        context += f"- {event.get('message', '')}\n"
    
    prompt = f"""Analyze this repository "{repo_name}" based on recent commits:

{context}

Provide 3-4 key insights about:
1. Code quality and development practices
2. Project direction and priorities  
3. Team collaboration patterns
4. Technical debt or areas for improvement

Keep insights concise and actionable."""
    
    return await query_llm(prompt)

# Helper functions

def extract_commit_hash(query: str) -> str:
    """Extract commit hash from query"""
    import re
    # Look for hexadecimal strings that could be commit hashes
    matches = re.findall(r'\b[a-fA-F0-9]{6,40}\b', query)
    return matches[0] if matches else ""

def extract_number_from_query(query: str) -> int:
    """Extract number from query (e.g., 'show 10 commits')"""
    import re
    numbers = re.findall(r'\b(\d+)\b', query)
    return int(numbers[0]) if numbers else 0

def format_timestamp(timestamp: str) -> str:
    """Format timestamp for display"""
    if not timestamp:
        return "Unknown"
    try:
        from datetime import datetime
        dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
        return dt.strftime("%Y-%m-%d %H:%M")
    except:
        return timestamp[:16] if len(timestamp) > 16 else timestamp

def format_files_changed(files: List[str]) -> str:
    """Format files changed for display"""
    if not files:
        return "No files specified"
    
    if len(files) <= 10:
        return "\n".join(f"• {file}" for file in files)
    else:
        result = "\n".join(f"• {file}" for file in files[:10])
        result += f"\n• ... and {len(files) - 10} more files"
        return result

async def analyze_commit_context(commit: Dict, all_events: List[Dict]) -> str:
    """Analyze commit in context of surrounding commits"""
    commit_hash = commit.get('commit_hash', '')
    
    # Find position in timeline
    commit_index = -1
    for i, event in enumerate(all_events):
        if event.get('commit_hash') == commit_hash:
            commit_index = i
            break
    
    if commit_index == -1:
        return "Could not determine commit context"
    
    context = ""
    
    # Previous commit
    if commit_index < len(all_events) - 1:
        prev_commit = all_events[commit_index + 1]
        context += f"**Previous:** [{prev_commit.get('commit_hash', '')[:8]}] {prev_commit.get('message', '')[:50]}\n"
    
    # Next commit
    if commit_index > 0:
        next_commit = all_events[commit_index - 1]
        context += f"**Following:** [{next_commit.get('commit_hash', '')[:8]}] {next_commit.get('message', '')[:50]}\n"
    
    return context if context else "This is an isolated commit"

def classify_commit_impact(commit: Dict) -> str:
    """Classify the impact and type of commit"""
    message = commit.get('message', '').lower()
    files_count = len(commit.get('files_changed', []))
    
    # Determine impact level
    if files_count > 10:
        impact = "High Impact (10+ files changed)"
    elif files_count > 3:
        impact = "Medium Impact (3-10 files changed)"
    else:
        impact = "Low Impact (1-3 files changed)"
    
    # Determine change type
    change_type = "General Change"
    if any(word in message for word in ['feat', 'add', 'new', 'implement']):
        change_type = "Feature Addition"
    elif any(word in message for word in ['fix', 'bug', 'patch']):
        change_type = "Bug Fix"
    elif any(word in message for word in ['refactor', 'improve', 'optimize']):
        change_type = "Code Refactoring"
    elif any(word in message for word in ['doc', 'readme']):
        change_type = "Documentation Update"
    elif any(word in message for word in ['test', 'spec']):
        change_type = "Test Addition/Update"
    
    return f"**Impact:** {impact}\n**Type:** {change_type}"

async def generate_repository_overview(events: List[Dict]) -> str:
    """Generate comprehensive overview for any repository"""
    if not events:
        return "No repository data available."
    
    # Extract repository name dynamically
    repo_path = events[0].get("repository", "unknown")
    repo_name = repo_path.split('\\')[-1] if '\\' in repo_path else repo_path.split('/')[-1]
    
    # Analyze commits
    total_commits = len(events)
    authors = set(e.get("author", "") for e in events)
    files_touched = set()
    for e in events:
        files_touched.update(e.get("files_changed", []))
    
    # Categorize commits by type
    commit_types = {
        "features": [],
        "fixes": [],
        "refactors": [],
        "docs": [],
        "tests": [],
        "builds": [],
        "other": []
    }
    
    for event in events:
        msg = event.get("message", "").lower()
        categorized = False
        
        if any(keyword in msg for keyword in ["feat", "add", "implement", "create", "new"]):
            commit_types["features"].append(event)
            categorized = True
        elif any(keyword in msg for keyword in ["fix", "bug", "patch", "resolve", "correct"]):
            commit_types["fixes"].append(event)
            categorized = True
        elif any(keyword in msg for keyword in ["refactor", "improve", "enhance", "optimize"]):
            commit_types["refactors"].append(event)
            categorized = True
        elif any(keyword in msg for keyword in ["doc", "readme", "comment"]):
            commit_types["docs"].append(event)
            categorized = True
        elif any(keyword in msg for keyword in ["test", "spec"]):
            commit_types["tests"].append(event)
            categorized = True
        elif any(keyword in msg for keyword in ["build", "deps", "dependency", "version"]):
            commit_types["builds"].append(event)
            categorized = True
        
        if not categorized:
            commit_types["other"].append(event)
    
    # Get date range
    date_range = ""
    if events:
        timestamps = [e.get("timestamp", "") for e in events if e.get("timestamp")]
        if timestamps:
            timestamps.sort()
            first_date = timestamps[0][:10] if timestamps[0] else "unknown"
            last_date = timestamps[-1][:10] if timestamps[-1] else "unknown"
            date_range = f"from {first_date} to {last_date}"
    
    # Infer project type from files
    project_type = infer_project_type(files_touched)
    
    # Build overview
    overview = f"""## {repo_name} Repository Overview

**Project Type:** {project_type}

**Repository Statistics:**
- Total Commits: {total_commits}
- Contributors: {len(authors)}
- Files Modified: {len(files_touched)}
- Development Period: {date_range}

**Development Activity Breakdown:**"""
    
    # Add non-empty categories
    for category, commits in commit_types.items():
        if commits and category != "other":
            percentage = len(commits) * 100 // max(total_commits, 1)
            emoji = get_category_emoji(category)
            overview += f"\n- {emoji} {category.title()}: {len(commits)} commits ({percentage}%)"
    
    if commit_types["other"]:
        overview += f"\n- 📝 Other: {len(commit_types['other'])} commits"
    
    # Analyze technology stack
    tech_stack = analyze_tech_stack(files_touched)
    if tech_stack:
        overview += "\n\n**Technology Stack:**"
        for tech, count in tech_stack[:5]:
            overview += f"\n- {tech}: {count} files"
    
    # Key areas of development
    if commit_types["features"]:
        overview += "\n\n**Key Features Developed:**"
        for feat in commit_types["features"][:5]:
            msg = feat.get("message", "")
            msg = msg.replace('\n', ' ').strip()
            overview += f"\n- {msg[:80]}"
    
    # Development velocity
    overview += "\n\n**Development Insights:**"
    overview += analyze_velocity(events)
    
    # Recent focus
    if len(events) > 5:
        recent = events[:5]
        overview += "\n\n**Recent Development Focus:**"
        recent_types = set()
        for event in recent:
            msg = event.get("message", "").lower()
            if "fix" in msg:
                recent_types.add("bug fixing")
            elif "feat" in msg:
                recent_types.add("feature development")
            elif "refactor" in msg:
                recent_types.add("code refactoring")
        
        if recent_types:
            overview += f"\nCurrently focused on: {', '.join(recent_types)}"
    
    return overview

def infer_project_type(files: set) -> str:
    """Infer project type from file extensions"""
    """Infer project type from file extensions"""
    extensions = {}
    for file in files:
        if '.' in file:
            ext = file.split('.')[-1].lower()
            extensions[ext] = extensions.get(ext, 0) + 1
    
    if not extensions:
        return "General Software Project"
    
    # Sort by frequency
    top_ext = sorted(extensions.items(), key=lambda x: x[1], reverse=True)
    
    # Infer based on dominant extensions
    project_indicators = {
        "js": "JavaScript/Node.js",
        "ts": "TypeScript",
        "py": "Python",
        "java": "Java",
        "cpp": "C++",
        "cs": "C#",
        "go": "Go",
        "rs": "Rust",
        "rb": "Ruby",
        "php": "PHP",
        "swift": "Swift/iOS",
        "kt": "Kotlin/Android",
        "vue": "Vue.js",
        "jsx": "React",
        "tsx": "React TypeScript"
    }
    
    for ext, _ in top_ext:
        if ext in project_indicators:
            return f"{project_indicators[ext]} Project"
    
    # Check for web project
    web_extensions = {"html", "css", "js", "jsx", "tsx", "vue", "scss", "sass"}
    if any(ext in web_extensions for ext, _ in top_ext[:5]):
        return "Web Application"
    
    # Check for mobile
    mobile_extensions = {"swift", "kt", "java", "xml"}
    if any(ext in mobile_extensions for ext, _ in top_ext[:5]):
        return "Mobile Application"
    
    return "Software Project"

def analyze_tech_stack(files: set) -> list:
    """Analyze technology stack from files"""
    """Analyze technology stack from files"""
    tech_categories = {
        "Frontend": ["html", "css", "js", "jsx", "tsx", "vue", "scss"],
        "Backend": ["py", "java", "go", "rb", "php", "cs"],
        "Configuration": ["json", "yaml", "yml", "toml", "ini"],
        "Documentation": ["md", "txt", "rst", "doc"],
        "Testing": ["test.js", "spec.js", "test.py", "spec.ts"],
        "Database": ["sql", "db", "sqlite"],
        "DevOps": ["dockerfile", "docker-compose.yml", "jenkinsfile"]
    }
    
    stack = {}
    for file in files:
        file_lower = file.lower()
        for category, extensions in tech_categories.items():
            if any(ext in file_lower for ext in extensions):
                stack[category] = stack.get(category, 0) + 1
    
    return sorted(stack.items(), key=lambda x: x[1], reverse=True)

def get_category_emoji(category: str) -> str:
    """Get emoji for commit category"""
    """Get emoji for commit category"""
    emojis = {
        "features": "🚀",
        "fixes": "🐛",
        "refactors": "🔧",
        "docs": "📚",
        "tests": "🧪",
        "builds": "📦"
    }
    return emojis.get(category, "📝")

def analyze_velocity(events: List[Dict]) -> str:
    """Analyze development velocity"""
    """Analyze development velocity"""
    if len(events) < 2:
        return "\n- Limited data for velocity analysis"
    
    from datetime import datetime
    timestamps = []
    for event in events:
        ts = event.get("timestamp", "")
        if ts:
            try:
                timestamps.append(datetime.fromisoformat(ts.replace("Z", "")))
            except:
                pass
    
    if len(timestamps) < 2:
        return "\n- Unable to calculate velocity"
    
    # Calculate time span
    time_span = (timestamps[0] - timestamps[-1]).days
    if time_span == 0:
        time_span = 1
    
    commits_per_day = len(events) / time_span
    
    insights = f"\n- Average velocity: {commits_per_day:.1f} commits per day"
    
    if commits_per_day > 5:
        insights += "\n- High development activity indicates active project"
    elif commits_per_day > 1:
        insights += "\n- Steady development pace"
    else:
        insights += "\n- Low commit frequency, possibly maintenance mode"
    
    return insights

async def explain_development_decisions(events: List[Dict], query: str) -> str:
    """Explain why certain development decisions were made"""
    """Explain why certain development decisions were made"""
    if not events:
        return "No events to analyze."
    
    explanation = "Based on the commit history, here's the analysis:\n\n"
    
    # Look for patterns that indicate reasons
    fixes = [e for e in events if "fix" in e.get("message", "").lower()]
    features = [e for e in events if "feat" in e.get("message", "").lower() or "add" in e.get("message", "").lower()]
    refactors = [e for e in events if "refactor" in e.get("message", "").lower()]
    
    if fixes:
        explanation += "**Bug Fixes Indicate:**\n"
        explanation += "- Ongoing maintenance and quality improvement\n"
        explanation += "- Response to user feedback or testing\n"
        common_fix_areas = analyze_common_areas(fixes)
        if common_fix_areas:
            explanation += f"- Problem areas: {', '.join(common_fix_areas[:3])}\n"
        explanation += "\n"
    
    if features:
        explanation += "**Feature Development Shows:**\n"
        explanation += "- Active growth and expansion\n"
        explanation += "- Response to requirements or user needs\n"
        feature_areas = analyze_common_areas(features)
        if feature_areas:
            explanation += f"- Focus areas: {', '.join(feature_areas[:3])}\n"
        explanation += "\n"
    
    if refactors:
        explanation += "**Refactoring Suggests:**\n"
        explanation += "- Code quality improvements\n"
        explanation += "- Technical debt reduction\n"
        explanation += "- Performance optimization efforts\n"
    
    return explanation

def analyze_common_areas(events: List[Dict]) -> List[str]:
    """Analyze common areas from commit messages and files"""
    areas = {}
    for event in events:
        # Extract from message
        msg = event.get("message", "").lower()
        words = msg.split()
        for word in words:
            if len(word) > 4 and word.isalpha():  # Meaningful words
                areas[word] = areas.get(word, 0) + 1
        
        # Extract from files
        for file in event.get("files_changed", []):
            if '/' in file:
                component = file.split('/')[0]
                areas[component] = areas.get(component, 0) + 1
    
    # Sort by frequency
    sorted_areas = sorted(areas.items(), key=lambda x: x[1], reverse=True)
    return [area for area, _ in sorted_areas[:5] if area not in ["the", "and", "for", "with", "from"]]

# Add these missing helper functions that are referenced but not in your current code:
async def generate_timeline_summary(events: List[Dict]) -> str:
    """Generate timeline summary of development"""
    if not events:
        return "No timeline data available."
    
    from datetime import datetime
    timeline = defaultdict(list)
    
    for event in events:
        timestamp = event.get("timestamp", "")
        if timestamp:
            try:
                dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
                week = dt.strftime("%Y-W%U")
                timeline[week].append(event)
            except:
                pass
    
    summary = "## Development Timeline\n\n"
    sorted_weeks = sorted(timeline.items(), reverse=True)[:8]
    
    for week, commits in sorted_weeks:
        summary += f"**Week {week}:** {len(commits)} commits\n"
        for commit in commits[:2]:
            summary += f"  - {commit.get('message', '')[:60]}\n"
        summary += "\n"
    
    return summary

async def analyze_development_patterns(events: List[Dict]) -> str:
    """Analyze patterns in development"""
    patterns = analyze_commit_patterns(events)
    
    analysis = "## Development Patterns\n\n"
    analysis += f"**Commit Types:**\n"
    for commit_type, count in patterns["commit_types"].items():
        if count > 0:
            analysis += f"- {commit_type.title()}: {count} commits\n"
    
    if patterns.get("most_changed_files"):
        analysis += f"\n**Most Changed Files:**\n"
        for file, count in patterns["most_changed_files"][:5]:
            analysis += f"- {file}: {count} changes\n"
    
    return analysis

async def explain_implementation_approach(events: List[Dict], query: str) -> str:
    """Explain how something was implemented"""
    if not events:
        return "No implementation details found."
    
    explanation = "## Implementation Approach\n\n"
    
    for event in events[:5]:
        explanation += f"**{event.get('message', '')[:60]}**\n"
        if event.get("files_changed"):
            explanation += f"Files modified: {', '.join(event.get('files_changed', [])[:3])}\n"
        explanation += "\n"
    
    return explanation

async def analyze_changes(events: List[Dict], query: str) -> str:
    """Analyze what changed between commits"""
    if not events:
        return "No changes found."
    
    analysis = "## Changes Analysis\n\n"
    
    for event in events[:5]:
        analysis += f"**{event.get('timestamp', '')[:10]}** - {event.get('message', '')[:60]}\n"
        if event.get("files_changed"):
            analysis += f"- Modified: {', '.join(event.get('files_changed', [])[:3])}\n"
        analysis += "\n"
    
    return analysis

async def generate_contextual_listing(query: str, events: List[Dict]) -> str:
    """Generate a contextual listing of events"""
    if not events:
        return "No events found matching your query."
    
    listing = f"Found {len(events)} relevant commits:\n\n"
    
    for i, event in enumerate(events[:10], 1):
        listing += f"{i}. **{event.get('message', '')[:60]}**\n"
        listing += f"   Author: {event.get('author', 'Unknown')}\n"
        listing += f"   Date: {event.get('timestamp', '')[:10]}\n"
        if event.get("files_changed"):
            listing += f"   Files: {', '.join(event.get('files_changed', [])[:3])}\n"
        listing += "\n"
    
    return listing

async def query_llm(prompt: str) -> str:
    """Query local LLM using Ollama"""
    logger.info(f"Calling Ollama with prompt: {prompt[:100]}...")
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": "qwen2.5-coder:1.5b",
                    "prompt": prompt,
                    "stream": False
                },
                timeout=60.0
            )
            if response.status_code == 200:
                return response.json().get("response", "")
                logger.info(f"Ollama responded successfully: {len(result)} chars")
                return result
            else:
                logger.error(f"Ollama error: {response.status_code} - {response.text}")
    except httpx.TimeoutException:
        logger.error("Ollama timeout - increase timeout or use smaller prompts")
    except Exception as e:
        logger.warning(f"LLM query failed: {e}")
    return ""

async def check_commit_exists(commit_hash: str) -> bool:
    """Check if a commit already exists in Qdrant"""
    if not qdrant_client or not commit_hash:
        return False
    
    try:
        # Search for exact commit hash in payloads
        results = qdrant_client.scroll(
            collection_name=events_collection,
            scroll_filter={
                "must": [
                    {
                        "key": "commit_hash",
                        "match": {"value": commit_hash}
                    }
                ]
            },
            limit=1,
            with_payload=False,
            with_vectors=False
        )[0]
        
        return len(results) > 0
    except Exception as e:
        logger.debug(f"Error checking commit existence: {e}")
        return False

# ============= Endpoints =============
@app.get("/api/test-llm")
async def test_llm():
    """Test if Ollama LLM is working"""
    try:
        prompt = "Explain what Context Keeper does in one sentence."
        response = await query_llm(prompt)
        
        if response:
            return {
                "status": "working",
                "llm_response": response,
                "service": "ollama"
            }
        else:
            return {
                "status": "not_working",
                "message": "Ollama returned empty response",
                "suggestion": "Check if Ollama is running: docker ps | grep ollama"
            }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "suggestion": "Start Ollama with: docker run -d -p 11434:11434 ollama/ollama"
        }
    
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

@app.post("/api/check/repository")
async def check_repository_status(request: dict):
    """Check how many commits from a repository are already indexed"""
    repo_path = request.get("repo_path")
    
    if not repo_path:
        raise HTTPException(status_code=400, detail="repo_path required")
    
    # Get commit hashes from the repository
    import subprocess
    try:
        result = subprocess.run(
            ["git", "log", "--pretty=format:%H", "-100"],  # Get last 100 commit hashes
            cwd=repo_path,
            capture_output=True,
            text=True,
            check=True
        )
        
        commit_hashes = result.stdout.strip().split('\n')
        total_commits = len(commit_hashes)
        
        # Check how many already exist
        existing_count = 0
        missing_commits = []
        
        for commit_hash in commit_hashes:
            if await check_commit_exists(commit_hash):
                existing_count += 1
            else:
                missing_commits.append(commit_hash[:8])
        
        return {
            "repository": repo_path,
            "total_commits_checked": total_commits,
            "already_indexed": existing_count,
            "missing": total_commits - existing_count,
            "missing_commits": missing_commits[:10],  # Show first 10 missing
            "fully_indexed": existing_count == total_commits
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error checking repository: {e}")

@app.post("/api/repositories/select")
async def select_repository(request: dict):
    """Mark a repository as selected for queries"""
    repository = request.get("repository")
    selected = request.get("selected", True)
    
    if repository in indexed_repositories:
        indexed_repositories[repository]["selected"] = selected
        save_repositories()
        return {"status": "success", "repository": repository, "selected": selected}
    else:
        raise HTTPException(status_code=404, detail="Repository not found")
    

@app.delete("/api/repositories/clear")
async def clear_all_repositories():
    """Clear all repository data and reset"""
    global indexed_repositories
    
    # Clear from memory
    indexed_repositories = {}
    
    # Clear from Qdrant (optional - keeps the data but removes repo tracking)
    # You might want to keep the commits and just reset the repo list
    
    # Save empty state
    save_repositories()
    
    return {"status": "success", "message": "All repositories cleared"}

@app.delete("/api/repositories/{repo_path:path}")
async def delete_repository(repo_path: str):
    """Delete a specific repository and all its commits"""
    global indexed_repositories
    
    if repo_path not in indexed_repositories:
        raise HTTPException(status_code=404, detail="Repository not found")
    
    try:
        # First, remove all commits for this repository from Qdrant
        if qdrant_client:
            logger.info(f"Removing all commits for repository: {repo_path}")
            
            # Get all commit IDs for this repository
            commit_ids_to_delete = []
            offset = None
            
            while True:
                results = qdrant_client.scroll(
                    collection_name=events_collection,
                    scroll_filter={
                        "must": [{"key": "repository", "match": {"value": repo_path}}]
                    },
                    limit=100,
                    offset=offset,
                    with_payload=False,
                    with_vectors=False
                )
                points, next_offset = results
                
                for point in points:
                    commit_ids_to_delete.append(point.id)
                
                if next_offset is None:
                    break
                offset = next_offset
            
            # Delete all commits for this repository
            if commit_ids_to_delete:
                qdrant_client.delete(
                    collection_name=events_collection,
                    points_selector=commit_ids_to_delete
                )
                logger.info(f"Deleted {len(commit_ids_to_delete)} commits from Qdrant")
        
        # Remove from in-memory storage
        global git_events
        git_events = [e for e in git_events if e.get("repository") != repo_path]
        
        # Remove from repository tracking
        del indexed_repositories[repo_path]
        save_repositories()
        
        return {
            "status": "success", 
            "message": f"Repository {repo_path} and all its commits deleted successfully",
            "commits_deleted": len(commit_ids_to_delete) if qdrant_client else 0
        }
        
    except Exception as e:
        logger.error(f"Error deleting repository {repo_path}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete repository: {e}")

@app.post("/api/analyze/commit")
async def analyze_commit(request: dict):
    """Analyze what a specific commit does"""
    commit_hash = request.get("commit_hash")
    
    # Find the commit
    commit_data = None
    for event in git_events:
        if event.get("commit_hash", "").startswith(commit_hash):
            commit_data = event
            break
    
    if not commit_data:
        raise HTTPException(status_code=404, detail="Commit not found")
    
    # Analyze the commit
    analysis = {
        "summary": commit_data.get("message"),
        "impact": {
            "files_changed": len(commit_data.get("files_changed", [])),
            "type": classify_commit_type(commit_data.get("message", ""))
        },
        "context": {}
    }
    
    # Find related commits (before and after)
    commit_index = git_events.index(commit_data)
    if commit_index > 0:
        analysis["context"]["previous"] = git_events[commit_index - 1].get("message")
    if commit_index < len(git_events) - 1:
        analysis["context"]["next"] = git_events[commit_index + 1].get("message")
    
    # Generate intelligent explanation
    if commit_data.get("code_diffs"):
        prompt = f"""Explain this commit in detail:
        
Message: {commit_data.get('message')}
Files changed: {', '.join(commit_data.get('files_changed', []))}

What does this commit do and why was it likely needed?"""
        
        explanation = await query_llm(prompt)
        if explanation:
            analysis["explanation"] = explanation
    
    return analysis

@app.post("/api/analyze/repository")
async def analyze_repository(request: dict):
    """Provide intelligent repository analysis"""
    repo_path = request.get("repository")
    
    # Filter events for this repository
    repo_events = [e for e in git_events if repo_path in e.get("repository", "")]
    
    if not repo_events:
        raise HTTPException(status_code=404, detail="Repository not found")
    
    # Analyze patterns
    analysis = {
        "overview": {
            "total_commits": len(repo_events),
            "first_commit": repo_events[-1].get("timestamp") if repo_events else None,
            "last_commit": repo_events[0].get("timestamp") if repo_events else None
        },
        "patterns": analyze_commit_patterns(repo_events),
        "recommendations": []
    }
    
    # Generate insights
    prompt = f"""Analyze this repository's development patterns:
    
Total commits: {len(repo_events)}
Recent activity: {', '.join(e.get('message', '')[:30] for e in repo_events[:5])}

Provide insights about:
1. Development velocity
2. Code quality trends
3. Areas of focus
4. Potential improvements"""
    
    insights = await query_llm(prompt)
    if insights:
        analysis["insights"] = insights
    
    return analysis

def analyze_commit_patterns(events):
    """Analyze patterns in commit history"""
    patterns = {
        "commit_frequency": {},
        "common_files": {},
        "commit_types": {
            "feature": 0,
            "fix": 0,
            "refactor": 0,
            "docs": 0,
            "other": 0
        }
    }
    
    for event in events:
        # Classify commit type
        message = event.get("message", "").lower()
        if "feat" in message or "add" in message:
            patterns["commit_types"]["feature"] += 1
        elif "fix" in message or "bug" in message:
            patterns["commit_types"]["fix"] += 1
        elif "refactor" in message:
            patterns["commit_types"]["refactor"] += 1
        elif "doc" in message:
            patterns["commit_types"]["docs"] += 1
        else:
            patterns["commit_types"]["other"] += 1
        
        # Track file changes
        for file in event.get("files_changed", []):
            patterns["common_files"][file] = patterns["common_files"].get(file, 0) + 1
    
    # Get most changed files
    patterns["most_changed_files"] = sorted(
        patterns["common_files"].items(), 
        key=lambda x: x[1], 
        reverse=True
    )[:10]
    
    return patterns

@app.post("/api/ingest/git")
async def ingest_git_event(event: Dict[str, Any], background_tasks: BackgroundTasks):
    """Ingest git event with repository tracking"""
    try:
        commit_hash = event.get("commit_hash", "")
        repository = event.get("repository", "unknown")

        # Normalize the repository path
        if repository != "unknown":
            repository = normalize_repo_path(repository)
            event["repository"] = repository  # Update the event with normalized path
        
        # Track repository
        if repository != "unknown" and repository not in indexed_repositories:
            indexed_repositories[repository] = {
                "path": repository,
                "first_indexed": datetime.now().isoformat(),
                "last_updated": datetime.now().isoformat(),
                "commit_count": 0,
                "selected": True
            }
        
        if repository in indexed_repositories:
            indexed_repositories[repository]["last_updated"] = datetime.now().isoformat()
            indexed_repositories[repository]["commit_count"] = \
                indexed_repositories[repository].get("commit_count", 0) + 1
            
            # Save periodically
            if indexed_repositories[repository]["commit_count"] % 10 == 0:
                save_repositories()
        
        # Check for duplicate
        if commit_hash and await check_commit_exists(commit_hash):
            return {
                "status": "duplicate",
                "message": "Commit already indexed",
                "commit": commit_hash[:8]
            }
        
        # Add to memory and Qdrant
        event["timestamp"] = event.get("timestamp", datetime.now().isoformat())
        event["type"] = "git_commit"
        event["repository"] = repository
        
        import uuid
        event_id = str(uuid.uuid4())
        
        git_events.append(event)
        
        if qdrant_client:
            background_tasks.add_task(store_in_qdrant, event, event_id)
        
        return {
            "status": "success",
            "message": "Git event ingested",
            "event_id": event_id,
            "commit": commit_hash[:8] if commit_hash else "",
            "repository": repository
        }
        
    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/query", response_model=QueryResponse)
async def query_context(request: QueryRequest):
    """Enhanced query with code understanding"""
    try:
        query = request.query
        logger.info(f"Query: {query}, Repository: {request.repository}")
        
        # Get relevant events
        limit = 50 if "summar" in query.lower() else request.limit * 2
        similar_events = await search_similar(query, limit, request.repository)
        
        # Use the new generalized answer generation
        answer = await generate_intelligent_answer(query, similar_events)
        # Format sources with better context
        formatted_sources = []
        for i, event in enumerate(similar_events[:request.limit]):
            formatted_sources.append({
                "type": event.get("type", "git_commit"),
                "content": event.get("message", ""),
                "timestamp": event.get("timestamp", ""),
                "author": event.get("author", ""),
                "commit": event.get("commit_hash", "")[:8] if event.get("commit_hash") else "",
                "repository": event.get("repository", "unknown"),
                "files_changed": event.get("files_changed", [])[:5],  # Add files
                "context": f"Commit {i+1} of {len(similar_events)}"
            })
        
        # Calculate confidence based on response quality
        confidence = calculate_confidence(query, answer, formatted_sources)
        
        return QueryResponse(
            answer=answer,
            sources=formatted_sources,
            confidence=confidence
        )
        
    except Exception as e:
        logger.error(f"Query failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

def classify_query_intent(query: str) -> str:
    """Understand what the user is asking for"""
    query_lower = query.lower()
    
    # Check for specific patterns
    if any(word in query_lower for word in ["summary", "summarize", "overview", "describe"]):
        return "summary"
    elif any(word in query_lower for word in ["why", "how", "explain", "understand", "reason"]):
        return "explanation"
    elif any(word in query_lower for word in ["timeline", "history", "evolution", "progress"]):
        return "timeline"
    elif any(word in query_lower for word in ["compare", "difference", "between", "versus"]):
        return "comparison"
    elif any(word in query_lower for word in ["list", "show all", "all commits"]):
        return "listing"
    else:
        return "search"

async def generate_repository_summary(query: str, events: List[Dict]) -> str:
    """Generate intelligent repository summary"""
    if not events:
        return "No repository data available for summary."
    
    # Analyze the repository
    analysis = {
        "total_commits": len(events),
        "authors": len(set(e.get("author", "") for e in events)),
        "files_touched": len(set(f for e in events for f in e.get("files_changed", []))),
        "date_range": f"{events[-1].get('timestamp', '')} to {events[0].get('timestamp', '')}",
    }
    
    # Categorize commits
    features = []
    fixes = []
    refactors = []
    docs = []
    
    for event in events:
        msg = event.get("message", "").lower()
        if "feat" in msg or "add" in msg or "implement" in msg:
            features.append(event)
        elif "fix" in msg or "bug" in msg or "patch" in msg:
            fixes.append(event)
        elif "refactor" in msg or "improve" in msg or "optimize" in msg:
            refactors.append(event)
        elif "doc" in msg or "readme" in msg:
            docs.append(event)
    
    # Build intelligent summary
    summary = f"""## Repository Analysis

**Overview:**
This repository contains {analysis['total_commits']} commits from {analysis['authors']} contributors, 
modifying {analysis['files_touched']} files between {analysis['date_range']}.

**Development Activity:**
- 🚀 Features/Additions: {len(features)} commits
- 🐛 Bug Fixes: {len(fixes)} commits
- 🔧 Refactoring: {len(refactors)} commits
- 📚 Documentation: {len(docs)} commits

**Key Features Developed:**"""
    
    # Add top features
    for feat in features[:5]:
        summary += f"\n- {feat.get('message', '')[:60]}"
    
    # Add recent focus
    summary += "\n\n**Recent Development Focus:**"
    recent = events[:10]
    
    # Identify patterns in recent commits
    recent_files = {}
    for event in recent:
        for file in event.get("files_changed", []):
            ext = file.split('.')[-1] if '.' in file else 'other'
            recent_files[ext] = recent_files.get(ext, 0) + 1
    
    if recent_files:
        top_types = sorted(recent_files.items(), key=lambda x: x[1], reverse=True)[:3]
        summary += f"\nMost active file types: {', '.join(f'.{ext} ({count} changes)' for ext, count in top_types)}"
    
    # Add AI-generated insights if LLM is available
    if events[:5]:  # Use recent commits for context
        prompt = f"""Based on these recent commits, provide 2-3 key insights about this repository's development:

Recent commits:
{chr(10).join(f"- {e.get('message', '')}" for e in events[:5])}

Provide brief, actionable insights about the codebase direction and quality."""
        
        insights = await query_llm(prompt)
        if insights:
            summary += f"\n\n**AI Insights:**\n{insights}"
    
    return summary

async def explain_code_changes(query: str, events: List[Dict]) -> str:
    """Explain what code changes were made and why"""
    if not events:
        return "No commits found matching your query."
    
    explanation = f"Based on the commit history, here's what I found:\n\n"
    
    # Group commits by type/purpose
    grouped = {}
    for event in events[:10]:  # Analyze top 10
        msg = event.get("message", "")
        # Extract the type (feat, fix, etc.)
        if ":" in msg:
            type_prefix = msg.split(":")[0].strip()
            if type_prefix not in grouped:
                grouped[type_prefix] = []
            grouped[type_prefix].append(event)
        else:
            if "other" not in grouped:
                grouped["other"] = []
            grouped["other"].append(event)
    
    # Explain each group
    for commit_type, commits in grouped.items():
        explanation += f"**{commit_type.title()} Changes:**\n"
        for commit in commits[:3]:
            explanation += f"- {commit.get('message', '')[:80]}\n"
            if commit.get("files_changed"):
                explanation += f"  Files: {', '.join(commit.get('files_changed', [])[:3])}\n"
        explanation += "\n"
    
    # Add contextual explanation
    if "why" in query.lower():
        # Try to explain the reasoning
        explanation += "**Likely Reasons for Changes:**\n"
        
        if any("fix" in e.get("message", "").lower() for e in events):
            explanation += "- Bug fixes indicate ongoing maintenance and quality improvements\n"
        if any("feat" in e.get("message", "").lower() for e in events):
            explanation += "- New features show active development and expansion\n"
        if any("refactor" in e.get("message", "").lower() for e in events):
            explanation += "- Refactoring suggests code quality and maintainability focus\n"
    
    return explanation

async def generate_timeline_analysis(query: str, events: List[Dict]) -> str:
    """Analyze development timeline"""
    if not events:
        return "No timeline data available."
    
    # Group by time periods
    from collections import defaultdict
    from datetime import datetime, timedelta
    
    timeline = defaultdict(list)
    
    for event in events:
        timestamp = event.get("timestamp", "")
        if timestamp:
            try:
                dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
                week = dt.strftime("%Y-W%U")
                timeline[week].append(event)
            except:
                pass
    
    analysis = "## Development Timeline\n\n"
    
    # Show weekly activity
    sorted_weeks = sorted(timeline.items(), reverse=True)[:8]  # Last 8 weeks
    
    for week, commits in sorted_weeks:
        analysis += f"**Week {week}:** {len(commits)} commits\n"
        # Show key commits
        for commit in commits[:2]:
            analysis += f"  - {commit.get('message', '')[:60]}\n"
        analysis += "\n"
    
    return analysis

async def get_repository_events(repository: str = None, limit: int = 100) -> List[Dict]:
    """Get events for a repository"""
    if repository:
        # Filter by repository
        return [e for e in git_events if repository in e.get("repository", "")][:limit]
    else:
        # Return all events
        return git_events[:limit]

def calculate_confidence(query: str, answer: str, sources: List[Dict]) -> float:
    """Calculate confidence score for the answer"""
    confidence = 0.5  # Base confidence
    
    # Increase confidence based on factors
    if sources:
        confidence += min(0.3, len(sources) * 0.05)  # More sources = higher confidence
    
    if len(answer) > 100:  # Detailed answer
        confidence += 0.1
    
    if "I found" in answer or "Based on" in answer:  # Concrete answer
        confidence += 0.1
    
    # Cap at 0.95
    return min(0.95, confidence)

@app.get("/api/repositories")
async def get_repositories():
    """Get list of all indexed repositories with stats"""
    # Update counts from Qdrant
    if qdrant_client:
        try:
            # Get actual commit counts per repository
            for repo_path in indexed_repositories:
                filter_condition = {
                    "must": [{"key": "repository", "match": {"value": repo_path}}]
                }
                
                count = 0
                offset = None
                while True:
                    results = qdrant_client.scroll(
                        collection_name=events_collection,
                        scroll_filter=filter_condition,
                        limit=100,
                        offset=offset,
                        with_payload=False,
                        with_vectors=False
                    )
                    points, next_offset = results
                    count += len(points)
                    if next_offset is None:
                        break
                    offset = next_offset
                
                indexed_repositories[repo_path]["commit_count"] = count
        except Exception as e:
            logger.error(f"Error updating repository counts: {e}")
    
    return {
        "repositories": indexed_repositories,
        "total": len(indexed_repositories)
    }



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

@app.post("/api/repositories/cleanup")
async def cleanup_orphaned_commits():
    """Remove commits from repositories that are no longer tracked"""
    if not qdrant_client:
        return {"status": "error", "message": "Qdrant not available"}
    
    try:
        # Get all unique repositories from Qdrant
        all_points = []
        offset = None
        while True:
            results = qdrant_client.scroll(
                collection_name=events_collection,
                limit=100,
                offset=offset,
                with_payload=True,
                with_vectors=False
            )
            points, next_offset = results
            all_points.extend(points)
            if next_offset is None:
                break
            offset = next_offset
        
        # Find orphaned commits (from repos not in indexed_repositories)
        orphaned_ids = []
        for point in all_points:
            repo = point.payload.get("repository", "")
            if repo and repo not in indexed_repositories:
                orphaned_ids.append(point.id)
        
        # Delete orphaned commits
        if orphaned_ids:
            qdrant_client.delete(
                collection_name=events_collection,
                points_selector=orphaned_ids
            )
            
        return {
            "status": "success", 
            "removed": len(orphaned_ids),
            "message": f"Removed {len(orphaned_ids)} orphaned commits"
        }
        
    except Exception as e:
        return {"status": "error", "message": str(e)}

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

@app.get("/api/commits/{commit_hash}")
async def get_commit_details(commit_hash: str):
    """Get detailed information about a specific commit"""
    if not commit_hash or len(commit_hash) < 6:
        raise HTTPException(status_code=400, detail="Invalid commit hash")
    
    # Search for the commit in memory first
    target_commit = None
    for event in git_events:
        if event.get('commit_hash', '').startswith(commit_hash):
            target_commit = event
            break
    
    # If not found in memory, search in Qdrant
    if not target_commit and qdrant_client:
        try:
            results = qdrant_client.scroll(
                collection_name=events_collection,
                scroll_filter={
                    "should": [
                        {"key": "commit_hash", "match": {"value": commit_hash}},
                        {"key": "commit_hash", "match": {"text": commit_hash}}
                    ]
                },
                limit=1,
                with_payload=True,
                with_vectors=False
            )[0]
            
            if results:
                target_commit = results[0].payload
        except Exception as e:
            logger.error(f"Error searching Qdrant for commit: {e}")
    
    if not target_commit:
        raise HTTPException(status_code=404, detail=f"Commit {commit_hash} not found")
    
    # Get repository context (other commits from same repo)
    repo_path = target_commit.get('repository', '')
    repo_commits = []
    if repo_path:
        repo_commits = [e for e in git_events if e.get('repository') == repo_path][:50]
    
    # Generate detailed analysis
    analysis = await generate_detailed_commit_analysis(target_commit, repo_commits)
    
    return {
        "commit": target_commit,
        "analysis": analysis,
        "repository": repo_path,
        "context_commits": len(repo_commits)
    }

@app.post("/api/commits/compare")
async def compare_commits(request: dict):
    """Compare two commits"""
    commit_hash_1 = request.get("commit_1")
    commit_hash_2 = request.get("commit_2")
    
    if not commit_hash_1 or not commit_hash_2:
        raise HTTPException(status_code=400, detail="Both commit hashes required")
    
    # Find both commits
    commits = []
    for hash_val in [commit_hash_1, commit_hash_2]:
        commit = None
        for event in git_events:
            if event.get('commit_hash', '').startswith(hash_val):
                commit = event
                break
        
        if not commit:
            raise HTTPException(status_code=404, detail=f"Commit {hash_val} not found")
        commits.append(commit)
    
    # Generate comparison
    comparison = await generate_commit_comparison(commits[0], commits[1])
    
    return {
        "commit_1": commits[0],
        "commit_2": commits[1],
        "comparison": comparison
    }

async def generate_commit_comparison(commit1: Dict, commit2: Dict) -> str:
    """Generate detailed comparison between two commits"""
    hash1 = commit1.get('commit_hash', '')[:8]
    hash2 = commit2.get('commit_hash', '')[:8]
    
    comparison = f"""# Commit Comparison: {hash1} vs {hash2}

## Basic Information
**Commit 1:** [{hash1}] {commit1.get('message', '')}
**Author:** {commit1.get('author', '')}
**Date:** {format_timestamp(commit1.get('timestamp', ''))}

**Commit 2:** [{hash2}] {commit2.get('message', '')}
**Author:** {commit2.get('author', '')}
**Date:** {format_timestamp(commit2.get('timestamp', ''))}

## File Changes Comparison
**Commit 1 Files:** {len(commit1.get('files_changed', []))} files
{format_files_changed(commit1.get('files_changed', [])[:5])}

**Commit 2 Files:** {len(commit2.get('files_changed', []))} files
{format_files_changed(commit2.get('files_changed', [])[:5])}

## Impact Analysis
**Commit 1:** {classify_commit_impact(commit1)}
**Commit 2:** {classify_commit_impact(commit2)}
"""
    
    # Add LLM comparison if available
    llm_comparison = await get_llm_commit_comparison(commit1, commit2)
    if llm_comparison:
        comparison += f"\n## Detailed Comparison\n{llm_comparison}"
    
    return comparison

async def get_llm_commit_comparison(commit1: Dict, commit2: Dict) -> str:
    """Get LLM comparison between two commits"""
    prompt = f"""Compare these two git commits in detail:

Commit 1: {commit1.get('commit_hash', '')[:8]}
Message: {commit1.get('message', '')}
Files: {', '.join(commit1.get('files_changed', [])[:5])}
Author: {commit1.get('author', '')}

Commit 2: {commit2.get('commit_hash', '')[:8]}
Message: {commit2.get('message', '')}
Files: {', '.join(commit2.get('files_changed', [])[:5])}
Author: {commit2.get('author', '')}

Please analyze:
1. What are the key differences between these commits?
2. How do they relate to each other (if at all)?
3. Which commit has a bigger impact and why?
4. What development patterns do they show?

Provide a detailed technical comparison."""
    
    return await query_llm(prompt)

@app.get("/api/commits/indexed")
async def get_indexed_commits():
    """Get list of all indexed commit hashes"""
    indexed = set()
    
    # Get from Qdrant
    if qdrant_client:
        try:
            offset = None
            while True:
                results = qdrant_client.scroll(
                    collection_name=events_collection,
                    limit=100,
                    offset=offset,
                    with_payload=True,
                    with_vectors=False
                )
                
                points, next_offset = results
                
                for point in points:
                    commit_hash = point.payload.get("commit_hash")
                    if commit_hash:
                        indexed.add(commit_hash)
                
                if next_offset is None:
                    break
                offset = next_offset
                
        except Exception as e:
            logger.error(f"Error getting indexed commits: {e}")
    
    # Also add from memory
    for event in git_events:
        commit_hash = event.get("commit_hash")
        if commit_hash:
            indexed.add(commit_hash)
    
    return {
        "total_indexed": len(indexed),
        "commit_hashes": list(indexed)[:100],  # Return first 100
        "source": "qdrant+memory" if qdrant_client else "memory"
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

@app.post("/api/repositories/deduplicate")
async def deduplicate_repositories():
    """Remove duplicate repositories with same name but different paths"""
    global indexed_repositories
    
    seen = {}
    to_remove = []
    
    for path, repo in indexed_repositories.items():
        normalized = normalize_repo_path(path)
        
        if normalized in seen:
            # Keep the one with more commits
            if repo.get("commit_count", 0) > seen[normalized]["count"]:
                to_remove.append(seen[normalized]["path"])
                seen[normalized] = {"path": path, "count": repo.get("commit_count", 0)}
            else:
                to_remove.append(path)
        else:
            seen[normalized] = {"path": path, "count": repo.get("commit_count", 0)}
    
    # Remove duplicates
    for path in to_remove:
        if path in indexed_repositories:
            del indexed_repositories[path]
    
    save_repositories()
    
    return {
        "status": "success",
        "removed": len(to_remove),
        "remaining": len(indexed_repositories)
    }

@app.post("/api/ingest/start")
async def start_ingestion(request: dict):
    """Start ingesting a repository"""
    repo_path = request.get("repo_path")
    force = request.get("force", False)
    # max_commits = request.get("max_commits", 100)
    
    if not repo_path:
        raise HTTPException(status_code=400, detail="repo_path required")
    
    # Normalize the path
    normalized_path = normalize_repo_path(repo_path)
    
    # Check if already exists with different normalization
    for existing_path in list(indexed_repositories.keys()):
        if normalize_repo_path(existing_path) == normalized_path and existing_path != normalized_path:
            # Remove the old entry
            del indexed_repositories[existing_path]

    if normalized_path not in indexed_repositories:
        indexed_repositories[normalized_path] = {
            "path": normalized_path,
            "first_indexed": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat(),
            "commit_count": 0,
            "selected": True  # Auto-select new repositories
        }
        save_repositories()
    
    # Check repository status first
    check_result = await check_repository_status({"repo_path": repo_path})
    
    if check_result["fully_indexed"] and not force:
        return {
            "status": "already_complete",
            "message": f"Repository {repo_path} is already fully indexed",
            "stats": check_result
        }
    
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
                "--api-url", "http://localhost:8000",
                "--skip-duplicates"
            ])
            
            ingestion_status["is_ingesting"] = False
            ingestion_status["last_ingested"] = datetime.now().isoformat()
            save_repositories()
        except Exception as e:
            ingestion_status["is_ingesting"] = False
            ingestion_status["error"] = str(e)
            logger.error(f"Ingestion error: {e}")
    
    thread = threading.Thread(target=run_collector)
    thread.start()
    indexed_repos.add(repo_path)
    
    return {"status": "started", "repo": repo_path, "to_ingest": check_result["missing"],
        "already_indexed": check_result["already_indexed"]}

@app.get("/api/ingest/status")
async def get_ingestion_status():
    """Get current ingestion status"""
    return ingestion_status

@app.post("/api/reset")
async def reset_everything():
    """Complete reset - removes all data"""
    global indexed_repositories, git_events
    
    # Clear memory
    indexed_repositories = {}
    git_events = []
    save_repositories()
    
    # Clear Qdrant completely
    if qdrant_client:
        try:
            qdrant_client.delete_collection(events_collection)
            qdrant_client.create_collection(
                collection_name=events_collection,
                vectors_config=VectorParams(size=384, distance=Distance.COSINE)
            )
        except:
            pass
    
    return {"status": "success", "message": "Complete reset done"}

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
    """Get accurate statistics based on active repositories"""
    stats = {
        "events": {
            "in_memory": len(git_events) + len(context_events),
            "in_qdrant": 0
        },
        "repositories": len(indexed_repositories),  # Show repository count
        "queries_made": 0 # Placeholder for future tracking
    }
    # Calculate total commits only from tracked repositories
    total_commits = 0
    for repo_path, repo_info in indexed_repositories.items():
        total_commits += repo_info.get("commit_count", 0)
    
    if qdrant_client:
        try:
            collection_info = qdrant_client.get_collection(events_collection)
            # Only count if we're showing all repos
            if len(indexed_repositories) > 0:
                stats["events"]["in_qdrant"] = total_commits
            else:
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
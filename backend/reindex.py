#!/usr/bin/env python3
"""
Clean and re-index Context Keeper
Save as: reindex.py
"""
import requests
import subprocess
import json
import time

API_URL = "http://localhost:8000"
REPO_PATH = "D:/projects/sidekick-ai/sidekick-ai"  # Update this path

def pretty_print(data, title=""):
    if title:
        print(f"\n{'='*50}")
        print(f" {title}")
        print('='*50)
    print(json.dumps(data, indent=2))

def main():
    print("🧹 Context Keeper - Clean Re-indexing")
    print("="*50)
    
    # Step 1: Check current status
    print("\n📊 Checking current status...")
    try:
        response = requests.get(f"{API_URL}/api/stats")
        stats = response.json()
        print(f"  Events in memory: {stats['events']['in_memory']}")
        print(f"  Events in Qdrant: {stats['events']['in_qdrant']}")
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Make sure Context Keeper is running!")
        return
    
    # Step 2: Debug what's in there
    print("\n🔍 Checking for bad entries...")
    response = requests.get(f"{API_URL}/api/debug/commits")
    debug_info = response.json()
    
    bad_entries = 0
    for entry in debug_info.get("in_qdrant", []):
        if entry["commit_hash"] == "MISSING":
            bad_entries += 1
    
    if bad_entries > 0:
        print(f"  Found {bad_entries} entries without commit hashes")
    
    # Step 3: Ask user what to do
    print("\n❓ What would you like to do?")
    print("1. Clean up bad entries only")
    print("2. Clear everything and re-index from scratch")
    print("3. Cancel")
    
    choice = input("\nEnter your choice (1-3): ").strip()
    
    if choice == "3":
        print("Cancelled.")
        return
    
    if choice == "1":
        # Clean bad entries
        print("\n🧹 Cleaning bad entries...")
        response = requests.post(f"{API_URL}/api/cleanup")
        result = response.json()
        print(f"  {result['message']}")
        
    elif choice == "2":
        # Clear everything
        print("\n🗑️  Clearing all data...")
        response = requests.delete(f"{API_URL}/api/clear")
        result = response.json()
        print(f"  {result['message']}")
    
    # Step 4: Re-ingest
    print(f"\n📚 Re-ingesting repository: {REPO_PATH}")
    print("This may take a few minutes...")
    
    # Change to collector directory
    collector_path = "D:/projects/context-keeper/collectors/git/git_collector.py"
    
    try:
        # Run the collector
        result = subprocess.run(
            ["python", collector_path, "--repo", REPO_PATH, "--history", "1000", "--api-url", API_URL],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            print("✅ Re-ingestion complete!")
        else:
            print(f"❌ Error during ingestion: {result.stderr}")
    except Exception as e:
        print(f"❌ Failed to run collector: {e}")
        print("\nTry running manually:")
        print(f"cd /d/projects/context-keeper/collectors/git")
        print(f"python git_collector.py --repo {REPO_PATH} --history 1000")
        return
    
    # Step 5: Verify
    print("\n✅ Verifying results...")
    time.sleep(2)  # Give it a moment to process
    
    response = requests.get(f"{API_URL}/api/stats")
    stats = response.json()
    print(f"  Events in memory: {stats['events']['in_memory']}")
    print(f"  Events in Qdrant: {stats['events']['in_qdrant']}")
    
    # Test a query
    print("\n🔍 Testing query...")
    response = requests.post(
        f"{API_URL}/api/query",
        json={"query": "show recent commits"},
        headers={"Content-Type": "application/json"}
    )
    
    if response.status_code == 200:
        result = response.json()
        print(f"  Found {len(result['sources'])} results")
        print("\n  Sample commits:")
        for source in result['sources'][:3]:
            commit = source.get('commit', 'no-hash')
            message = source.get('content', '')[:50]
            print(f"    [{commit}] {message}")
    
    print("\n✨ Complete! Your Context Keeper is now clean and indexed properly.")

if __name__ == "__main__":
    main()
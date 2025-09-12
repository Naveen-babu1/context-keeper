#!/usr/bin/env python3
"""
Git Collector - Captures git events and sends to Context Keeper
"""

# Check if GitPython is available
HAS_GITPYTHON = True
try:
    import git
except ImportError:
    HAS_GITPYTHON = False
    print("⚠️  GitPython not installed. Using subprocess fallback.")
import os
import sys
import json
import time
import subprocess
from pathlib import Path
from datetime import datetime
import requests
import argparse
import logging
from typing import Dict, List, Any
import click


class GitCollector:
    def __init__(self, repo_path= str, api_url= str):
        self.repo_path = Path(repo_path).expanduser().resolve()
        self.api_url = api_url
        # self.repo = git.Repo(self.repo_path)
        self.processed_commits = set()
        
        if not (self.repo_path / ".git").exists():
            print(f"❌ Error: {self.repo_path} is not a git repository")
            sys.exit(1)
        
        print(f"✅ Git repository: {self.repo_path}")
        print(f"📡 API URL: {self.api_url}")

        if HAS_GITPYTHON:
            try:
                self.repo = git.Repo(self.repo_path)
                self.use_gitpython = True
                print(f"✅ Using GitPython for repository: {self.repo_path}")
            except Exception as e:
                print(f"⚠️  GitPython failed, using subprocess: {e}")
                self.use_gitpython = False
        else:
            self.use_gitpython = False
            print(f"✅ Using subprocess for repository: {self.repo_path}")
        
        print(f"📡 API URL: {self.api_url}")
    
    def run_git_command(self, *args):
        """Run a git command and return output"""
        try:
            result = subprocess.run(
                ["git"] + list(args),
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                check=True
            )
            return result.stdout.strip()
        except subprocess.CalledProcessError as e:
            print(f"Git command failed: {e}")
            return None
    
    def get_commits(self, max_count=100):
        """Get recent commits from the repository"""
        format_string = "%H|%an|%ae|%at|%s"
        output = self.run_git_command(
            "log",
            f"--pretty=format:{format_string}",
            f"-{max_count}"
        )
        
        if not output:
            return []
        
        commits = []
        for line in output.split('\n'):
            if line:
                parts = line.split('|')
                if len(parts) >= 5:
                    commit = {
                        "commit_hash": parts[0],
                        "author_name": parts[1],
                        "author": f"{parts[1]} <{parts[2]}>",
                        "timestamp": datetime.fromtimestamp(int(parts[3])).isoformat(),
                        "message": parts[4],
                        "branch": self.run_git_command("rev-parse", "--abbrev-ref", "HEAD") or "unknown"
                    }
                    
                    # Get files changed (simplified)
                    files_output = self.run_git_command("diff-tree", "--no-commit-id", "--name-only", "-r", parts[0])
                    if files_output:
                        commit["files_changed"] = files_output.split('\n')
                    
                    commits.append(commit)
        
        return commits
    
    def send_event(self, event):
        """Send event to Context Keeper API"""
        try:
            response = requests.post(
                f"{self.api_url}/api/ingest/git",
                json=event,
                headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.ConnectionError:
            print(f"❌ Cannot connect to API at {self.api_url}")
            return None
        except Exception as e:
            print(f"❌ Error: {e}")
            return None
    
    def collect_history(self, max_commits=100):
        """Collect and send historical commits"""
        print(f"\n📚 Collecting last {max_commits} commits...")
        commits = self.get_commits(max_commits)
        
        if not commits:
            print("No commits found")
            return
        
        success_count = 0
        for i, commit in enumerate(commits, 1):
            result = self.send_event(commit)
            if result:
                success_count += 1
                print(f"✓ [{i}/{len(commits)}] {commit['commit_hash'][:8]}: {commit['message'][:50]}")
            else:
                print(f"✗ [{i}/{len(commits)}] Failed: {commit['commit_hash'][:8]}")
            
            if i % 10 == 0:
                time.sleep(0.1)
        
        print(f"\n✅ Successfully sent {success_count}/{len(commits)} commits")
        return success_count

def main():
    parser = argparse.ArgumentParser(description='Git Collector for Context Keeper')
    parser.add_argument('--repo', default='.', help='Repository path')
    parser.add_argument('--api-url', default='http://localhost:8000', help='Context Keeper API URL')
    parser.add_argument('--history', type=int, default=50, help='Number of commits to import')
    
    args = parser.parse_args()
    
    collector = GitCollector(args.repo, args.api_url)
    collector.collect_history(args.history)

if __name__ == "__main__":
    main()

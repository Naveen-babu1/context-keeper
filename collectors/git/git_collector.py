# #!/usr/bin/env python3
# """
# Git Collector - Captures git events and sends to Context Keeper
# """

# # Check if GitPython is available
# HAS_GITPYTHON = True
# try:
#     import git
# except ImportError:
#     HAS_GITPYTHON = False
#     print("⚠️  GitPython not installed. Using subprocess fallback.")
# import os
# import sys
# import json
# import time
# import subprocess
# from pathlib import Path
# from datetime import datetime
# import requests
# import argparse
# import logging
# from typing import Dict, List, Any
# import click


# class GitCollector:
#     def __init__(self, repo_path= str, api_url= str):
#         self.repo_path = Path(repo_path).expanduser().resolve()
#         self.api_url = api_url
#         # self.repo = git.Repo(self.repo_path)
#         self.processed_commits = set()
#         self.indexed_commits = set()
        
#         if not (self.repo_path / ".git").exists():
#             print(f"❌ Error: {self.repo_path} is not a git repository")
#             sys.exit(1)
        
#         print(f"✅ Git repository: {self.repo_path}")
#         print(f"📡 API URL: {self.api_url}")

#         if HAS_GITPYTHON:
#             try:
#                 self.repo = git.Repo(self.repo_path)
#                 self.use_gitpython = True
#                 print(f"✅ Using GitPython for repository: {self.repo_path}")
#             except Exception as e:
#                 print(f"⚠️  GitPython failed, using subprocess: {e}")
#                 self.use_gitpython = False
#         else:
#             self.use_gitpython = False
#             print(f"✅ Using subprocess for repository: {self.repo_path}")
        
#         print(f"📡 API URL: {self.api_url}")
#         # Load already indexed commits from Context Keeper
#         self.load_indexed_commits()
    
#     def run_git_command(self, *args):
#         """Run a git command and return output"""
#         try:
#             result = subprocess.run(
#                 ["git"] + list(args),
#                 cwd=self.repo_path,
#                 capture_output=True,
#                 text=True,
#                 check=True
#             )
#             return result.stdout.strip()
#         except subprocess.CalledProcessError as e:
#             print(f"Git command failed: {e}")
#             return None
    
#     def get_commits(self, max_count=100):
#         """Get recent commits from the repository"""
#         format_string = "%H|%an|%ae|%at|%s"
#         output = self.run_git_command(
#             "log",
#             f"--pretty=format:{format_string}",
#             f"-{max_count}"
#         )
        
#         if not output:
#             return []
        
#         commits = []
#         for line in output.split('\n'):
#             if line:
#                 parts = line.split('|')
#                 if len(parts) >= 5:
#                     commit = {
#                         "commit_hash": parts[0],
#                         "author_name": parts[1],
#                         "author": f"{parts[1]} <{parts[2]}>",
#                         "timestamp": datetime.fromtimestamp(int(parts[3])).isoformat(),
#                         "message": parts[4],
#                         "branch": self.run_git_command("rev-parse", "--abbrev-ref", "HEAD") or "unknown"
#                     }
                    
#                     # Get files changed (simplified)
#                     files_output = self.run_git_command("diff-tree", "--no-commit-id", "--name-only", "-r", parts[0])
#                     if files_output:
#                         commit["files_changed"] = files_output.split('\n')
                    
#                     commits.append(commit)
        
#         return commits
    
#     def send_event(self, event):
#         """Send event to Context Keeper API"""
#         try:
#             response = requests.post(
#                 f"{self.api_url}/api/ingest/git",
#                 json=event,
#                 headers={"Content-Type": "application/json"}
#             )
#             response.raise_for_status()
#             return response.json()
#         except requests.exceptions.ConnectionError:
#             print(f"❌ Cannot connect to API at {self.api_url}")
#             return None
#         except Exception as e:
#             print(f"❌ Error: {e}")
#             return None
    
#     def collect_history(self, max_commits=100):
#         """Collect and send historical commits"""
#         print(f"\n📚 Collecting last {max_commits} commits...")
#         commits = self.get_commits(max_commits)
        
#         if not commits:
#             print("No commits found")
#             return
        
#         success_count = 0
#         for i, commit in enumerate(commits, 1):
#             result = self.send_event(commit)
#             if result:
#                 success_count += 1
#                 print(f"✓ [{i}/{len(commits)}] {commit['commit_hash'][:8]}: {commit['message'][:50]}")
#             else:
#                 print(f"✗ [{i}/{len(commits)}] Failed: {commit['commit_hash'][:8]}")
            
#             if i % 10 == 0:
#                 time.sleep(0.1)
        
#         print(f"\n✅ Successfully sent {success_count}/{len(commits)} commits")
#         return success_count

# def main():
#     parser = argparse.ArgumentParser(description='Git Collector for Context Keeper')
#     parser.add_argument('--repo', default='.', help='Repository path')
#     parser.add_argument('--api-url', default='http://localhost:8000', help='Context Keeper API URL')
#     parser.add_argument('--history', type=int, default=50, help='Number of commits to import')
    
#     args = parser.parse_args()
    
#     collector = GitCollector(args.repo, args.api_url)
#     collector.collect_history(args.history)

# if __name__ == "__main__":
#     main()
#!/usr/bin/env python3
"""
Git Collector - Captures git events and sends to Context Keeper
With duplicate prevention and smart collection
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
from typing import Dict, List, Any, Set

class GitCollector:
    def __init__(self, repo_path: str, api_url: str):  # Fixed: Added colons
        self.repo_path = Path(repo_path).expanduser().resolve()
        self.api_url = api_url
        self.processed_commits = set()
        self.indexed_commits = set()
        
        # Check if repository exists
        if not (self.repo_path / ".git").exists():
            print(f"❌ Error: {self.repo_path} is not a git repository")
            sys.exit(1)
        
        # Setup git access method
        if HAS_GITPYTHON:
            try:
                self.repo = git.Repo(self.repo_path)
                self.use_gitpython = True
                print(f"✅ Using GitPython for repository: {self.repo_path}")
            except Exception as e:
                print(f"⚠️  GitPython failed, using subprocess: {e}")
                self.use_gitpython = False
                self.repo = None
        else:
            self.use_gitpython = False
            self.repo = None
            print(f"✅ Using subprocess for repository: {self.repo_path}")
        
        print(f"📡 API URL: {self.api_url}")
        
        # Load already indexed commits from Context Keeper
        self.load_indexed_commits()
    
    def load_indexed_commits(self):
        """Load list of already indexed commits from Context Keeper"""
        print("🔍 Checking for already indexed commits...")
        
        try:
            response = requests.get(f"{self.api_url}/api/commits/indexed", timeout=10)
            if response.status_code == 200:
                data = response.json()
                self.indexed_commits = set(data.get("commit_hashes", []))
                print(f"✅ Found {len(self.indexed_commits)} commits already indexed")
            else:
                print("⚠️  Could not get indexed commits list (endpoint may not exist)")
                print("    Will rely on duplicate detection during ingestion")
        except Exception as e:
            print(f"⚠️  Error loading indexed commits: {e}")
            print("    Will rely on duplicate detection during ingestion")
    
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
    
    def get_all_commit_hashes(self, max_count=10000) -> List[str]:
        """Get all commit hashes from repository"""
        if self.use_gitpython and self.repo:
            hashes = []
            try:
                for commit in self.repo.iter_commits(max_count=max_count):
                    hashes.append(commit.hexsha)
            except Exception as e:
                print(f"Error getting commits with GitPython: {e}")
                return []
            return hashes
        else:
            output = self.run_git_command("log", "--pretty=format:%H", f"-{max_count}")
            if output:
                return output.strip().split('\n')
            return []
    
    def check_repository_status(self):
        """Check how many commits need to be indexed"""
        print(f"\n📊 Analyzing repository status...")
        
        # Get all commit hashes from repo
        all_commits = self.get_all_commit_hashes(10000)
        total = len(all_commits)
        
        # Check which ones are already indexed
        new_commits = [c for c in all_commits if c not in self.indexed_commits]
        already_indexed = total - len(new_commits)
        
        print(f"📈 Repository statistics:")
        print(f"  Total commits found: {total}")
        print(f"  Already indexed: {already_indexed}")
        print(f"  Need to index: {len(new_commits)}")
        
        return {
            "total": total,
            "indexed": already_indexed,
            "new": len(new_commits),
            "new_commits": new_commits
        }
    
    def get_commits_gitpython(self, max_count=100, commit_list=None):
        """Get commits using GitPython"""
        commits = []
        
        if not self.repo:
            return []
        
        try:
            if commit_list:
                # Get specific commits
                for commit_hash in commit_list[:max_count]:
                    try:
                        commit = self.repo.commit(commit_hash)
                        commits.append(self.extract_commit_data_gitpython(commit))
                    except:
                        continue
            else:
                # Get recent commits
                for commit in self.repo.iter_commits(max_count=max_count):
                    commits.append(self.extract_commit_data_gitpython(commit))
        except Exception as e:
            print(f"Error getting commits with GitPython: {e}")
        
        return commits
    
    def extract_commit_data_gitpython(self, commit):
        """Extract data from a GitPython commit object"""
        # Get files changed
        files_changed = []
        try:
            if not commit.parents:
                # First commit - all files are new
                for item in commit.tree.traverse():
                    if item.type == 'blob':
                        files_changed.append(item.path)
            else:
                # Get diff with parent
                for parent in commit.parents[:1]:
                    diffs = commit.diff(parent)
                    for diff in diffs:
                        if diff.a_path:
                            files_changed.append(diff.a_path)
        except:
            pass
        
        return {
            "commit_hash": commit.hexsha,
            "author_name": commit.author.name,
            "author": f"{commit.author.name} <{commit.author.email}>",
            "timestamp": datetime.fromtimestamp(commit.committed_date).isoformat(),
            "message": commit.message.strip(),
            "branch": self.repo.active_branch.name if not self.repo.head.is_detached else "detached",
            "files_changed": list(set(files_changed))[:20]
        }
    
    def get_commits_subprocess(self, max_count=100, commit_list=None):
        """Get commits using subprocess"""
        if commit_list:
            # Get specific commits
            commits = []
            for commit_hash in commit_list[:max_count]:
                commit_data = self.get_single_commit_subprocess(commit_hash)
                if commit_data:
                    commits.append(commit_data)
            return commits
        else:
            # Get recent commits
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
                    parts = line.split('|', 4)
                    if len(parts) >= 5:
                        commit = {
                            "commit_hash": parts[0],
                            "author_name": parts[1],
                            "author": f"{parts[1]} <{parts[2]}>",
                            "timestamp": datetime.fromtimestamp(int(parts[3])).isoformat(),
                            "message": parts[4],
                            "branch": self.run_git_command("rev-parse", "--abbrev-ref", "HEAD") or "unknown"
                        }
                        
                        # Get files changed
                        files_output = self.run_git_command("diff-tree", "--no-commit-id", "--name-only", "-r", parts[0])
                        if files_output:
                            commit["files_changed"] = files_output.split('\n')[:20]
                        else:
                            commit["files_changed"] = []
                        
                        commits.append(commit)
            
            return commits
    
    def get_single_commit_subprocess(self, commit_hash):
        """Get data for a single commit using subprocess"""
        format_string = "%H|%an|%ae|%at|%s"
        output = self.run_git_command("show", "--pretty=format:" + format_string, "--no-patch", commit_hash)
        
        if not output:
            return None
        
        parts = output.split('|', 4)
        if len(parts) >= 5:
            # Get files changed
            files_output = self.run_git_command("diff-tree", "--no-commit-id", "--name-only", "-r", commit_hash)
            files_changed = files_output.split('\n')[:20] if files_output else []
            
            return {
                "commit_hash": parts[0],
                "author_name": parts[1],
                "author": f"{parts[1]} <{parts[2]}>",
                "timestamp": datetime.fromtimestamp(int(parts[3])).isoformat(),
                "message": parts[4],
                "branch": self.run_git_command("rev-parse", "--abbrev-ref", "HEAD") or "unknown",
                "files_changed": files_changed
            }
        return None
    
    def get_commits(self, max_count=100, commit_list=None):
        """Get commits using the best available method"""
        if self.use_gitpython:
            return self.get_commits_gitpython(max_count, commit_list)
        else:
            return self.get_commits_subprocess(max_count, commit_list)
    
    def send_event(self, event):
        """Send event to Context Keeper API"""
        try:
            response = requests.post(
                f"{self.api_url}/api/ingest/git",
                json=event,
                headers={"Content-Type": "application/json"},
                timeout=10
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.ConnectionError:
            print(f"❌ Cannot connect to API at {self.api_url}")
            return None
        except Exception as e:
            print(f"❌ Error: {e}")
            return None
    
    def collect_history(self, max_commits=100, skip_duplicates=True):
        """Collect and send historical commits with duplicate checking"""
        if skip_duplicates:
            # Check repository status first
            status = self.check_repository_status()
            
            if status["new"] == 0:
                print("\n✅ All commits are already indexed! Nothing to do.")
                return 0
            
            print(f"\n📚 Collecting {min(status['new'], max_commits)} new commits...")
            commits_to_process = status["new_commits"][:max_commits]
            
            # Get commit data for new commits only
            commits = self.get_commits(max_count=len(commits_to_process), commit_list=commits_to_process)
        else:
            print(f"\n📚 Collecting last {max_commits} commits (without duplicate check)...")
            commits = self.get_commits(max_commits)
        
        if not commits:
            print("No commits found")
            return 0
        
        success_count = 0
        skip_count = 0
        
        for i, commit in enumerate(commits, 1):
            # Skip if already indexed (double check)
            if skip_duplicates and commit['commit_hash'] in self.indexed_commits:
                skip_count += 1
                print(f"⏩ [{i}/{len(commits)}] Already indexed: {commit['commit_hash'][:8]}")
                continue
            
            result = self.send_event(commit)
            if result:
                if result.get('status') == 'duplicate':
                    skip_count += 1
                    print(f"⏩ [{i}/{len(commits)}] Duplicate: {commit['commit_hash'][:8]}")
                    self.indexed_commits.add(commit['commit_hash'])
                else:
                    success_count += 1
                    msg = commit['message'].split('\n')[0][:50]
                    print(f"✅ [{i}/{len(commits)}] {commit['commit_hash'][:8]}: {msg}")
                    self.indexed_commits.add(commit['commit_hash'])
                    self.processed_commits.add(commit['commit_hash'])
            else:
                print(f"❌ [{i}/{len(commits)}] Failed: {commit['commit_hash'][:8]}")
            
            # Progress and rate limiting
            if i % 10 == 0:
                time.sleep(0.1)
                if len(commits) > 50:
                    print(f"    Progress: {i}/{len(commits)} ({i*100//len(commits)}%)")
        
        print(f"\n📊 Collection Summary:")
        print(f"  ✅ Successfully indexed: {success_count}")
        print(f"  ⏩ Skipped (duplicates): {skip_count}")
        print(f"  📈 Total processed: {len(commits)}")
        
        return success_count

def main():
    parser = argparse.ArgumentParser(description='Git Collector for Context Keeper')
    parser.add_argument('--repo', default='.', help='Repository path')
    parser.add_argument('--api-url', default='http://localhost:8000', help='Context Keeper API URL')
    parser.add_argument('--history', type=int, default=50, help='Number of commits to import')
    parser.add_argument('--skip-duplicates', action='store_true', default=True, help='Skip already indexed commits')
    parser.add_argument('--force', action='store_true', help='Force re-index all commits (ignore duplicates)')
    parser.add_argument('--check-only', action='store_true', help='Only check status, don\'t index')
    
    args = parser.parse_args()
    
    # Check API health first
    try:
        response = requests.get(f"{args.api_url}/health", timeout=5)
        if response.status_code != 200:
            print(f"⚠️  Context Keeper API returned status {response.status_code}")
    except Exception as e:
        print(f"❌ Cannot connect to Context Keeper at {args.api_url}")
        print(f"   Error: {e}")
        print("\nMake sure Context Keeper is running:")
        print("  cd /d/projects/context-keeper/backend")
        print("  python app/main.py")
        sys.exit(1)
    
    # Create collector
    collector = GitCollector(args.repo, args.api_url)
    
    if args.check_only:
        # Just check status
        collector.check_repository_status()
        print("\n✅ Check complete. No changes made.")
    else:
        # Determine skip_duplicates setting
        skip_duplicates = not args.force and args.skip_duplicates
        
        if args.force:
            print("🔄 Force mode: Re-indexing all commits regardless of duplicates")
        
        # Run collection
        collector.collect_history(args.history, skip_duplicates=skip_duplicates)

if __name__ == "__main__":
    main()
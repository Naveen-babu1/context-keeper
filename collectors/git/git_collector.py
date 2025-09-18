# #!/usr/bin/env python3
# """
# Git Collector - Windows Compatible Version
# Captures git events and sends to Context Keeper
# """

# # Check if GitPython is available
# HAS_GITPYTHON = True
# try:
#     import git
# except ImportError:
#     HAS_GITPYTHON = False
#     # Use plain text instead of Unicode emoji for Windows compatibility
#     print("WARNING: GitPython not installed. Using subprocess fallback.")

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
# from typing import Dict, List, Any, Set

# class GitCollector:
#     def __init__(self, repo_path: str, api_url: str):
#         self.repo_path = Path(repo_path).expanduser().resolve()
#         self.api_url = api_url
#         self.processed_commits = set()
#         self.indexed_commits = set()
        
#         # Check if repository exists
#         if not (self.repo_path / ".git").exists():
#             print(f"ERROR: {self.repo_path} is not a git repository")
#             sys.exit(1)
        
#         # Setup git access method
#         if HAS_GITPYTHON:
#             try:
#                 self.repo = git.Repo(self.repo_path)
#                 self.use_gitpython = True
#                 print(f"OK: Using GitPython for repository: {self.repo_path}")
#             except Exception as e:
#                 print(f"WARNING: GitPython failed, using subprocess: {e}")
#                 self.use_gitpython = False
#                 self.repo = None
#         else:
#             self.use_gitpython = False
#             self.repo = None
#             print(f"OK: Using subprocess for repository: {self.repo_path}")
        
#         print(f"API URL: {self.api_url}")
        
#         # Load already indexed commits from Context Keeper
#         self.load_indexed_commits()

#     def normalize_repo_path(self, path):
#         """Normalize repository path"""
#         normalized = str(Path(path).resolve())
#         normalized = normalized.rstrip('/\\')
#         if os.name == 'nt':  # Windows
#             normalized = normalized.lower()
#         return normalized
    
#     def load_indexed_commits(self):
#         """Load list of already indexed commits from Context Keeper"""
#         print("INFO: Checking for already indexed commits...")
        
#         try:
#             response = requests.get(f"{self.api_url}/api/commits/indexed", timeout=10)
#             if response.status_code == 200:
#                 data = response.json()
#                 self.indexed_commits = set(data.get("commit_hashes", []))
#                 print(f"OK: Found {len(self.indexed_commits)} commits already indexed")
#             else:
#                 print("WARNING: Could not get indexed commits list (endpoint may not exist)")
#                 print("    Will rely on duplicate detection during ingestion")
#         except Exception as e:
#             print(f"WARNING: Error loading indexed commits: {e}")
#             print("    Will rely on duplicate detection during ingestion")
    
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
    
#     def get_all_commit_hashes(self, max_count=10000) -> List[str]:
#         """Get all commit hashes from repository"""
#         if self.use_gitpython and self.repo:
#             hashes = []
#             try:
#                 for commit in self.repo.iter_commits(max_count=max_count):
#                     hashes.append(commit.hexsha)
#             except Exception as e:
#                 print(f"Error getting commits with GitPython: {e}")
#                 return []
#             return hashes
#         else:
#             output = self.run_git_command("log", "--pretty=format:%H", f"-{max_count}")
#             if output:
#                 return output.strip().split('\n')
#             return []
    
#     def check_repository_status(self):
#         """Check how many commits need to be indexed"""
#         print(f"\nINFO: Analyzing repository status...")
        
#         # Get all commit hashes from repo
#         all_commits = self.get_all_commit_hashes(10000)
#         total = len(all_commits)
        
#         # Check which ones are already indexed
#         new_commits = [c for c in all_commits if c not in self.indexed_commits]
#         already_indexed = total - len(new_commits)
        
#         print(f"Repository statistics:")
#         print(f"  Total commits found: {total}")
#         print(f"  Already indexed: {already_indexed}")
#         print(f"  Need to index: {len(new_commits)}")
        
#         return {
#             "total": total,
#             "indexed": already_indexed,
#             "new": len(new_commits),
#             "new_commits": new_commits
#         }
    
#     def get_commits_subprocess(self, repo_path, max_count=100, commit_list=None):
#         """Get commits using subprocess"""
#         repo_path = self.normalize_repo_path(repo_path)
#         if commit_list:
#             # Get specific commits
#             commits = []
#             for commit_hash in commit_list[:max_count]:
#                 commit_data = self.get_single_commit_subprocess(commit_hash)
#                 if commit_data:
#                     commits.append(commit_data)
#             return commits
#         else:
#             # Get recent commits
#             format_string = "%H|%an|%ae|%at|%s"
#             output = self.run_git_command(
#                 "log",
#                 f"--pretty=format:{format_string}",
#                 f"-{max_count}"
#             )
            
#             if not output:
#                 return []
            
#             commits = []
#             for line in output.split('\n'):
#                 if line:
#                     parts = line.split('|', 4)
#                     if len(parts) >= 5:
#                         commit = {
#                             "commit_hash": parts[0],
#                             "author_name": parts[1],
#                             "author": f"{parts[1]} <{parts[2]}>",
#                             "timestamp": datetime.fromtimestamp(int(parts[3])).isoformat(),
#                             "message": parts[4],
#                             "branch": self.run_git_command("rev-parse", "--abbrev-ref", "HEAD") or "unknown",
#                             "repository": str(self.repo_path) 
#                         }
                        
#                         # Get files changed
#                         files_output = self.run_git_command("diff-tree", "--no-commit-id", "--name-only", "-r", parts[0])
#                         if files_output:
#                             commit["files_changed"] = files_output.split('\n')[:20]
#                         else:
#                             commit["files_changed"] = []
                        
#                         commits.append(commit)
            
#             return commits
    
#     def get_single_commit_subprocess(self, commit_hash):
#         """Get data for a single commit using subprocess"""
#         format_string = "%H|%an|%ae|%at|%s"
#         output = self.run_git_command("show", "--pretty=format:" + format_string, "--no-patch", commit_hash)
        
#         if not output:
#             return None
        
#         parts = output.split('|', 4)
#         if len(parts) >= 5:
#             # Get files changed
#             files_output = self.run_git_command("diff-tree", "--no-commit-id", "--name-only", "-r", commit_hash)
#             files_changed = files_output.split('\n')[:20] if files_output else []
            
#             return {
#                 "commit_hash": parts[0],
#                 "author_name": parts[1],
#                 "author": f"{parts[1]} <{parts[2]}>",
#                 "timestamp": datetime.fromtimestamp(int(parts[3])).isoformat(),
#                 "message": parts[4],
#                 "branch": self.run_git_command("rev-parse", "--abbrev-ref", "HEAD") or "unknown",
#                 "files_changed": files_changed,
#                 "repository": str(self.repo_path) 
#             }
#         return None
    
#     def get_commits(self, max_count=100, commit_list=None):
#         """Get commits using the best available method"""
#         # Always use subprocess for Windows compatibility
#         return self.get_commits_subprocess(self.repo_path, max_count, commit_list)
    
#     def send_event(self, event):
#         """Send event to Context Keeper API"""
#         try:
#             response = requests.post(
#                 f"{self.api_url}/api/ingest/git",
#                 json=event,
#                 headers={"Content-Type": "application/json"},
#                 timeout=10
#             )
#             response.raise_for_status()
#             return response.json()
#         except requests.exceptions.ConnectionError:
#             print(f"ERROR: Cannot connect to API at {self.api_url}")
#             return None
#         except Exception as e:
#             print(f"ERROR: {e}")
#             return None
    
#     def collect_history(self, max_commits=100, skip_duplicates=True):
#         """Collect and send historical commits with duplicate checking"""
#         if skip_duplicates:
#             # Check repository status first
#             status = self.check_repository_status()
            
#             if status["new"] == 0:
#                 print("\nOK: All commits are already indexed! Nothing to do.")
#                 return 0
            
#             print(f"\nINFO: Collecting {min(status['new'], max_commits)} new commits...")
#             commits_to_process = status["new_commits"][:max_commits]
            
#             # Get commit data for new commits only
#             commits = self.get_commits(max_count=len(commits_to_process), commit_list=commits_to_process)
#         else:
#             print(f"\nINFO: Collecting last {max_commits} commits (without duplicate check)...")
#             commits = self.get_commits(max_commits)
        
#         if not commits:
#             print("No commits found")
#             return 0
        
#         success_count = 0
#         skip_count = 0
        
#         for i, commit in enumerate(commits, 1):
#             # Skip if already indexed (double check)
#             if skip_duplicates and commit['commit_hash'] in self.indexed_commits:
#                 skip_count += 1
#                 print(f"SKIP [{i}/{len(commits)}] Already indexed: {commit['commit_hash'][:8]}")
#                 continue
            
#             result = self.send_event(commit)
#             if result:
#                 if result.get('status') == 'duplicate':
#                     skip_count += 1
#                     print(f"SKIP [{i}/{len(commits)}] Duplicate: {commit['commit_hash'][:8]}")
#                     self.indexed_commits.add(commit['commit_hash'])
#                 else:
#                     success_count += 1
#                     msg = commit['message'].split('\n')[0][:50]
#                     print(f"OK [{i}/{len(commits)}] {commit['commit_hash'][:8]}: {msg}")
#                     self.indexed_commits.add(commit['commit_hash'])
#                     self.processed_commits.add(commit['commit_hash'])
#             else:
#                 print(f"ERROR [{i}/{len(commits)}] Failed: {commit['commit_hash'][:8]}")
            
#             # Progress and rate limiting
#             if i % 10 == 0:
#                 time.sleep(0.1)
#                 if len(commits) > 50:
#                     print(f"    Progress: {i}/{len(commits)} ({i*100//len(commits)}%)")
        
#         print(f"\nCollection Summary:")
#         print(f"  Successfully indexed: {success_count}")
#         print(f"  Skipped (duplicates): {skip_count}")
#         print(f"  Total processed: {len(commits)}")
        
#         return success_count

# def main():
#     parser = argparse.ArgumentParser(description='Git Collector for Context Keeper')
#     parser.add_argument('--repo', default='.', help='Repository path')
#     parser.add_argument('--api-url', default='http://localhost:8000', help='Context Keeper API URL')
#     parser.add_argument('--history', type=int, default=50, help='Number of commits to import')
#     parser.add_argument('--skip-duplicates', action='store_true', default=True, help='Skip already indexed commits')
#     parser.add_argument('--force', action='store_true', help='Force re-index all commits (ignore duplicates)')
#     parser.add_argument('--check-only', action='store_true', help='Only check status, don\'t index')
    
#     args = parser.parse_args()
    
#     # Check API health first
#     try:
#         response = requests.get(f"{args.api_url}/health", timeout=5)
#         if response.status_code != 200:
#             print(f"WARNING: Context Keeper API returned status {response.status_code}")
#     except Exception as e:
#         print(f"ERROR: Cannot connect to Context Keeper at {args.api_url}")
#         print(f"   Error: {e}")
#         print("\nMake sure Context Keeper is running:")
#         print("  cd /d/projects/context-keeper/backend")
#         print("  python app/main.py")
#         sys.exit(1)
    
#     # Create collector
#     collector = GitCollector(args.repo, args.api_url)
    
#     if args.check_only:
#         # Just check status
#         collector.check_repository_status()
#         print("\nOK: Check complete. No changes made.")
#     else:
#         # Determine skip_duplicates setting
#         skip_duplicates = not args.force and args.skip_duplicates
        
#         if args.force:
#             print("INFO: Force mode: Re-indexing all commits regardless of duplicates")
        
#         # Run collection
#         collector.collect_history(args.history, skip_duplicates=skip_duplicates)

# if __name__ == "__main__":
#     main()

#!/usr/bin/env python3
"""
Multi-Branch Git Collector - Enhanced Version
Captures git events from multiple branches and sends to Context Keeper
"""

# Check if GitPython is available
HAS_GITPYTHON = True
try:
    import git
except ImportError:
    HAS_GITPYTHON = False
    print("WARNING: GitPython not installed. Using subprocess fallback.")

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
from collections import defaultdict

class MultiBranchGitCollector:
    def __init__(self, repo_path: str, api_url: str, branches: List[str] = None):
        self.repo_path = Path(repo_path).expanduser().resolve()
        self.api_url = api_url
        self.processed_commits = set()
        self.indexed_commits = set()
        self.target_branches = branches or []
        self.branch_commit_map = {}
        
        # Check if repository exists
        if not (self.repo_path / ".git").exists():
            print(f"ERROR: {self.repo_path} is not a git repository")
            sys.exit(1)
        
        # Setup git access method
        if HAS_GITPYTHON:
            try:
                self.repo = git.Repo(self.repo_path)
                self.use_gitpython = True
                print(f"OK: Using GitPython for repository: {self.repo_path}")
            except Exception as e:
                print(f"WARNING: GitPython failed, using subprocess: {e}")
                self.use_gitpython = False
                self.repo = None
        else:
            self.use_gitpython = False
            self.repo = None
            print(f"OK: Using subprocess for repository: {self.repo_path}")
        
        print(f"API URL: {self.api_url}")
        
        # Auto-discover branches if none specified
        if not self.target_branches:
            self.target_branches = self.get_all_branches()
            print(f"INFO: Auto-discovered {len(self.target_branches)} branches")
        else:
            print(f"INFO: Targeting {len(self.target_branches)} specified branches")
        
        # Load already indexed commits from Context Keeper
        self.load_indexed_commits()

    def normalize_repo_path(self, path):
        """Normalize repository path"""
        normalized = str(Path(path).resolve())
        normalized = normalized.rstrip('/\\')
        if os.name == 'nt':  # Windows
            normalized = normalized.lower()
        return normalized
    
    def get_all_branches(self) -> List[str]:
        """Get all branches in the repository"""
        if self.use_gitpython and self.repo:
            try:
                branches = []
                # Get local branches
                for branch in self.repo.branches:
                    branches.append(str(branch))
                # Get remote branches
                for remote in self.repo.remotes:
                    for ref in remote.refs:
                        branch_name = ref.name.replace(f"{remote.name}/", "")
                        if branch_name != "HEAD" and branch_name not in branches:
                            branches.append(branch_name)
                return branches
            except Exception as e:
                print(f"WARNING: Error getting branches with GitPython: {e}")
        
        # Fallback to subprocess
        try:
            output = self.run_git_command("branch", "-a")
            if output:
                branches = []
                for line in output.split('\n'):
                    branch = line.strip().replace('* ', '').replace('remotes/origin/', '')
                    if branch and branch != 'HEAD' and '->' not in branch:
                        branches.append(branch)
                return list(set(branches))
        except:
            pass
        
        # Ultimate fallback
        return ['main', 'master', 'develop']
    
    def get_active_branches(self, days: int = 30) -> List[str]:
        """Get branches with recent activity"""
        cutoff_date = datetime.now().timestamp() - (days * 24 * 60 * 60)
        active_branches = []
        
        for branch in self.get_all_branches():
            try:
                # Get the last commit date for this branch
                if self.use_gitpython and self.repo:
                    try:
                        last_commit = self.repo.commit(branch)
                        if last_commit.committed_date > cutoff_date:
                            active_branches.append(branch)
                    except:
                        continue
                else:
                    # Subprocess fallback
                    output = self.run_git_command("log", "-1", "--format=%ct", branch)
                    if output:
                        commit_date = int(output.strip())
                        if commit_date > cutoff_date:
                            active_branches.append(branch)
            except:
                continue
        
        return active_branches
    
    def load_indexed_commits(self):
        """Load list of already indexed commits from Context Keeper"""
        print("INFO: Checking for already indexed commits...")
        
        try:
            response = requests.get(f"{self.api_url}/api/commits/indexed", timeout=10)
            if response.status_code == 200:
                data = response.json()
                self.indexed_commits = set(data.get("commit_hashes", []))
                print(f"OK: Found {len(self.indexed_commits)} commits already indexed")
            else:
                print("WARNING: Could not get indexed commits list")
        except Exception as e:
            print(f"WARNING: Error loading indexed commits: {e}")
    
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
    
    def get_commit_branches(self, commit_hash: str) -> List[str]:
        """Get all branches that contain a specific commit"""
        if self.use_gitpython and self.repo:
            try:
                branches = []
                for branch in self.repo.branches:
                    try:
                        if self.repo.is_ancestor(commit_hash, branch.commit):
                            branches.append(str(branch))
                    except:
                        continue
                return branches
            except:
                pass
        
        # Subprocess fallback
        try:
            output = self.run_git_command("branch", "--contains", commit_hash)
            if output:
                branches = []
                for line in output.split('\n'):
                    branch = line.strip().replace('* ', '')
                    if branch and not branch.startswith('('):
                        branches.append(branch)
                return branches
        except:
            pass
        return []
    
    def get_parent_commits(self, commit_hash: str) -> List[str]:
        """Get parent commit hashes"""
        if self.use_gitpython and self.repo:
            try:
                commit = self.repo.commit(commit_hash)
                return [str(parent) for parent in commit.parents]
            except:
                pass
        
        try:
            output = self.run_git_command("rev-list", "--parents", "-n", "1", commit_hash)
            if output:
                parts = output.strip().split()
                return parts[1:] if len(parts) > 1 else []
        except:
            pass
        return []
    
    def classify_branch_type(self, branch_name: str) -> str:
        """Classify the type of branch"""
        branch_lower = branch_name.lower()
        
        if branch_lower in ['main', 'master']:
            return 'main'
        elif branch_lower in ['develop', 'dev', 'development']:
            return 'develop'
        elif branch_lower.startswith('feature/'):
            return 'feature'
        elif branch_lower.startswith('hotfix/'):
            return 'hotfix'
        elif branch_lower.startswith('release/'):
            return 'release'
        elif branch_lower.startswith('bugfix/'):
            return 'bugfix'
        else:
            return 'other'
    
    def enhance_commit_with_branch_info(self, commit: dict, current_branch: str) -> dict:
        """Add branch information to commit data"""
        commit_hash = commit.get('commit_hash', '')
        
        if commit_hash:
            # Get all branches containing this commit
            all_branches = self.get_commit_branches(commit_hash)
            if current_branch not in all_branches:
                all_branches.append(current_branch)
            
            commit['all_branches'] = all_branches
            
            # Determine primary branch (prefer main/master, then develop, then current)
            primary_branch = current_branch
            if 'main' in all_branches:
                primary_branch = 'main'
            elif 'master' in all_branches:
                primary_branch = 'master'
            elif 'develop' in all_branches:
                primary_branch = 'develop'
            
            commit['primary_branch'] = primary_branch
            commit['current_branch'] = current_branch
            
            # Get parent commits and determine if it's a merge
            parent_commits = self.get_parent_commits(commit_hash)
            commit['parent_commits'] = parent_commits
            commit['merge_commit'] = len(parent_commits) > 1
            
            # Add branch context
            commit['branch_context'] = {
                'branch_type': self.classify_branch_type(current_branch),
                'is_main_branch': primary_branch in ['main', 'master'],
                'is_feature_branch': current_branch.startswith('feature/'),
                'branch_count': len(all_branches)
            }
        
        return commit
    
    def get_branch_commits(self, branch: str, max_count: int = 1000) -> List[dict]:
        """Get commits for a specific branch"""
        try:
            format_string = "%H|%an|%ae|%at|%s"
            output = self.run_git_command(
                "log",
                f"--pretty=format:{format_string}",
                f"-{max_count}",
                branch
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
                            "branch": branch,
                            "repository": str(self.repo_path)
                        }
                        
                        # Get files changed
                        files_output = self.run_git_command("diff-tree", "--no-commit-id", "--name-only", "-r", parts[0])
                        if files_output:
                            commit["files_changed"] = files_output.split('\n')[:20]
                        else:
                            commit["files_changed"] = []
                        
                        # Enhance with branch information
                        enhanced_commit = self.enhance_commit_with_branch_info(commit, branch)
                        commits.append(enhanced_commit)
            
            return commits
        except Exception as e:
            print(f"ERROR: Failed to get commits for branch {branch}: {e}")
            return []
    
    def collect_multi_branch_history(self, max_commits_per_branch: int = 100):
        """Collect commits from multiple branches"""
        print(f"INFO: Collecting from {len(self.target_branches)} branches...")
        
        all_commits = {}  # Use dict to avoid duplicates by commit hash
        branch_stats = {}
        
        for branch in self.target_branches:
            print(f"INFO: Processing branch: {branch}")
            
            try:
                # Get commits for this branch
                branch_commits = self.get_branch_commits(branch, max_commits_per_branch)
                
                branch_stats[branch] = {
                    'total': len(branch_commits),
                    'new': 0,
                    'skipped': 0
                }
                
                for commit in branch_commits:
                    commit_hash = commit.get('commit_hash')
                    
                    if commit_hash and commit_hash not in all_commits:
                        # New commit
                        all_commits[commit_hash] = commit
                        branch_stats[branch]['new'] += 1
                    elif commit_hash in all_commits:
                        # Update existing commit with additional branch info
                        existing = all_commits[commit_hash]
                        existing_branches = set(existing.get('all_branches', []))
                        new_branches = set(commit.get('all_branches', []))
                        combined_branches = list(existing_branches.union(new_branches))
                        existing['all_branches'] = combined_branches
                        branch_stats[branch]['skipped'] += 1
                
            except Exception as e:
                print(f"ERROR: Failed to process branch {branch}: {e}")
                branch_stats[branch] = {'total': 0, 'new': 0, 'skipped': 0, 'error': str(e)}
        
        # Print branch statistics
        print(f"\nBranch Processing Summary:")
        for branch, stats in branch_stats.items():
            if 'error' in stats:
                print(f"  {branch}: ERROR - {stats['error']}")
            else:
                print(f"  {branch}: {stats['total']} commits ({stats['new']} new, {stats['skipped']} duplicates)")
        
        return list(all_commits.values())
    
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
            print(f"ERROR: Cannot connect to API at {self.api_url}")
            return None
        except Exception as e:
            print(f"ERROR: {e}")
            return None
    
    def collect_and_send(self, max_commits_per_branch: int = 100, skip_duplicates: bool = True):
        """Collect commits from all branches and send to Context Keeper"""
        # Collect commits from all branches
        all_commits = self.collect_multi_branch_history(max_commits_per_branch)
        
        if not all_commits:
            print("No commits found across all branches")
            return 0
        
        # Filter out already indexed commits if requested
        if skip_duplicates:
            new_commits = [c for c in all_commits if c.get('commit_hash') not in self.indexed_commits]
            print(f"INFO: Found {len(all_commits)} total commits, {len(new_commits)} are new")
            all_commits = new_commits
        
        if not all_commits:
            print("OK: All commits are already indexed!")
            return 0
        
        # Sort commits by timestamp (oldest first for better context)
        all_commits.sort(key=lambda x: x.get('timestamp', ''))
        
        # Send commits to Context Keeper
        success_count = 0
        skip_count = 0
        
        print(f"\nINFO: Sending {len(all_commits)} commits to Context Keeper...")
        
        for i, commit in enumerate(all_commits, 1):
            commit_hash = commit.get('commit_hash', '')[:8]
            message = commit.get('message', '')[:50]
            branch = commit.get('current_branch', 'unknown')
            
            result = self.send_event(commit)
            if result:
                if result.get('status') == 'duplicate':
                    skip_count += 1
                    print(f"SKIP [{i}/{len(all_commits)}] {commit_hash} ({branch}): Duplicate")
                elif result.get('status') == 'success':
                    success_count += 1
                    print(f"OK [{i}/{len(all_commits)}] {commit_hash} ({branch}): {message}")
                else:
                    print(f"WARN [{i}/{len(all_commits)}] {commit_hash} ({branch}): {result.get('message', 'Unknown status')}")
            else:
                print(f"ERROR [{i}/{len(all_commits)}] {commit_hash} ({branch}): Failed to send")
            
            # Rate limiting
            if i % 10 == 0:
                time.sleep(0.1)
                if len(all_commits) > 50:
                    print(f"    Progress: {i}/{len(all_commits)} ({i*100//len(all_commits)}%)")
        
        print(f"\nCollection Summary:")
        print(f"  Successfully indexed: {success_count}")
        print(f"  Skipped (duplicates): {skip_count}")
        print(f"  Total processed: {len(all_commits)}")
        print(f"  Branches processed: {len(self.target_branches)}")
        
        return success_count

def main():
    parser = argparse.ArgumentParser(description='Multi-Branch Git Collector for Context Keeper')
    parser.add_argument('--repo', default='.', help='Repository path')
    parser.add_argument('--api-url', default='http://localhost:8000', help='Context Keeper API URL')
    parser.add_argument('--history', type=int, default=100, help='Number of commits to import per branch')
    parser.add_argument('--branches', help='Comma-separated list of branches (e.g., main,develop,feature/auth)')
    parser.add_argument('--all-branches', action='store_true', help='Collect from all branches')
    parser.add_argument('--active-branches', action='store_true', help='Collect from recently active branches only')
    parser.add_argument('--days', type=int, default=30, help='Days to look back for active branches')
    parser.add_argument('--skip-duplicates', action='store_true', default=True, help='Skip already indexed commits')
    parser.add_argument('--force', action='store_true', help='Force re-index all commits')
    parser.add_argument('--list-branches', action='store_true', help='List all branches and exit')
    
    args = parser.parse_args()
    
    # Check API health first
    try:
        response = requests.get(f"{args.api_url}/health", timeout=5)
        if response.status_code != 200:
            print(f"WARNING: Context Keeper API returned status {response.status_code}")
    except Exception as e:
        print(f"ERROR: Cannot connect to Context Keeper at {args.api_url}")
        print(f"   Error: {e}")
        sys.exit(1)
    
    # Determine target branches
    target_branches = None
    if args.branches:
        target_branches = [b.strip() for b in args.branches.split(',')]
    elif args.active_branches:
        # Will be determined by collector
        pass
    elif args.all_branches:
        # Will be determined by collector
        pass
    else:
        # Default to main branches
        target_branches = ['main', 'master', 'develop']
    
    # Create collector
    collector = MultiBranchGitCollector(args.repo, args.api_url, target_branches)
    
    # Handle special operations
    if args.list_branches:
        print("Available branches:")
        for branch in collector.get_all_branches():
            print(f"  - {branch}")
        return
    
    # Update target branches based on options
    if args.active_branches and not target_branches:
        collector.target_branches = collector.get_active_branches(args.days)
        print(f"INFO: Found {len(collector.target_branches)} active branches in last {args.days} days")
    elif not target_branches and args.all_branches:
        collector.target_branches = collector.get_all_branches()
        print(f"INFO: Processing all {len(collector.target_branches)} branches")
    
    if not collector.target_branches:
        print("ERROR: No branches to process")
        sys.exit(1)
    
    # Determine skip_duplicates setting
    skip_duplicates = not args.force and args.skip_duplicates
    
    if args.force:
        print("INFO: Force mode: Re-indexing all commits regardless of duplicates")
    
    # Run collection
    collector.collect_and_send(args.history, skip_duplicates)

if __name__ == "__main__":
    main()
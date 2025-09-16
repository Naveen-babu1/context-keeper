#!/usr/bin/env python3
"""
Install Context Keeper git hooks in repositories
"""
import os
import sys
import json
from pathlib import Path
import stat
import subprocess

HOOK_TEMPLATE = '''#!/usr/bin/env python3
"""
Context Keeper Auto-Indexing Hook
Automatically sends new commits to Context Keeper
"""
import subprocess
import sys
import os
from pathlib import Path

# Configuration
CONTEXT_KEEPER_PATH = "{context_keeper_path}"
API_URL = "{api_url}"

def index_commit():
    try:
        # Get the latest commit hash
        result = subprocess.run(
            ["git", "log", "-1", "--format=%H"],
            capture_output=True,
            text=True,
            cwd=os.getcwd()
        )
        commit_hash = result.stdout.strip()
        
        # Run the git collector for just this commit
        collector_script = Path(CONTEXT_KEEPER_PATH) / "collectors" / "git" / "git_collector.py"
        
        result = subprocess.run([
            sys.executable,
            str(collector_script),
            "--repo", os.getcwd(),
            "--history", "1",
            "--api-url", API_URL
        ], capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0:
            print("Context Keeper: Indexed commit " + commit_hash[:8])
        else:
            print("Context Keeper: Failed to index (service might be down)")
            
    except subprocess.TimeoutExpired:
        print("Context Keeper: Timeout (service might be down)")
    except Exception as e:
        print("Context Keeper: Error - " + str(e))

if __name__ == "__main__":
    # Don't block the commit if Context Keeper fails
    try:
        index_commit()
    except:
        pass
    sys.exit(0)
'''

class HookInstaller:
    def __init__(self, context_keeper_path=None, api_url="http://localhost:8000"):
        self.context_keeper_path = context_keeper_path or str(Path(__file__).parent.parent.parent)
        self.api_url = api_url
        self.repositories_file = Path(self.context_keeper_path) / "backend" / "data" / "repositories.json"
        
    def load_repositories(self):
        """Load tracked repositories from Context Keeper"""
        if not self.repositories_file.exists():
            print("[ERROR] No repositories found. Add repositories through Context Keeper first.")
            return []
        
        with open(self.repositories_file, 'r') as f:
            repos = json.load(f)
            return list(repos.keys())
    
    def install_hook(self, repo_path):
        """Install post-commit hook in a single repository"""
        repo_path = Path(repo_path)
        
        if not (repo_path / ".git").exists():
            print(f"[ERROR] {repo_path} is not a git repository")
            return False
        
        hooks_dir = repo_path / ".git" / "hooks"
        hooks_dir.mkdir(exist_ok=True)
        
        # Create post-commit hook
        hook_path = hooks_dir / "post-commit"
        
        # Generate hook content with paths
        hook_content = HOOK_TEMPLATE.format(
            context_keeper_path=self.context_keeper_path.replace('\\', '/'),
            api_url=self.api_url
        )
        
        # Check if hook already exists
        if hook_path.exists():
            with open(hook_path, 'r', encoding='utf-8') as f:
                existing = f.read()
                if "Context Keeper" in existing:
                    print(f"[OK] Hook already installed in {repo_path}")
                    return True
                else:
                    # Backup existing hook
                    backup_path = hooks_dir / "post-commit.backup"
                    with open(backup_path, 'w', encoding='utf-8') as backup:
                        backup.write(existing)
                    print(f"[BACKUP] Backed up existing hook to {backup_path}")
        
        # Write new hook with explicit encoding
        with open(hook_path, 'w', encoding='utf-8', newline='\n') as f:
            f.write(hook_content)
        
        # Make executable on Unix systems
        if os.name != 'nt':
            st = os.stat(hook_path)
            os.chmod(hook_path, st.st_mode | stat.S_IEXEC)
        
        print(f"[OK] Installed hook in {repo_path}")
        
        # Test the hook
        self.test_hook(repo_path)
        
        return True
    
    def test_hook(self, repo_path):
        """Test if the hook works"""
        hook_path = Path(repo_path) / ".git" / "hooks" / "post-commit"
        if hook_path.exists():
            try:
                result = subprocess.run(
                    [sys.executable, str(hook_path)],
                    capture_output=True,
                    text=True,
                    timeout=5,
                    cwd=repo_path
                )
                if "Context Keeper" in result.stdout or "Context Keeper" in result.stderr:
                    print("   [TEST] Hook test successful")
                else:
                    print("   [WARNING] Hook installed but test unclear")
            except Exception as e:
                print(f"   [WARNING] Hook test failed: {e}")
    
    def uninstall_hook(self, repo_path):
        """Remove Context Keeper hook from repository"""
        hook_path = Path(repo_path) / ".git" / "hooks" / "post-commit"
        
        if hook_path.exists():
            with open(hook_path, 'r', encoding='utf-8') as f:
                content = f.read()
                if "Context Keeper" in content:
                    # Check for backup
                    backup_path = Path(repo_path) / ".git" / "hooks" / "post-commit.backup"
                    if backup_path.exists():
                        # Restore backup
                        with open(backup_path, 'r', encoding='utf-8') as backup:
                            with open(hook_path, 'w', encoding='utf-8') as hook:
                                hook.write(backup.read())
                        backup_path.unlink()
                        print(f"[OK] Restored original hook in {repo_path}")
                    else:
                        # Just remove
                        hook_path.unlink()
                        print(f"[OK] Removed hook from {repo_path}")
                else:
                    print(f"[WARNING] No Context Keeper hook found in {repo_path}")
        else:
            print(f"[WARNING] No post-commit hook found in {repo_path}")
    
    def install_all(self):
        """Install hooks in all tracked repositories"""
        repos = self.load_repositories()
        
        if not repos:
            return
        
        print(f"\n[INFO] Found {len(repos)} tracked repositories")
        print("="*50)
        
        success = 0
        for repo_path in repos:
            if self.install_hook(repo_path):
                success += 1
            print()
        
        print("="*50)
        print(f"[OK] Successfully installed hooks in {success}/{len(repos)} repositories")
        
        if success > 0:
            print("\n[SUCCESS] Hooks are now active!")
            print("   Every new commit will be automatically indexed.")
            print("   No manual action required.")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Install Context Keeper Git Hooks')
    parser.add_argument('action', choices=['install', 'uninstall', 'install-all'], 
                       help='Action to perform')
    parser.add_argument('--repo', help='Repository path (for single repo operations)')
    parser.add_argument('--api-url', default='http://localhost:8000', 
                       help='Context Keeper API URL')
    parser.add_argument('--context-keeper-path', 
                       help='Path to Context Keeper installation')
    
    args = parser.parse_args()
    
    installer = HookInstaller(
        context_keeper_path=args.context_keeper_path,
        api_url=args.api_url
    )
    
    if args.action == 'install-all':
        installer.install_all()
    elif args.action == 'install':
        if not args.repo:
            print("[ERROR] Please specify --repo for single installation")
            sys.exit(1)
        installer.install_hook(args.repo)
    elif args.action == 'uninstall':
        if not args.repo:
            print("[ERROR] Please specify --repo to uninstall")
            sys.exit(1)
        installer.uninstall_hook(args.repo)

if __name__ == "__main__":
    main()
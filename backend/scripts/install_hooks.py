#!/usr/bin/env python3
"""
Install Context Keeper git hooks in repositories - Enhanced Version
"""
import os
import sys
import json
from pathlib import Path
import stat
import subprocess
import shutil
from datetime import datetime

# Enhanced hook template with better error handling and logging
HOOK_TEMPLATE = '''#!/usr/bin/env python3
"""
Context Keeper Auto-Indexing Hook
Automatically sends new commits to Context Keeper
"""
import subprocess
import sys
import os
from pathlib import Path
import json
from datetime import datetime

# Configuration
CONTEXT_KEEPER_PATH = "{context_keeper_path}"
API_URL = "{api_url}"

def log_message(message):
    """Log messages to a file for debugging"""
    log_file = Path.home() / ".context-keeper-hook.log"
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(f"[{{timestamp}}] {{message}}\\n")
    except:
        pass

def check_api_health():
    """Check if Context Keeper API is accessible"""
    try:
        import urllib.request
        import urllib.error
        
        req = urllib.request.Request(f"{{API_URL}}/health")
        with urllib.request.urlopen(req, timeout=3) as response:
            return response.status == 200
    except:
        return False

def index_commit():
    try:
        log_message("Hook triggered")
        
        # Get current working directory (the git repository)
        repo_path = os.getcwd()
        log_message(f"Repository path: {{repo_path}}")
        
        # Check if Context Keeper API is accessible
        if not check_api_health():
            log_message("Context Keeper API not accessible, skipping")
            print("Context Keeper: API not accessible, skipping indexing")
            return
        
        # Get the latest commit hash
        result = subprocess.run(
            ["git", "log", "-1", "--format=%H"],
            capture_output=True,
            text=True,
            cwd=repo_path,
            timeout=5
        )
        
        if result.returncode != 0:
            log_message(f"Failed to get commit hash: {{result.stderr}}")
            return
            
        commit_hash = result.stdout.strip()
        log_message(f"Processing commit: {{commit_hash[:8]}}")
        
        # Find git collector script
        collector_script = Path(CONTEXT_KEEPER_PATH) / "collectors" / "git" / "git_collector.py"
        
        if not collector_script.exists():
            log_message(f"Collector script not found at: {{collector_script}}")
            print(f"Context Keeper: Collector not found at {{collector_script}}")
            return
        
        log_message(f"Using collector: {{collector_script}}")
        
        # Run the git collector for just this commit
        cmd = [
            sys.executable,
            str(collector_script),
            "--repo", repo_path,
            "--history", "1",
            "--api-url", API_URL,
            "--skip-duplicates"
        ]
        
        log_message(f"Running command: {{' '.join(cmd)}}")
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=15  # Increased timeout
        )
        
        if result.returncode == 0:
            log_message(f"Successfully indexed commit {{commit_hash[:8]}}")
            print("Context Keeper: Indexed commit " + commit_hash[:8])
        else:
            log_message(f"Failed to index commit: stdout={{result.stdout}}, stderr={{result.stderr}}")
            print("Context Keeper: Failed to index (check logs)")
            
    except subprocess.TimeoutExpired:
        log_message("Timeout occurred while indexing")
        print("Context Keeper: Timeout (service might be slow)")
    except Exception as e:
        log_message(f"Exception occurred: {{str(e)}}")
        print(f"Context Keeper: Error - {{str(e)}}")

if __name__ == "__main__":
    # Don't block the commit if Context Keeper fails
    try:
        index_commit()
    except Exception as e:
        log_message(f"Hook failed with exception: {{str(e)}}")
    sys.exit(0)
'''

class EnhancedHookInstaller:
    def __init__(self, context_keeper_path=None, api_url="http://localhost:8000"):
        self.context_keeper_path = context_keeper_path or str(Path(__file__).parent.parent.parent)
        self.api_url = api_url
        self.repositories_file = Path(self.context_keeper_path) / "backend" / "data" / "repositories.json"
        
    def validate_environment(self):
        """Validate the environment before installing hooks"""
        issues = []
        
        # Check if Context Keeper path exists
        if not Path(self.context_keeper_path).exists():
            issues.append(f"Context Keeper path not found: {self.context_keeper_path}")
        
        # Check if git collector exists
        collector_path = Path(self.context_keeper_path) / "collectors" / "git" / "git_collector.py"
        if not collector_path.exists():
            issues.append(f"Git collector not found: {collector_path}")
        
        # Check if Python is available
        try:
            result = subprocess.run([sys.executable, "--version"], capture_output=True)
            if result.returncode != 0:
                issues.append("Python interpreter not working")
        except:
            issues.append("Python interpreter not accessible")
        
        # Try to reach Context Keeper API
        try:
            import urllib.request
            req = urllib.request.Request(f"{self.api_url}/health")
            with urllib.request.urlopen(req, timeout=3):
                pass
        except:
            issues.append(f"Context Keeper API not accessible at {self.api_url}")
        
        return issues
    
    def load_repositories(self):
        """Load tracked repositories from Context Keeper"""
        if not self.repositories_file.exists():
            print("[ERROR] No repositories found. Add repositories through Context Keeper first.")
            return []
        
        try:
            with open(self.repositories_file, 'r') as f:
                repos = json.load(f)
                return list(repos.keys())
        except Exception as e:
            print(f"[ERROR] Failed to load repositories: {e}")
            return []
    
    def install_hook(self, repo_path):
        """Install post-commit hook in a single repository"""
        repo_path = Path(repo_path).resolve()
        
        print(f"[INFO] Installing hook in: {repo_path}")
        
        if not (repo_path / ".git").exists():
            print(f"[ERROR] {repo_path} is not a git repository")
            return False
        
        # Validate environment first
        issues = self.validate_environment()
        if issues:
            print("[WARNING] Environment issues detected:")
            for issue in issues:
                print(f"  - {issue}")
            if input("Continue anyway? (y/N): ").lower() != 'y':
                return False
        
        hooks_dir = repo_path / ".git" / "hooks"
        hooks_dir.mkdir(exist_ok=True)
        
        # Create post-commit hook
        hook_path = hooks_dir / "post-commit"
        
        # Generate hook content with paths
        hook_content = HOOK_TEMPLATE.format(
            context_keeper_path=str(Path(self.context_keeper_path).resolve()).replace('\\', '/'),
            api_url=self.api_url
        )
        
        # Check if hook already exists
        if hook_path.exists():
            with open(hook_path, 'r', encoding='utf-8') as f:
                existing = f.read()
                if "Context Keeper" in existing:
                    print(f"[OK] Hook already installed in {repo_path}")
                    return self.test_hook(repo_path)
                else:
                    # Backup existing hook
                    backup_path = hooks_dir / "post-commit.backup"
                    with open(backup_path, 'w', encoding='utf-8') as backup:
                        backup.write(existing)
                    print(f"[BACKUP] Backed up existing hook to {backup_path}")
        
        # Write new hook with explicit encoding
        try:
            with open(hook_path, 'w', encoding='utf-8', newline='\n') as f:
                f.write(hook_content)
        except Exception as e:
            print(f"[ERROR] Failed to write hook: {e}")
            return False
        
        # Make executable on Unix systems
        if os.name != 'nt':
            try:
                st = os.stat(hook_path)
                os.chmod(hook_path, st.st_mode | stat.S_IEXEC)
                print(f"[OK] Made hook executable")
            except Exception as e:
                print(f"[WARNING] Failed to make hook executable: {e}")
        
        print(f"[OK] Installed hook in {repo_path}")
        
        # Test the hook
        return self.test_hook(repo_path)
    
    def test_hook(self, repo_path):
        """Test if the hook works"""
        repo_path = Path(repo_path)
        hook_path = repo_path / ".git" / "hooks" / "post-commit"
        
        if not hook_path.exists():
            print("   [ERROR] Hook file not found")
            return False
        
        print("   [INFO] Testing hook...")
        
        try:
            # Test basic execution
            result = subprocess.run(
                [sys.executable, str(hook_path)],
                capture_output=True,
                text=True,
                timeout=10,
                cwd=repo_path
            )
            
            success_indicators = [
                "Context Keeper:" in result.stdout,
                "Context Keeper:" in result.stderr,
                result.returncode == 0
            ]
            
            if any(success_indicators):
                print("   [OK] Hook test successful")
                if result.stdout.strip():
                    print(f"   Output: {result.stdout.strip()}")
                return True
            else:
                print("   [WARNING] Hook test unclear")
                print(f"   Return code: {result.returncode}")
                print(f"   Stdout: {result.stdout}")
                print(f"   Stderr: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            print("   [WARNING] Hook test timed out")
            return False
        except Exception as e:
            print(f"   [WARNING] Hook test failed: {e}")
            return False
    
    def create_test_commit(self, repo_path):
        """Create a test commit to verify hook functionality"""
        repo_path = Path(repo_path)
        
        print(f"[INFO] Creating test commit in {repo_path}")
        
        try:
            # Create a test file
            test_file = repo_path / ".context-keeper-test.txt"
            with open(test_file, 'w') as f:
                f.write(f"Test file created at {datetime.now()}")
            
            # Add and commit
            subprocess.run(["git", "add", ".context-keeper-test.txt"], cwd=repo_path, check=True)
            result = subprocess.run(
                ["git", "commit", "-m", "Test: Context Keeper hook verification"],
                cwd=repo_path,
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                print("[OK] Test commit created successfully")
                print("Check the output above for 'Context Keeper: Indexed commit'")
                
                # Clean up test file
                test_file.unlink()
                subprocess.run(["git", "rm", ".context-keeper-test.txt"], cwd=repo_path)
                subprocess.run(["git", "commit", "-m", "Clean up test file"], cwd=repo_path)
                
                return True
            else:
                print(f"[ERROR] Failed to create test commit: {result.stderr}")
                return False
                
        except Exception as e:
            print(f"[ERROR] Test commit failed: {e}")
            return False
    
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
            print(f"\n[INFO] Hook logs are written to: {Path.home() / '.context-keeper-hook.log'}")

def main():
    import argparse
    from datetime import datetime
    
    parser = argparse.ArgumentParser(description='Install Context Keeper Git Hooks - Enhanced')
    parser.add_argument('action', choices=['install', 'uninstall', 'install-all', 'test'], 
                       help='Action to perform')
    parser.add_argument('--repo', help='Repository path (for single repo operations)')
    parser.add_argument('--api-url', default='http://localhost:8000', 
                       help='Context Keeper API URL')
    parser.add_argument('--context-keeper-path', 
                       help='Path to Context Keeper installation')
    parser.add_argument('--create-test-commit', action='store_true',
                       help='Create a test commit to verify hook works')
    
    args = parser.parse_args()
    
    installer = EnhancedHookInstaller(
        context_keeper_path=args.context_keeper_path,
        api_url=args.api_url
    )
    
    if args.action == 'install-all':
        installer.install_all()
    elif args.action == 'install':
        if not args.repo:
            print("[ERROR] Please specify --repo for single installation")
            sys.exit(1)
        success = installer.install_hook(args.repo)
        if success and args.create_test_commit:
            installer.create_test_commit(args.repo)
    elif args.action == 'uninstall':
        if not args.repo:
            print("[ERROR] Please specify --repo to uninstall")
            sys.exit(1)
        installer.uninstall_hook(args.repo)
    elif args.action == 'test':
        if not args.repo:
            print("[ERROR] Please specify --repo for testing")
            sys.exit(1)
        installer.test_hook(args.repo)

if __name__ == "__main__":
    main()
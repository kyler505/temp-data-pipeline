"""Google Colab bootstrap utility.

This module provides a single entry point for setting up a Colab environment
to work with the temp-data-pipeline. It handles:
1. Mounting Google Drive
2. Syncing the latest code from Git
3. Installing required dependencies
4. Configuring persistent data paths
"""

import os
import sys
import subprocess
from pathlib import Path
import tempdata.config as config


def bootstrap(
    project_name: str = "temp-data-pipeline",
    use_wandb: bool = False,
    drive_mount: str = "/content/drive",
    workspace_subdir: str = "MyDrive/temp-data-pipeline",
    install_deps: bool = True,
) -> None:
    """Automates the setup for Google Colab environments.

    Args:
        project_name: Name of the project/repo.
        use_wandb: Whether to initialize Weights & Biases.
        drive_mount: The path where Google Drive should be mounted.
        workspace_subdir: Relative path from drive_mount to the project root.
        install_deps: Whether to run pip install for dependencies.
    """
    # 1. Check if running in Google Colab environment
    try:
        from google.colab import drive
        in_colab = True
    except ImportError:
        in_colab = False
        print("[colab] Not running in Google Colab. Skipping bootstrap.")
        return

    print(f"\n{'='*60}")
    print(f"COLAB BOOTSTRAP: {project_name}")
    print(f"{'='*60}")

    # 2. Fix Python 3.12 compatibility (imp module removal)
    # Python 3.12 removed 'imp', but some versions of IPython's autoreload still use it.
    try:
        import imp
    except ImportError:
        print("[colab] Applying 'imp' module shim for Python 3.12 compatibility...")
        from types import ModuleType
        import importlib
        imp_shim = ModuleType("imp")
        imp_shim.reload = importlib.reload
        sys.modules["imp"] = imp_shim

    # 3. Mount Google Drive
    print(f"[colab] Mounting Google Drive to {drive_mount}...")
    try:
        drive.mount(drive_mount, force_remount=False)
        print("[colab] ✓ Drive mounted.")
    except Exception as e:
        print(f"[colab] ⚠ Error mounting Drive: {e}")
        return

    # 3. Set workspace and change directory
    workspace_path = Path(drive_mount) / workspace_subdir
    if not workspace_path.exists():
        print(f"[colab] ⚠ Workspace path '{workspace_path}' not found.")
        print("[colab] Please ensure the repository is cloned to your Drive.")
        return

    # Add workspace to path and change directory
    os.chdir(workspace_path)
    if str(workspace_path) not in sys.path:
        sys.path.insert(0, str(workspace_path))

    # Ensure current directory is in sys.path to allow imports from local src
    if "." not in sys.path:
        sys.path.insert(0, ".")

    print(f"[colab] ✓ Working directory: {os.getcwd()}")

    # 4. Sync Git (Pull latest changes)
    print("[colab] Syncing code with repository (git pull)...")
    try:
        # Check if we are in a git repo
        if (workspace_path / ".git").exists():
            # Try to get current branch name
            branch_res = subprocess.run(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                capture_output=True, text=True
            )
            branch = branch_res.stdout.strip()

            if branch == "HEAD":
                print("[colab] ℹ Detached HEAD detected. Attempting pull from origin main/master...")
                # Try common names
                subprocess.run(["git", "pull", "origin", "main"], check=False, capture_output=True)
                subprocess.run(["git", "pull", "origin", "master"], check=False, capture_output=True)
            else:
                result = subprocess.run(
                    ["git", "pull", "origin", branch],
                    check=False,
                    capture_output=True,
                    text=True
                )
                if result.returncode == 0:
                    print(f"[colab] ✓ Git pull successful (branch: {branch}).")
                else:
                    print(f"[colab] ℹ Git pull: {result.stderr.strip() or 'Already up to date.'}")
        else:
            print("[colab] ⚠ Not a git repository, skipping sync.")
    except Exception as e:
        print(f"[colab] ℹ Git sync skipped or failed: {e}")

    # 5. Install Dependencies and Package in editable mode
    if install_deps:
        print("[colab] Installing dependencies and package (pip install -e .)...")
        try:
            # Install with all relevant extras
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "-q", "-e", ".[eval,backtest,era5]"],
                check=True
            )
            print("[colab] ✓ Dependencies and package installed.")
        except subprocess.CalledProcessError as e:
            print(f"[colab] ⚠ Dependency installation failed: {e}")

    # 6. Configure Data Root Persistence
    # By default, config.data_root() uses project_root() / "data"
    # In Colab, we want to ensure it points to the Drive location
    drive_data_dir = workspace_path / "data"
    drive_data_dir.mkdir(parents=True, exist_ok=True)

    # Dynamically override the data_root function in the config module
    config.data_root = lambda: drive_data_dir
    print(f"[colab] ✓ Persistent data root: {drive_data_dir}")

    # 7. Weights & Biases (Optional)
    if use_wandb:
        print("[colab] Setting up Weights & Biases...")
        try:
            import wandb
            # Check if logged in
            if wandb.api.api_key is None:
                print("[colab] Please authenticate W&B:")
                wandb.login()
            else:
                print("[colab] ✓ W&B already authenticated.")
        except ImportError:
            print("[colab] ⚠ W&B library not found. Install with 'pip install wandb'.")
        except Exception as e:
            print(f"[colab] ⚠ W&B setup failed: {e}")

    print(f"\n{'='*60}")
    print("BOOTSTRAP COMPLETE - Environment is ready.")
    print(f"{'='*60}\n")

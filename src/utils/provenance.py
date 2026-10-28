"""Reproducibility helpers shared by every milestone report.

The project folder is not a Git repository, so the "code revision" is a SHA-256 over the
contents of every source, config and test file (sorted by path). Any edit changes it.
"""
from __future__ import annotations

import hashlib
import platform
import subprocess
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

TREE_PATTERNS = ("src/**/*.py", "configs/*.yaml", "tests/*.py", "requirements*.txt", "pytest.ini")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def code_tree_sha256(repo: Path) -> dict[str, Any]:
    """{'tree_sha256': ..., 'n_files': ..., 'files': {relative path: sha256}}."""
    files: dict[str, str] = {}
    for pattern in TREE_PATTERNS:
        for path in repo.glob(pattern):
            if path.is_file() and "__pycache__" not in path.parts:
                files[path.relative_to(repo).as_posix()] = sha256_bytes(path.read_bytes())
    listing = "".join(f"{name}\t{files[name]}\n" for name in sorted(files))
    return {"tree_sha256": sha256_bytes(listing.encode("utf-8")), "n_files": len(files), "files": dict(sorted(files.items()))}


def _pkg(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "not installed"


def environment() -> dict[str, Any]:
    """OS, Python, package versions and GPU details (driver via nvidia-smi if available)."""
    env: dict[str, Any] = {
        "os": platform.platform(),
        "python": sys.version.split()[0],
        "packages": {p: _pkg(p) for p in ("numpy", "pandas", "scipy", "wfdb", "PyYAML", "pytest", "torch", "matplotlib")},
    }
    try:
        import torch
        env["torch_cuda_build"] = torch.version.cuda
        env["cudnn"] = torch.backends.cudnn.version()
        env["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            env["gpu"] = torch.cuda.get_device_name(0)
            env["gpu_capability"] = ".".join(map(str, torch.cuda.get_device_capability(0)))
            env["gpu_total_memory_MiB"] = round(torch.cuda.get_device_properties(0).total_memory / 2 ** 20)
    except ImportError:
        env["torch"] = "not installed"
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=20)
        env["nvidia_driver"] = out.stdout.strip() or "unknown"
    except (OSError, subprocess.SubprocessError):
        env["nvidia_driver"] = "nvidia-smi not available"
    return env

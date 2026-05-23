from __future__ import annotations

import importlib.util
import os
import site
import subprocess
import sys
from pathlib import Path


DEFAULT_PACKAGE_DIR = Path("/data/python_packages")
DEFAULT_TORCH_SPEC = "torch==2.5.1+cpu"
DEFAULT_TORCH_EXTRA_INDEX = "https://download.pytorch.org/whl/cpu"


def main() -> None:
    package_dir = Path(os.environ.get("ARENA_RUNTIME_PYTHON_PACKAGES", str(DEFAULT_PACKAGE_DIR)))
    _activate_package_dir(package_dir)
    _ensure_torch(package_dir)

    from .live_bot import main as live_main

    live_main()


def _activate_package_dir(package_dir: Path) -> None:
    package_dir.mkdir(parents=True, exist_ok=True)
    package_dir_s = str(package_dir)
    if package_dir_s not in sys.path:
        sys.path.insert(0, package_dir_s)
    site.addsitedir(package_dir_s)
    existing = os.environ.get("PYTHONPATH", "")
    parts = [part for part in existing.split(os.pathsep) if part]
    if package_dir_s not in parts:
        os.environ["PYTHONPATH"] = package_dir_s + (os.pathsep + existing if existing else "")


def _ensure_torch(package_dir: Path) -> None:
    if importlib.util.find_spec("torch") is not None:
        return

    torch_spec = os.environ.get("ARENA_TORCH_SPEC", DEFAULT_TORCH_SPEC)
    extra_index = os.environ.get("ARENA_TORCH_EXTRA_INDEX_URL", DEFAULT_TORCH_EXTRA_INDEX)
    cmd = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--no-cache-dir",
        "--prefer-binary",
        "--target",
        str(package_dir),
        "--extra-index-url",
        extra_index,
        torch_spec,
    ]
    _bootstrap_log(f"torch missing; installing {torch_spec} into {package_dir}")
    try:
        subprocess.check_call(cmd)
    except Exception as exc:
        _bootstrap_log(f"torch install failed; live bot will start with Kronos fallback: {exc}")
        return
    importlib.invalidate_caches()
    if importlib.util.find_spec("torch") is None:
        _bootstrap_log(f"torch install finished but import still fails from {package_dir}; using Kronos fallback")
        return
    _bootstrap_log("torch install/import ok")


def _bootstrap_log(message: str) -> None:
    text = f"[arena-bootstrap] {message}"
    print(text, flush=True)
    log_dir = Path(os.environ.get("ARENA_LOGS_DIR", "/data/logs"))
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
        with (log_dir / "bootstrap.log").open("a", encoding="utf-8") as f:
            f.write(text + "\n")
    except Exception:
        pass


if __name__ == "__main__":
    main()

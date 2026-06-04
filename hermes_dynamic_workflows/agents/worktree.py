"""Per-child workspace leases for workflow agents."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path


@dataclass
class WorkspaceLease:
    task_id: str
    cwd: str
    isolation: str | None = None
    path: str | None = None
    branch: str | None = None
    repo_root: str | None = None
    keep: bool = False

    def cleanup(self) -> None:
        if self.isolation != "worktree" or not self.path or not self.repo_root:
            return
        if self.keep:
            return
        wt_path = Path(self.path)
        if not wt_path.exists():
            return
        if _worktree_has_changes(wt_path):
            return
        subprocess.run(
            ["git", "worktree", "remove", "--force", str(wt_path)],
            cwd=self.repo_root,
            capture_output=True,
            text=True,
            timeout=30,
        )
        if self.branch:
            subprocess.run(
                ["git", "branch", "-D", self.branch],
                cwd=self.repo_root,
                capture_output=True,
                text=True,
                timeout=30,
            )


def create_workspace_lease(
    *,
    cwd: str,
    isolation: str | None,
    label: str,
    task_id: str | None = None,
    keep_worktree: bool = False,
) -> WorkspaceLease:
    task_id = task_id or f"workflow-{uuid.uuid4().hex[:12]}"
    base_cwd = str(Path(cwd or os.getcwd()).expanduser().resolve())
    if isolation in (None, "", "shared"):
        return WorkspaceLease(task_id=task_id, cwd=base_cwd)
    if isolation != "worktree":
        raise ValueError(f"unsupported isolation mode: {isolation!r}")

    repo_root = _git_repo_root(base_cwd)
    if not repo_root:
        raise ValueError("isolation='worktree' requires running inside a git repository")

    short_id = uuid.uuid4().hex[:8]
    safe_label = _safe_label(label)
    wt_name = f"hermes-wf-{safe_label}-{short_id}"[:80].rstrip("-")
    branch = f"hermes/{wt_name}"
    worktrees_dir = Path(repo_root) / ".worktrees"
    worktrees_dir.mkdir(parents=True, exist_ok=True)
    _ensure_gitignore(repo_root)
    wt_path = worktrees_dir / wt_name

    result = subprocess.run(
        ["git", "worktree", "add", str(wt_path), "-b", branch, "HEAD"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        timeout=30,
    )
    if result.returncode != 0:
        raise ValueError(f"failed to create worktree: {(result.stderr or result.stdout).strip()}")

    _copy_worktree_includes(Path(repo_root), wt_path)
    return WorkspaceLease(
        task_id=task_id,
        cwd=str(wt_path),
        isolation="worktree",
        path=str(wt_path),
        branch=branch,
        repo_root=repo_root,
        keep=keep_worktree,
    )


def _git_repo_root(cwd: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except Exception:
        return None
    if result.returncode != 0:
        return None
    return str(Path(result.stdout.strip()).expanduser().resolve())


def _ensure_gitignore(repo_root: str) -> None:
    gitignore = Path(repo_root) / ".gitignore"
    entry = ".worktrees/"
    try:
        existing = gitignore.read_text(encoding="utf-8") if gitignore.exists() else ""
        if entry in existing.splitlines():
            return
        with gitignore.open("a", encoding="utf-8") as handle:
            if existing and not existing.endswith("\n"):
                handle.write("\n")
            handle.write(entry + "\n")
    except Exception:
        pass


def _copy_worktree_includes(repo_root: Path, wt_path: Path) -> None:
    include_file = repo_root / ".worktreeinclude"
    if not include_file.exists():
        return
    repo_root_resolved = repo_root.resolve()
    wt_path_resolved = wt_path.resolve()
    try:
        lines = include_file.read_text(encoding="utf-8").splitlines()
    except Exception:
        return
    for line in lines:
        entry = line.strip()
        if not entry or entry.startswith("#"):
            continue
        src = repo_root / entry
        dst = wt_path / entry
        try:
            src_resolved = src.resolve(strict=False)
            dst_resolved = dst.resolve(strict=False)
        except (OSError, ValueError):
            continue
        if not _is_within(src_resolved, repo_root_resolved):
            continue
        if not _is_within(dst_resolved, wt_path_resolved):
            continue
        if src.is_file():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(str(src), str(dst))
        elif src.is_dir() and not dst.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            try:
                os.symlink(str(src_resolved), str(dst))
            except (OSError, NotImplementedError):
                if sys.platform == "win32":
                    shutil.copytree(str(src_resolved), str(dst), symlinks=True)
                else:
                    raise


def _worktree_has_changes(wt_path: Path) -> bool:
    checks = [
        ["git", "status", "--porcelain"],
        ["git", "log", "--oneline", "HEAD", "--not", "--remotes"],
    ]
    for command in checks:
        try:
            result = subprocess.run(
                command,
                cwd=str(wt_path),
                capture_output=True,
                text=True,
                timeout=10,
            )
        except Exception:
            return True
        if result.returncode != 0:
            return True
        if result.stdout.strip():
            return True
    return False


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _safe_label(label: str) -> str:
    raw = "".join(ch.lower() if ch.isalnum() else "-" for ch in str(label or "agent"))
    clean = "-".join(part for part in raw.split("-") if part)
    return clean[:32] or "agent"

import shutil
import subprocess
from pathlib import Path
from typing import List, Optional, Union

import pandas as pd


def prepare_SWE_env(instance_id, clone_root):
    """
    Clone repo from SWE-bench mirror. It ensures that commit history stays the same.
    """
    if clone_root is None:
        clone_root = Path(__file__).parent.parent / "output"
    else:
        clone_root = Path(clone_root)

    clone_root.mkdir(parents=True, exist_ok=True)

    parquet_path = Path(__file__).parent.parent / "swe_bench_verified.parquet"

    df = pd.read_parquet(parquet_path)

    instance_data = df[df["instance_id"] == instance_id]

    if instance_data.empty:
        raise ValueError(f"Instance ID '{instance_id}' not found in parquet file")

    row = instance_data.iloc[0]
    repo = row["repo"]
    problem_statement = row.get("problem_statement", "")
    base_commit = row["base_commit"]

    # Convert repo format: "owner/repo" -> "owner__repo" for SWE-bench mirror
    repo_name = repo.replace("/", "__")
    repo_dir = clone_root / repo_name

    output = subprocess.run(
        [
            "git",
            "clone",
            f"https://github.com/SWE-bench-repos/{repo_name}.git",
            str(repo_dir),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    output = subprocess.run(
        ["git", "checkout", base_commit],
        cwd=repo_dir,
        capture_output=True,
        text=True,
        check=True,
    )

def _resolve_clone_url(repo: str) -> str:
    """Turn owner/repo or a full Git URL into a clone URL Git accepts."""
    r = repo.strip().rstrip("/")
    if r.startswith("git@") or r.startswith("http://") or r.startswith("https://"):
        return r
    if "/" in r and "://" not in r:
        return f"https://github.com/{r}.git" if not r.endswith(".git") else f"https://github.com/{r}"
    raise ValueError(
        f"repo={repo!r} is not a valid remote: use owner/repo, https://..., or git@..."
    )


def _local_repo_dir_name(repo: str) -> str:
    """Directory name under clone_root (avoids collisions for owner/repo)."""
    r = repo.strip().rstrip("/")
    if r.startswith("git@"):
        path = r.split(":", 1)[-1]
    elif "://" in r:
        path = r.split("://", 1)[-1].split("/", 1)[-1]
    else:
        path = r
    if path.endswith(".git"):
        path = path[:-4]
    if "/" in path:
        return path.replace("/", "__")
    return path


def _run_git(args: List[str], *, cwd: Optional[Path] = None) -> None:
    proc = subprocess.run(
        ["git", *args],
        cwd=cwd,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout or "").strip()
        raise RuntimeError(f"git {' '.join(args)} failed (exit {proc.returncode}): {err}")


def prepare_custom_env(
    repo: str,
    base_commit: Optional[str] = None,
    clone_root: Optional[Union[Path, str]] = None,
    *,
    problem_statement: Optional[str] = None,
    issue_filename: str = "AGENT_ISSUE.md",
    force_reclone: bool = True,
) -> Path:
    """
    """
    pin = (base_commit or "").strip() or None

    if clone_root is None:
        clone_root = Path(__file__).resolve().parent.parent / "output"
    else:
        clone_root = Path(clone_root).expanduser().resolve()

    clone_root.mkdir(parents=True, exist_ok=True)

    clone_url = _resolve_clone_url(repo)
    local_name = _local_repo_dir_name(repo)
    repo_dir = clone_root / local_name

    if repo_dir.exists():
        if not pin or force_reclone:
            shutil.rmtree(repo_dir)
        else:
            if not (repo_dir / ".git").is_dir():
                raise FileExistsError(
                    f"{repo_dir} exists and is not a git repo; remove it or use force_reclone=True"
                )
            _run_git(["fetch", "--all", "--tags"], cwd=repo_dir)
            _run_git(["checkout", "--force", pin], cwd=repo_dir)
            _run_git(["clean", "-fdx"], cwd=repo_dir)
            if problem_statement is not None:
                (repo_dir / issue_filename).write_text(problem_statement, encoding="utf-8")
            return repo_dir.resolve()

    _run_git(["clone", clone_url, str(repo_dir)])
    if pin:
        _run_git(["checkout", pin], cwd=repo_dir)

    if problem_statement is not None:
        (repo_dir / issue_filename).write_text(problem_statement, encoding="utf-8")

    return repo_dir.resolve()


if __name__ == "__main__":
    path = prepare_custom_env(
        "scikit-learn/scikit-learn",
        base_commit=None,
        problem_statement="Description of the bug.",
    )
    print(path)

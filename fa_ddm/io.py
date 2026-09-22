"""Configuration loading and reproducible result export."""

from datetime import datetime
from pathlib import Path
import shutil
import subprocess

import pandas as pd
import yaml


def load_yaml(path):
    """Load a YAML file and return a dictionary."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as stream:
        data = yaml.safe_load(stream)
    if not isinstance(data, dict):
        raise ValueError("YAML root must be a mapping.")
    return data


def ensure_directory(path):
    """Create a directory if needed and return it as a Path."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def git_commit_reference(repository_root="."):
    """Return the short Git commit hash for the current repository."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(repository_root),
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "no-git"
    return result.stdout.strip()


def git_is_dirty(repository_root="."):
    """Return True when tracked or untracked repository changes exist."""
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=str(repository_root),
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return False
    return bool(result.stdout.strip())


def build_run_reference(repository_root=".", date_string=None):
    """Build YYYYMMDD_<short-commit>, adding _dirty when needed."""
    if date_string is None:
        date_string = datetime.now().strftime("%Y%m%d")
    reference = "{0}_{1}".format(
        date_string, git_commit_reference(repository_root)
    )
    if git_is_dirty(repository_root):
        reference += "_dirty"
    return reference


def prepare_experiment_directory(
    results_root,
    experiment_name,
    repository_root=".",
    run_reference=None,
):
    """Create results/<experiment>/<date_commit>/ and return its path."""
    if run_reference is None:
        run_reference = build_run_reference(repository_root)
    return ensure_directory(
        Path(results_root) / str(experiment_name) / str(run_reference)
    )


def copy_configuration(source, destination):
    """Copy the exact configuration used by an experiment."""
    shutil.copy2(str(source), str(destination))


def save_dat(data, path, comments=None):
    """Save a PGFPlots-friendly whitespace-separated data file."""
    path = Path(path)
    frame = pd.DataFrame(data)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        if comments:
            for comment in comments:
                stream.write("# {0}\n".format(comment))
        frame.to_csv(
            stream,
            sep=" ",
            index=False,
            float_format="%.10g",
            lineterminator="\n",
        )

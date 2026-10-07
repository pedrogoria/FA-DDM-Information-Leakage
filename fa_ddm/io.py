r"""
Configuration loading, Git-aware run identification, and reproducible result
export for the FA-DDM numerical framework.

Author
------
Pedro E. Goria Silva

Module
------
fa_ddm.io

Purpose
-------
This module implements the input, output, and provenance utilities used by the
fluid-antenna-assisted dynamic directional modulation (FA-DDM) simulation and
numerical-analysis framework.

The module provides reusable functions for:

1. loading experiment configurations from YAML files;
2. creating output directories safely and recursively;
3. retrieving the current abbreviated Git commit identifier;
4. detecting tracked or untracked repository changes;
5. constructing date- and commit-based run references;
6. creating versioned experiment-result directories;
7. preserving the exact YAML configuration used by a run; and
8. exporting rectangular, PGFPlots-compatible numerical data files.

The central design goal is reproducibility. Every experiment can associate its
numerical outputs with the configuration, calendar date, Git commit, and
working-tree state used to generate the results.

Configuration loading
---------------------
Experiment parameters are stored in YAML files, normally under

    configs/figures/

The function ``load_yaml`` reads a UTF-8 YAML file using ``yaml.safe_load`` and
requires the document root to be a mapping. The function returns a standard
Python dictionary for use by experiment runners.

Using ``safe_load`` prevents execution of arbitrary Python objects encoded in a
YAML document. The function validates only the root data type; validation of
experiment-specific keys, values, dimensions, and physical parameters remains
the responsibility of the corresponding runner or scientific module.

Directory management
--------------------
The function ``ensure_directory`` converts its input to ``pathlib.Path``,
creates the complete parent hierarchy when necessary, and returns the resulting
path object.

Directory creation uses

    parents=True
    exist_ok=True

and is therefore idempotent for an existing directory. File-system permission
errors and invalid paths are allowed to propagate to the caller so that failed
experiment setup is not silently ignored.

Git provenance
--------------
The module records source-code provenance through two Git queries.

``git_commit_reference`` executes

    git rev-parse --short HEAD

inside the specified repository root and returns the abbreviated identifier of
the currently checked-out commit. If Git is unavailable, the path is not a Git
repository, or the command fails, the function returns

    no-git

rather than interrupting the numerical experiment.

``git_is_dirty`` executes

    git status --porcelain

and returns ``True`` when tracked modifications, staged changes, or untracked
files are present. A nonempty porcelain-format output therefore marks the
repository as dirty. If the Git query cannot be executed, the function returns
``False`` because the repository state cannot be established through Git.

Run references
--------------
The function ``build_run_reference`` constructs identifiers using the format

    YYYYMMDD_<short-commit>

where the date is obtained from the local system clock unless an explicit
``date_string`` is supplied. When ``git_is_dirty`` reports repository changes,
the suffix

    _dirty

is appended, producing

    YYYYMMDD_<short-commit>_dirty

A run reference therefore identifies the day, committed source-code state, and
whether uncommitted changes existed when the experiment directory was created.

If Git metadata are unavailable, the identifier takes the form

    YYYYMMDD_no-git

with ``_dirty`` appended only if the dirty-state query returns ``True``.

Versioned result directories
----------------------------
The function ``prepare_experiment_directory`` creates and returns

    <results_root>/<experiment_name>/<run_reference>/

For the standard repository layout, this becomes

    results/<experiment_name>/<YYYYMMDD>_<short-commit>[_dirty]/

A caller may provide an explicit ``run_reference`` for testing, deterministic
path construction, or controlled result regeneration. Otherwise, the reference
is generated automatically from the repository state.

The function creates the directory but does not delete, overwrite, or clean
existing contents. Repeated runs using the same reference may therefore write
to the same directory. Experiment runners should use stable output names and
should avoid unintentionally mixing incompatible results.

Configuration preservation
--------------------------
The function ``copy_configuration`` copies the exact source configuration to a
specified destination using ``shutil.copy2``. This operation preserves file
metadata when supported by the operating system.

Experiment runners normally store the copied configuration as

    config_used.yaml

inside the versioned result directory. Preserving the original file, rather
than serializing an in-memory dictionary, ensures that comments, ordering,
formatting, and unprocessed fields remain available for later inspection.

Scientific data export
----------------------
The function ``save_dat`` exports a mapping of named one-dimensional data
columns as a whitespace-separated table suitable for pandas, NumPy, TikZ,
PGFPlots, and PGFPlotstable workflows.

The mapping is first converted to a ``pandas.DataFrame``. Therefore, columns
must have compatible lengths or must be representable by pandas under its
normal DataFrame construction rules.

Optional metadata comments are written before the table header. Every comment
line begins with

    #

so that downstream readers can ignore the metadata while preserving scientific
context such as experiment identifiers, modulation formats, channel settings,
solver conventions, or reference-symbol indices.

The table is written using the conventions

    separator       = one space;
    index            = omitted;
    floating format  = %.10g;
    missing value    = nan;
    line terminator  = newline.

Writing missing values explicitly as ``nan`` is essential for rectangular
space-separated tables. Empty fields would shift subsequent columns and can
cause PGFPlots to report unbalanced rows or missing columns.

The output stream is opened with UTF-8 encoding and an explicit newline policy
to provide stable text output across supported platforms.

Public functions
----------------
load_yaml
    Loads a UTF-8 YAML document with ``yaml.safe_load`` and returns the root
    mapping as a dictionary.

ensure_directory
    Creates a directory and all missing parents, then returns a ``Path``
    object.

git_commit_reference
    Returns the abbreviated Git commit identifier or ``"no-git"`` when the
    commit cannot be obtained.

git_is_dirty
    Returns whether ``git status --porcelain`` reports tracked or untracked
    repository changes.

build_run_reference
    Builds the date-commit run identifier and appends ``_dirty`` when needed.

prepare_experiment_directory
    Creates the standard versioned result directory for one experiment.

copy_configuration
    Copies the exact experiment configuration to the result directory.

save_dat
    Writes named numerical columns and optional comment metadata to a
    PGFPlots-friendly whitespace-separated data file.

Input conventions
-----------------
path
    String or path-like object accepted by ``pathlib.Path``.

repository_root
    Directory in which the Git commands are executed. The default is the
    current working directory.

date_string
    Optional date or other caller-defined prefix. The standard automatically
    generated representation is ``YYYYMMDD``.

results_root
    Base directory for all numerical experiment outputs, normally ``results``.

experiment_name
    Stable identifier shared by the YAML configuration and experiment runner.

run_reference
    Optional explicit version-directory name. When omitted, Git-aware naming is
    used.

data
    Mapping from output column names to equally sized array-like values.

comments
    Optional iterable of metadata strings. The function adds the leading
    comment marker and one line terminator to each entry.

Returned quantities
-------------------
``load_yaml`` returns

    data
        Dictionary containing the YAML root mapping.

``ensure_directory`` and ``prepare_experiment_directory`` return

    path
        ``pathlib.Path`` object referring to the created or existing
        directory.

``git_commit_reference`` returns

    commit
        Abbreviated Git commit identifier or ``"no-git"``.

``git_is_dirty`` returns

    dirty
        Boolean repository-state indicator.

``build_run_reference`` returns

    reference
        Date-commit identifier, optionally ending in ``_dirty``.

``copy_configuration`` and ``save_dat`` perform file-system operations and do
not return a value.

Error handling
--------------
The YAML loader raises an exception when the file cannot be opened, the YAML is
invalid, or the root is not a mapping.

Directory, copy, and export functions allow normal file-system exceptions to
propagate. This behavior prevents failed writes, permission problems, invalid
paths, or missing source files from being mistaken for successful experiment
output.

Git helper functions are intentionally fault tolerant. Git-related operating-
system and command-execution errors are converted to fallback values so that
scientific experiments can still run in archives, copied directories, or
environments where Git is unavailable.

Reproducibility guarantees
--------------------------
The module supports reproducibility by recording:

- the exact experiment configuration;
- the execution date;
- the abbreviated Git commit;
- whether the working tree contained uncommitted changes;
- descriptive numerical column names;
- explicit missing values; and
- optional experiment metadata comments.

These records do not by themselves guarantee scientific reproducibility. The
experiment runner must also store the random seed, channel assumptions,
numerical tolerances, solver settings, and output conventions in the YAML file
or exported metadata.

Recommended workflow
--------------------
A final manuscript experiment should normally follow this sequence:

1. inspect the repository with ``git status``;
2. commit scientific code and configuration changes;
3. run the test suite;
4. execute the experiment from a clean commit;
5. inspect the generated PDF preview and data files;
6. copy the selected final ``.dat`` files to the Overleaf project; and
7. preserve the versioned result directory or archive its contents.

A result directory ending in ``_dirty`` is useful during development but should
not normally be treated as the final manuscript artifact without reviewing the
uncommitted changes that produced it.

PGFPlots compatibility
----------------------
The exported data file uses a plain header followed by whitespace-separated
rows. A typical file is

    # Experiment: figure_01_validation
    # Modulation: QPSK
    eve_snr_db vulnerability_theory vulnerability_mc
    -10 0.2638 0.2641
    -5 0.2729 0.2730

When the file is loaded through PGFPlotstable, comment handling should be
enabled, for example:

    comment chars={\#}

Plotting code should reference descriptive column names rather than numerical
column positions.

Dependencies
------------
datetime.datetime
    Local calendar date used in automatically generated run references.

pathlib.Path
    Platform-independent path construction and file handling.

shutil
    Exact configuration-file copying through ``copy2``.

subprocess
    Execution of Git commands for commit and working-tree metadata.

pandas
    Construction and export of rectangular named-column data tables.

PyYAML
    Safe deserialization of experiment configuration files.

Platform considerations
-----------------------
The module is designed for Windows, Linux, and macOS. Git commands require the
``git`` executable to be available through the process environment. When Git is
not available, experiments still run with the ``no-git`` fallback.

The date is obtained from the local system clock and does not include a time,
time zone, or monotonic run counter. Two runs of the same experiment on the same
day and commit therefore use the same default result directory. If unique
per-run directories are required, the caller should provide an explicit
``run_reference`` containing a timestamp or another unique identifier.

Security considerations
-----------------------
YAML files are loaded with ``yaml.safe_load`` rather than the unrestricted YAML
loader. This prevents construction of arbitrary Python objects from untrusted
configuration files.

The module executes fixed Git commands without ``shell=True``. Repository paths
are passed as the subprocess working directory rather than interpolated into a
shell command.

The comments written to ``.dat`` files are not escaped for LaTeX. Callers
should avoid inserting untrusted or publication-sensitive content into comment
metadata that may later be copied into another system.

Modeling limitations
--------------------
- Git commit identifiers are abbreviated and are not cryptographic provenance
  records.
- A clean working tree does not guarantee that external data, software
  versions, or operating-system details are identical.
- The local execution date may differ across time zones.
- A ``no-git`` run cannot be tied to a committed source state through this
  module alone.
- Repeated runs with the same default reference share one directory.
- The exporter does not attach units, schemas, or type declarations beyond the
  column names and optional comments.
- The module does not manage binary figure files, archive compression, remote
  storage, checksums, or Git commits.

Notes
-----
This module contains no scientific model equations, random-number generation,
plotting, or optimization logic. It provides infrastructure for experiment
configuration and reproducible export.

Experiment-specific validation should occur immediately after ``load_yaml`` so
that missing keys or invalid physical parameters are detected before expensive
numerical calculations begin.

See Also
--------
fa_ddm.quadrature
    Deterministic evaluation of Eve's posterior vulnerability.

fa_ddm.optimization
    Privacy-aware port-selection optimization and reliability constraints.

fa_ddm.vulnerability
    Gaussian-mixture likelihoods and Monte Carlo vulnerability estimation.

fa_ddm.bounds
    Finite-SNR and permutation-based vulnerability bounds.

scripts.run_figure_01
    Example experiment runner using the configuration and export workflow.

scripts.run_figure_07
    Vulnerability-bound experiment using versioned output directories.
"""


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
            na_rep="nan",
            lineterminator="\n",
        )

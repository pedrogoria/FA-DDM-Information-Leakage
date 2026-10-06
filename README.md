# FA-DDM Information Leakage

Simulation and numerical-analysis framework for **Fluid-Antenna-Assisted Dynamic Directional Modulation (FA-DDM)**, with emphasis on **posterior vulnerability**, **min-entropy information leakage**, **spatial correlation**, **directional dependence**, **finite-SNR distinguishability bounds**, **permutation-based mixture-overlap bounds**, and **privacy-aware port selection**.

The repository supports the numerical experiments and manuscript figures used to study how random fluid-antenna port activation and Bob-oriented phase compensation affect an optimal eavesdropper.

## Main Capabilities

The repository provides:

- normalized BPSK, PSK, and square-QAM constellations;
- one-dimensional fluid-antenna aperture geometry;
- three-dimensional receiver geometry;
- Clarke-Jakes spatial correlation matrices;
- spatially correlated Rician channel generation;
- deterministic far-field directional signatures;
- Bob-oriented phase precoding;
- Eve-side effective channel construction;
- Gaussian-mixture likelihood evaluation;
- optimal MAP symbol inference at Eve;
- posterior-vulnerability evaluation by deterministic quadrature;
- posterior-vulnerability estimation by Monte Carlo simulation;
- finite-SNR total-variation lower and upper bounds;
- permutation-based vulnerability bounds using linear assignment;
- low-SNR, high-SNR, and fixed-aperture analyses;
- privacy-aware port-selection optimization by linear programming;
- uniform, reliability-weighted, best-Bob fixed-port, and best-privacy fixed-port benchmarks;
- YAML-based configurations for manuscript experiments;
- versioned result folders tied to the Git commit used for each run;
- PGFPlots-ready `.dat` files;
- Python-generated PDF previews;
- standalone LaTeX figures for Overleaf.

---

## Scientific Scope

Alice uses one RF chain connected to a fluid antenna with multiple candidate ports. At each channel use, Alice selects one port and compensates the phase of Bob's instantaneous channel. For port `n`, the phase coefficient is

```text
q_n = conjugate(h_B,n) / |h_B,n|.
```

Bob therefore observes a phase-aligned symbol, while Eve observes the effective channel

```text
g_n = h_E,n * conjugate(h_B,n) / |h_B,n|.
```

The active port is not disclosed to Eve before the symbol decision. Conditioned on a confidential symbol, Eve's observation is consequently a Gaussian mixture over the possible active ports.

The main operational privacy metric is Eve's posterior vulnerability:

```text
V(S|Y_E) = maximum probability of correctly guessing S in one attempt.
```

For a uniform `M`-ary source, the prior vulnerability is

```text
V(S) = 1 / M.
```

The corresponding min-entropy leakage is

```text
L_inf(S -> Y_E) = log2(V(S|Y_E) / V(S)).
```

The repository evaluates these quantities from Eve's complete secret-conditioned Gaussian-mixture likelihoods. Eve is not assessed through a conventional nearest-neighbor detector unless such a detector is explicitly introduced as a benchmark.

### Analytical bounds

For a uniform secret, the finite-SNR analysis relates posterior vulnerability to pairwise total-variation distances between the secret-conditioned densities. The repository evaluates:

- the exact vulnerability by quadrature;
- the total-variation lower bound;
- the total-variation upper bound;
- the permutation-based upper bound.

The permutation-based certificate matches Gaussian components across two secret-conditioned mixtures. The maximizing permutation is computed as a linear assignment problem with `scipy.optimize.linear_sum_assignment`, avoiding explicit enumeration of all port permutations.

### Privacy-aware port selection

The activation vector `rho` is optimized to minimize Eve's posterior vulnerability subject to Bob's average symbol-error constraint. After quadrature and epigraph reformulation, the finite-grid design is a linear program.

The comparison policies are:

- **uniform selection:** equal probability for every port;
- **reliability-weighted selection:** probability proportional to Bob's instantaneous channel power;
- **best-Bob fixed port:** deterministic selection of Bob's strongest port;
- **best-privacy fixed port:** deterministic selection of the individually least vulnerable port.

---

## Repository Structure

```text
.
├── configs/
│   └── figures/
│       ├── figure_01_validation.yaml
│       ├── figure_02_eve_position_map.yaml
│       ├── figure_03_port_density_saturation.yaml
│       ├── figure_04_finite_snr_bounds.yaml
│       ├── figure_05_privacy_reliability.yaml
│       ├── figure_06_directional_vulnerability.yaml
│       └── figure_07_bounds.yaml
│
├── scripts/
│   ├── run_figure_01.py
│   ├── run_figure_02.py
│   ├── run_figure_03.py
│   ├── run_figure_04.py
│   ├── run_figure_05.py
│   ├── run_figure_06.py
│   └── run_figure_07.py
│
├── fa_ddm/
│   ├── __init__.py
│   ├── bounds.py
│   ├── channel.py
│   ├── distinguishability.py
│   ├── geometry.py
│   ├── io.py
│   ├── modulation.py
│   ├── optimization.py
│   ├── quadrature.py
│   ├── spatial_map.py
│   └── vulnerability.py
│
├── overleaf/
│   └── figures/
│       ├── figure_01_validation.tex
│       ├── figure_02_eve_position_map.tex
│       ├── figure_03_port_density_saturation.tex
│       ├── figure_04_finite_snr_bounds.tex
│       ├── figure_05a_privacy_reliability_tradeoff.tex
│       ├── figure_05b_port_selection_probabilities.tex
│       ├── figure_06_directional_vulnerability.tex
│       └── figure_07_vulnerability_bounds.tex
│
├── results/
│   └── <experiment_name>/
│       └── <YYYYMMDD>_<git_commit>[_dirty]/
│           ├── <experiment_name>.dat
│           ├── config_used.yaml
│           ├── <experiment_name>_preview.pdf
│           └── additional experiment-specific .dat files
│
├── tests/
│   ├── test_channel.py
│   ├── test_distinguishability.py
│   ├── test_geometry.py
│   ├── test_import.py
│   ├── test_io_nan_fix.py
│   ├── test_modulation.py
│   ├── test_optimization.py
│   ├── test_quadrature.py
│   ├── test_rician_channel.py
│   ├── test_spatial_map.py
│   └── test_vulnerability.py
│
├── .gitignore
├── pytest.ini
├── pyproject.toml
├── requirements.txt
├── setup.py
└── README.md
```

The exact list of tests may evolve as experiments are added. Scientific modules under `fa_ddm/` should remain independent of figure-specific presentation code.

---

## Python Version and Environment

The project is maintained for:

```text
Python 3.8.8
```

The recommended Windows setup uses a project-local Conda environment:

```text
FA-DDM-Information-Leakage/fa-ddm-env/
```

The environment directory must not be committed. The `.gitignore` file should include:

```gitignore
fa-ddm-env/
.venv/
venv/
env/
__pycache__/
*.py[cod]
.pytest_cache/
.idea/
build/
dist/
*.egg-info/
```

### Create the Conda environment

From Anaconda Prompt:

```cmd
conda create --prefix "C:\Users\<USER>\PycharmProjects\FA-DDM-Information-Leakage\fa-ddm-env" python=3.8.8 pip setuptools wheel pytest
```

Activate the environment:

```cmd
conda activate "C:\Users\<USER>\PycharmProjects\FA-DDM-Information-Leakage\fa-ddm-env"
```

Verify the interpreter:

```cmd
where python
python --version
```

The first Python path should point to the project-local environment and the version should be `Python 3.8.8`.

### Configure PyCharm

In PyCharm:

1. Open `File > Settings > Project > Python Interpreter`.
2. Select `Add Interpreter`.
3. Select `Add Local Interpreter`.
4. Choose an existing Conda environment.
5. Select `<repository>/fa-ddm-env/python.exe`.
6. Apply the configuration and open a new terminal.

Verify inside the PyCharm terminal:

```powershell
python -c "import sys; print(sys.executable)"
python --version
```

---

## Installation

From the repository root:

```powershell
python -m pip install -e . --no-build-isolation
```

Verify the package import:

```powershell
python -c "import fa_ddm; print(fa_ddm.__file__)"
```

The printed path should point to:

```text
<repository>/fa_ddm/__init__.py
```

### Core dependencies

- `numpy`
- `scipy`
- `matplotlib`
- `pandas`
- `PyYAML`
- `pytest`

Dependencies are declared in `pyproject.toml` and `requirements.txt`.

---

## Running the Test Suite

Run the project tests from the repository root:

```powershell
python -m pytest tests -v
```

Use `python -m pytest tests` rather than unrestricted test discovery when the project-local environment is stored inside the repository. Otherwise, `pytest` may attempt to collect third-party package tests from `fa-ddm-env`.

A root-level `pytest.ini` may be used to restrict discovery permanently:

```ini
[pytest]
testpaths = tests
norecursedirs =
    .git
    .idea
    .pytest_cache
    __pycache__
    fa-ddm-env
    results
    build
    dist
python_files = test_*.py
```

### Main validation groups

The tests cover:

- constellation size and normalization;
- uniform symbol probabilities;
- fluid-antenna geometry;
- Clarke-Jakes matrix properties;
- deterministic directional signatures;
- spatially correlated Rician channel normalization;
- random-seed reproducibility;
- Bob-oriented phase compensation;
- Eve's effective channel;
- Gaussian-mixture likelihood dimensions;
- MAP decision validity;
- Monte Carlo vulnerability limits;
- deterministic quadrature accuracy;
- quadrature versus Monte Carlo agreement;
- spatial map ordering;
- finite-SNR total-variation bounds;
- binary vulnerability identity;
- privacy-aware optimization feasibility;
- probability normalization;
- rectangular `.dat` export with missing values.

### Syntax checks

```powershell
python -m py_compile `
  fa_ddm\bounds.py `
  fa_ddm\channel.py `
  fa_ddm\distinguishability.py `
  fa_ddm\geometry.py `
  fa_ddm\io.py `
  fa_ddm\modulation.py `
  fa_ddm\optimization.py `
  fa_ddm\quadrature.py `
  fa_ddm\spatial_map.py `
  fa_ddm\vulnerability.py
```

### YAML checks

```powershell
python -c "from pathlib import Path; import yaml; paths=list(Path('configs').rglob('*.yaml')); [yaml.safe_load(p.read_text(encoding='utf-8')) for p in paths]; print('YAML files OK:', len(paths))"
```

---

## Configuration and Result Versioning

Every manuscript experiment has one YAML configuration under:

```text
configs/figures/
```

Configurations control the random seed, channel model, modulation, port geometry, SNR sweep, quadrature, Monte Carlo trials, reliability thresholds, and output names.

Each run creates:

```text
results/<experiment_name>/<YYYYMMDD>_<short_git_commit>[_dirty]/
```

The output directory normally contains:

- a PGFPlots-ready `.dat` file;
- the exact YAML copied as `config_used.yaml`;
- a Python-generated PDF preview;
- additional `.dat` files for figures with multiple datasets.

The `_dirty` suffix indicates that the repository contained uncommitted changes when the run started. Final manuscript results should preferably be generated from a clean commit.

---

## Data Export Convention

The `.dat` file is the primary scientific output. Python-generated PDFs are previews for checking trends, ranges, and implementation errors.

Data files are whitespace-separated and use named columns. Metadata lines begin with `#`. Missing values are written explicitly as `nan` to preserve rectangular tables.

The exporter uses:

```python
frame.to_csv(
    stream,
    sep=" ",
    index=False,
    float_format="%.10g",
    na_rep="nan",
    lineterminator="\n",
)
```

---

## Overleaf Workflow

The repository preserves versioned experiment outputs. The Overleaf project receives only the selected final `.dat` files.

Copy selected data files to:

```text
fig/data/
```

Current examples include:

```text
fig/data/figure_01_validation.dat
fig/data/figure_02_eve_position_map.dat
fig/data/figure_03_port_density_saturation.dat
fig/data/figure_04_finite_snr_bounds.dat
fig/data/figure_05_privacy_reliability.dat
fig/data/figure_05_port_probabilities.dat
fig/data/figure_06_directional_vulnerability.dat
fig/data/figure_07_bounds.dat
```

The final manuscript PDFs corresponding to the split privacy-design figures are:

```text
privacy_reliability_tradeoff.pdf
port_selection_probabilities.pdf
```

The Figure 7 standalone reads:

```latex
\newcommand{\boundsdata}{fig/data/figure_07_bounds.dat}
```

and the manuscript may include its compiled output as:

```text
vulnerability_bounds.pdf
```

### Plotting convention

1. Analytical or deterministic-quadrature results are lines.
2. Different theoretical cases use different colors and line styles.
3. Monte Carlo results use marks and, when applicable, confidence intervals.
4. Monte Carlo validation does not require duplicate legend entries.
5. Captions or manuscript text identify lines, marks, and bounds.
6. Python previews are diagnostic outputs, not publication figures.

---

## Supported Modulations

```text
BPSK
QPSK
8PSK
16PSK
16QAM
64QAM
```

Modulation names are case-insensitive and may contain spaces or hyphens. All constellations are normalized to unit average symbol energy.

```python
from fa_ddm.modulation import get_constellation

bpsk = get_constellation("BPSK")
qpsk = get_constellation("QPSK")
qam16 = get_constellation("16-QAM")
```

---

## Figure Experiments

### Figure 1: Multiple-Modulation Validation

```powershell
python scripts\run_figure_01.py
```

Configuration:

```text
configs/figures/figure_01_validation.yaml
```

Purpose:

- validate deterministic quadrature against Monte Carlo;
- compare BPSK, QPSK, 8PSK, and 16QAM;
- verify low-SNR behavior;
- verify likelihood consistency across modulation formats.

### Figure 2: Eve Position Map

```powershell
python scripts\run_figure_02.py
```

Purpose:

- fix Bob at a known position;
- evaluate Eve over a planar grid;
- compare QPSK and 16QAM vulnerability;
- export PGFPlots-compatible spatial-map data.

### Figure 3: Fixed-Aperture Port-Density Saturation

```powershell
python scripts\run_figure_03.py
```

Purpose:

- vary the port count at fixed aperture size;
- compare multiple aperture lengths;
- evaluate diminishing changes under dense port sampling;
- validate quadrature trends with Monte Carlo marks.

### Figure 4: Finite-SNR Distinguishability Bounds

```powershell
python scripts\run_figure_04.py
```

Purpose:

- validate the binary total-variation identity;
- compare exact QPSK vulnerability with finite-SNR lower and upper bounds;
- validate exact curves with Monte Carlo marks.

### Figure 5: Privacy-Reliability Design

```powershell
python scripts\run_figure_05.py
```

Purpose:

- minimize Eve's quadrature vulnerability;
- enforce Bob's average symbol-error constraint;
- compare optimized, uniform, reliability-weighted, and fixed-port policies;
- inspect optimized port-selection probabilities.

The experiment produces:

```text
figure_05_privacy_reliability.dat
figure_05_port_probabilities.dat
```

The publication output is split into:

```text
privacy_reliability_tradeoff.pdf
port_selection_probabilities.pdf
```

### Figure 6: Directional Vulnerability

```powershell
python scripts\run_figure_06.py
```

Purpose:

- sweep Eve's direction over 360 degrees;
- compare Eve-side Rician factors;
- display directional vulnerability in a polar plot;
- validate quadrature with Monte Carlo marks.

### Figure 7: Vulnerability-Bound Validation

```powershell
python scripts\run_figure_07.py
```

Configuration:

```text
configs/figures/figure_07_bounds.yaml
```

Scientific module:

```text
fa_ddm/bounds.py
```

Standalone figure:

```text
overleaf/figures/figure_07_vulnerability_bounds.tex
```

Purpose:

- evaluate exact QPSK posterior vulnerability;
- evaluate total-variation lower and upper bounds;
- evaluate the permutation-based vulnerability upper bound;
- use the same quadrature grid for exact vulnerability and total variation;
- solve component matching with linear assignment;
- export raw and probability-capped upper bounds.

The generated data columns are:

```text
eve_snr_db
vulnerability_exact
tv_lower_bound
tv_upper_bound_raw
tv_upper_bound_capped
permutation_upper_bound_raw
permutation_upper_bound_capped
prior_vulnerability
```

Typical output:

```text
results/figure_07_bounds/<run_reference>/
├── figure_07_bounds.dat
├── config_used.yaml
└── figure_07_bounds_preview.pdf
```

---

## Computational Cost

The most expensive operations are:

- two-dimensional quadrature over Eve's complex observation plane;
- likelihood evaluation for multiple symbols and ports;
- spatial maps with many receiver positions;
- fading averages over many channel realizations;
- linear programming with one epigraph variable per quadrature node;
- pairwise total-variation integration;
- repeated component matching across symbols and SNR points.

Start development runs with reduced quadrature, Monte Carlo, spatial-grid, and fading settings. Restore the final numerical settings only after validating the pipeline.

---

## Reproducibility Checklist

Every final result should satisfy the following:

1. The repository has no unintended uncommitted changes.
2. The YAML, runner, and scientific modules are committed.
3. The random seed is explicitly stored.
4. The output directory records the Git commit.
5. The exact YAML is copied as `config_used.yaml`.
6. `.dat` files use descriptive named columns.
7. Missing values are represented by `nan`.
8. The Python preview has been inspected.
9. The selected `.dat` file has been copied to Overleaf.
10. The Overleaf figure uses named columns.
11. The manuscript caption is consistent with the displayed data.

Recommended final workflow:

```powershell
git status
python -m pytest tests -v
python scripts\run_figure_XX.py
git add .
git diff --cached --stat
git commit -m "Describe the completed experiment"
git push origin main
```

If an experiment is run before committing, repeat the final run after the repository is clean so the result directory does not include `_dirty`.

---

## Common Problems

### `pytest` collects tests from `matplotlib`

Use:

```powershell
python -m pytest tests -v
```

or configure `pytest.ini` to exclude `fa-ddm-env`.

### `ModuleNotFoundError: No module named 'fa_ddm'`

```powershell
python -m pip install -e . --no-build-isolation
```

### PyCharm uses the wrong interpreter

```powershell
python -c "import sys; print(sys.executable)"
```

The path should point to `<repository>/fa-ddm-env/python.exe`.

### PGFPlots reports unbalanced columns

Confirm that missing values are exported as `nan` and that every row contains the same number of fields as the header.

### PGFPlots cannot find a column

Inspect the first noncomment line of the `.dat` file and verify the exact column name.

### Overleaf reads metadata as data

When using `pgfplotstable`, specify:

```latex
comment chars={\#}
```

### Result directory contains `_dirty`

Commit the source and configuration changes, then repeat the final experiment.

---

## Modeling Assumptions and Limitations

- Alice knows Bob's instantaneous channel at every candidate port.
- Instantaneous Eve-side CSI is required only for channel-adaptive privacy optimization and represents a full-information benchmark.
- Statistical Eve-side CSI can be handled through expected vulnerability because expectation preserves convexity.
- The active port is not disclosed to Eve before the symbol decision.
- Eve knows the protocol, distributions, channel model, and optimal likelihood rule.
- The channel is narrowband.
- Symbol-wise switching is idealized.
- Block-wise activation would require joint inference over observations sharing the same latent port.
- Mutual coupling, impedance mismatch, switching transients, and hardware impairments are not modeled.
- Clarke-Jakes correlation represents isotropic diffuse scattering.
- Direction enters primarily through the deterministic Rician component.
- Independent Bob and Eve diffuse samples do not enforce identical channels at co-located positions.
- Posterior vulnerability is a one-guess identity-recovery metric. Broader gain functions are outside the current implementation.

---

## Extending the Repository

When adding a new figure:

1. Add reusable scientific functions under `fa_ddm/`.
2. Add focused unit tests under `tests/`.
3. Add one YAML file under `configs/figures/`.
4. Add one runner under `scripts/`.
5. Export named `.dat` columns.
6. Generate a Python PDF preview.
7. Add a standalone PGFPlots source under `overleaf/figures/`.
8. Use the standard versioned result-directory function.
9. Update this README.

---

## Citation

A formal citation entry should be added after the associated manuscript is publicly available. Until then, cite the repository by title and Git commit hash:

```text
FA-DDM Information Leakage, numerical simulation repository,
commit <short_git_commit>.
```

## License

Add a repository license before public release. The selected license should be consistent with the manuscript, institutional requirements, and third-party dependencies.

## Contact and Contributions

Contributions should include:

- a clear scientific motivation;
- a YAML configuration when applicable;
- focused tests;
- PGFPlots-compatible output;
- documentation updates.

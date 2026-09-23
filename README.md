# FA-DDM Information Leakage

Simulation and numerical-analysis framework for **Fluid-Antenna-Assisted Dynamic Directional Modulation (FA-DDM)**, with emphasis on **posterior vulnerability**, **min-entropy information leakage**, **spatial correlation**, **directional dependence**, and **privacy-aware port selection**.

The repository supports the numerical experiments and manuscript figures for studying how random fluid-antenna port activation and Bob-oriented phase compensation affect an optimal eavesdropper.

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
- finite-SNR total-variation bounds;
- privacy-aware port-selection optimization by linear programming;
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

The active port is not disclosed to Eve. Conditioned on a confidential symbol, Eve's observation is consequently a Gaussian mixture over all possible active ports.

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

The repository evaluates these quantities directly from Eve's complete secret-conditioned Gaussian-mixture likelihoods. It does not assess Eve through a conventional nearest-neighbor detector unless such a detector is explicitly introduced as a benchmark.

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
│       └── figure_06_directional_vulnerability.yaml
│
├── scripts/
│   ├── run_figure_01.py
│   ├── run_figure_02.py
│   ├── run_figure_03.py
│   ├── run_figure_04.py
│   ├── run_figure_05.py
│   └── run_figure_06.py
│
├── fa_ddm/
│   ├── __init__.py
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
│       ├── figure_05_privacy_reliability.tex
│       └── figure_06_directional_vulnerability.tex
│
├── results/
│   └── <experiment_name>/
│       └── <YYYYMMDD>_<git_commit>[_dirty]/
│           ├── <experiment_name>.dat
│           ├── config_used.yaml
│           ├── <experiment_name>_preview.pdf
│           └── additional experiment-specific `.dat` files
│
├── tests/
│   ├── test_channel.py
│   ├── test_distinguishability.py
│   ├── test_geometry.py
│   ├── test_import.py
│   ├── test_io_nan_fix.py
│   ├── test_modulation.py
│   ├── test_modulation_step08.py
│   ├── test_optimization.py
│   ├── test_quadrature.py
│   ├── test_rician_channel.py
│   ├── test_spatial_map.py
│   └── test_vulnerability.py
│
├── .gitignore
├── pyproject.toml
├── requirements.txt
├── setup.py
└── README.md
```

The exact list of tests may evolve as new experiments are added. The scientific modules under `fa_ddm/` should remain independent of figure-specific presentation code.

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

The environment directory must not be committed to Git.

The `.gitignore` file should include:

```gitignore
fa-ddm-env/
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

The first Python path should point to the project-local environment, and the version should be `Python 3.8.8`.

### Configure PyCharm

In PyCharm:

1. Open `File > Settings > Project > Python Interpreter`.
2. Select `Add Interpreter`.
3. Select `Add Local Interpreter`.
4. Choose an existing Conda environment.
5. Select:

```text
<repository>/fa-ddm-env/python.exe
```

6. Apply the configuration.
7. Open a new PyCharm terminal.

Verify inside the PyCharm terminal:

```powershell
python -c "import sys; print(sys.executable)"
python --version
```

---

## Installation

The repository includes `setup.py` for compatibility with the packaging tools available in the Python 3.8.8 environment.

From the repository root:

```powershell
python -m pip install -e . --no-build-isolation
```

The editable installation allows changes under `fa_ddm/` to be used immediately without reinstalling the package after every edit.

Verify the package import:

```powershell
python -c "import fa_ddm; print(fa_ddm.__file__)"
```

The printed path should point to:

```text
<repository>/fa_ddm/__init__.py
```

### Core dependencies

The repository uses:

- `numpy`
- `scipy`
- `matplotlib`
- `pandas`
- `PyYAML`
- `pytest`

The dependencies are declared in `pyproject.toml` and `requirements.txt`.

---

## Running the Test Suite

Run all tests from the repository root:

```powershell
python -m pytest tests -v
```

Use `python -m pytest` instead of plain `pytest` to ensure that the tests use the active project interpreter.

### Main validation groups

The tests cover:

- constellation size and normalization;
- uniform symbol probabilities;
- fluid-antenna port geometry;
- Clarke-Jakes matrix symmetry and positive semidefiniteness;
- deterministic directional signatures;
- spatially correlated Rician channel power normalization;
- random-seed reproducibility;
- phase compensation at Bob;
- Eve's effective channel definition;
- Gaussian-mixture likelihood dimensions;
- MAP decision validity;
- Monte Carlo vulnerability limits;
- deterministic quadrature accuracy;
- quadrature versus Monte Carlo agreement;
- three-dimensional receiver geometry;
- spatial map ordering;
- finite-SNR total-variation bounds;
- binary vulnerability identity;
- privacy-aware optimization feasibility;
- optimization probability normalization;
- rectangular `.dat` export when values are missing.

### Syntax checks

To check all main Python modules:

```powershell
python -m py_compile `
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

From the repository root:

```powershell
python -c "from pathlib import Path; import yaml; paths=list(Path('configs').rglob('*.yaml')); [yaml.safe_load(p.read_text(encoding='utf-8')) for p in paths]; print('YAML files OK:', len(paths))"
```

---

## Configuration System

Every manuscript experiment has one YAML file under:

```text
configs/figures/
```

The YAML file controls:

- random seed;
- modulation;
- number of ports;
- aperture length;
- Bob and Eve directions or positions;
- large-scale gains;
- Rician factors;
- transmit power;
- receiver noise variance;
- SNR sweep;
- spatial grid;
- quadrature resolution;
- Monte Carlo trial count;
- reliability thresholds;
- output file names.

The exact YAML used in an execution is copied to the result folder as:

```text
config_used.yaml
```

This ensures that every result can be associated with both the source-code version and the numerical parameters.

---

## Versioned Result Directories

Every experiment run creates a versioned directory:

```text
results/<experiment_name>/<YYYYMMDD>_<short_git_commit>/
```

Example:

```text
results/figure_01_validation/20260923_a1b2c3d/
```

If the repository contains uncommitted changes, the suffix `_dirty` is added:

```text
results/figure_01_validation/20260923_a1b2c3d_dirty/
```

A `_dirty` result is useful during development, but final manuscript results should normally be generated from a clean committed state.

### Recommended execution sequence

```powershell
git status
git add .
git commit -m "Describe the experiment update"
git push
python scripts\run_figure_XX.py
```

The resulting directory contains:

- a PGFPlots-ready `.dat` file;
- a copy of the YAML configuration;
- a Python-generated PDF preview;
- additional `.dat` files when a figure has multiple panels with different table structures.

---

## Data Export Convention

The `.dat` file is the primary scientific output. The Python-generated PDF is only a preview for checking trends, ranges, and obvious implementation errors.

Data files are whitespace-separated and have descriptive column names. Example:

```text
eve_snr_db vulnerability_qpsk_theory vulnerability_qpsk_mc vulnerability_qpsk_ci95_lower vulnerability_qpsk_ci95_upper
```

Metadata lines begin with `#`:

```text
# Experiment: figure_01_validation
# Modulations: BPSK, QPSK, 8PSK, 16QAM
```

LaTeX readers must treat `#` as a comment character when loading the full table with `pgfplotstable`.

Missing numerical values are written explicitly as:

```text
nan
```

This is important because empty fields would make space-separated tables nonrectangular.

The exporter uses the equivalent of:

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

The repository preserves complete versioned experiment outputs. The Overleaf project receives only the selected final `.dat` files.

Copy selected data files to:

```text
fig/data/
```

Examples:

```text
fig/data/figure_01_validation.dat
fig/data/figure_02_eve_position_map.dat
fig/data/figure_03_port_density_saturation.dat
fig/data/figure_04_finite_snr_bounds.dat
fig/data/figure_05_privacy_reliability.dat
fig/data/figure_05_port_probabilities.dat
fig/data/figure_06_directional_vulnerability.dat
```

The standalone LaTeX files use stable paths such as:

```latex
\newcommand{\figureonedata}{fig/data/figure_01_validation.dat}
```

The final manuscript figures are generated in Overleaf using TikZ and PGFPlots.

### Plotting convention

All manuscript plots follow these conventions:

1. Analytical or deterministic-quadrature results are lines.
2. Different theoretical cases use different colors and line styles.
3. Each plot should contain multiple theoretically meaningful curves whenever possible.
4. Monte Carlo results use `only marks`.
5. Monte Carlo confidence intervals use error bars.
6. Monte Carlo results do not receive separate legend entries.
7. Figure captions or manuscript text explain that markers denote Monte Carlo simulation.
8. The Python PDF preview follows approximately the same convention but is not the publication figure.

---

## Supported Modulations

The modulation factory supports:

```text
BPSK
QPSK
8PSK
16PSK
16QAM
64QAM
```

Modulation names are case-insensitive and may contain spaces or hyphens.

Examples:

```python
from fa_ddm.modulation import get_constellation

bpsk = get_constellation("BPSK")
qpsk = get_constellation("QPSK")
qam16 = get_constellation("16-QAM")
```

All constellations are normalized to unit average symbol energy:

```text
E[|S|^2] = 1.
```

---

## Channel and Geometry Models

### Fluid-antenna ports

For a one-dimensional aperture of normalized length `W/lambda`, the candidate ports are uniformly distributed over the aperture.

The correlation between ports `n` and `m` is modeled as:

```text
R[n,m] = J_0(2*pi*|x_n-x_m|),
```

where positions are normalized by wavelength.

### Spatially correlated Rician channel

For receiver `q`, the channel is generated from:

```text
h_q = sqrt(beta_q) * [
    sqrt(kappa_q/(kappa_q+1)) * a_q
    + sqrt(1/(kappa_q+1)) * R_q^(1/2) w_q
].
```

The normalization ensures:

```text
E[|h_q,n|^2] = beta_q.
```

### Directional signature

For a linear aperture and a broadside angular convention:

```text
a_q[n] = exp(-j*2*pi*x_n*sin(theta_q)).
```

For three-dimensional coordinates:

```text
a_q[n] = exp(-j*2*pi*r_n^T*u_q).
```

The spatial correlation matrix describes similarity among diffuse samples at neighboring ports. Receiver direction enters through the deterministic spatial signature.

---

## Vulnerability Evaluation

### Gaussian-mixture centroids

For confidential symbol `s_m` and port `n`:

```text
mu[m,n] = sqrt(P_t) * g_n * s_m.
```

### Weighted symbol likelihood

For a symbol prior `p_S(s_m)`:

```text
ell_m(y; rho) = p_S(s_m) * p(y|s_m).
```

Eve's optimal decision is:

```text
argmax_m ell_m(y; rho).
```

### Deterministic quadrature

Posterior vulnerability is approximated on a rectangular grid over the complex observation plane:

```text
V_hat = sum_r omega_r * max_m ell_m(y_r; rho).
```

The grid is automatically centered around all Gaussian-mixture centroids and extended by a configurable noise margin.

### Monte Carlo validation

The Monte Carlo estimator generates:

- a confidential symbol;
- an active port;
- complex Gaussian receiver noise;
- Eve's observation;
- Eve's optimal MAP decision.

The estimator is the empirical fraction of correct optimal guesses.

Monte Carlo is used to validate deterministic calculations. It is not used as a substitute for the optimal likelihood model.

---

## Figure Experiments

## Figure 1: Multiple-Modulation Validation

Configuration:

```text
configs/figures/figure_01_validation.yaml
```

Runner:

```text
scripts/run_figure_01.py
```

Run:

```powershell
python scripts\run_figure_01.py
```

Purpose:

- validate quadrature against Monte Carlo;
- compare BPSK, QPSK, 8PSK, and 16QAM;
- verify low-SNR behavior;
- verify consistent likelihood implementation across modulations.

Typical output:

```text
results/figure_01_validation/<run_reference>/
├── figure_01_validation.dat
├── config_used.yaml
└── figure_01_validation_preview.pdf
```

---

## Figure 2: Eve Position Map

Configuration:

```text
configs/figures/figure_02_eve_position_map.yaml
```

Runner:

```text
scripts/run_figure_02.py
```

Run:

```powershell
python scripts\run_figure_02.py
```

Purpose:

- fix Bob at a known three-dimensional position;
- evaluate Eve over a planar grid;
- export distance, azimuth, and elevation;
- compare posterior vulnerability for QPSK and 16QAM;
- generate PGFPlots-compatible surface or heat-map data.

The red star in the final LaTeX figure represents Bob. Eve is represented by every evaluated point in the spatial map.

The `.dat` file includes:

```text
grid_row
grid_col
eve_x_lambda
eve_y_lambda
eve_z_lambda
eve_distance_lambda
eve_azimuth_deg
eve_elevation_deg
vulnerability_qpsk
min_entropy_leakage_qpsk
vulnerability_16qam
min_entropy_leakage_16qam
```

For a quick check, reduce the grid before the final run:

```yaml
map:
  x_points: 11
  y_points: 11

quadrature:
  points_per_axis: 61
```

---

## Figure 3: Fixed-Aperture Port-Density Saturation

Configuration:

```text
configs/figures/figure_03_port_density_saturation.yaml
```

Runner:

```text
scripts/run_figure_03.py
```

Run:

```powershell
python scripts\run_figure_03.py
```

Purpose:

- vary the number of ports while holding the physical aperture fixed;
- compare several aperture lengths;
- evaluate diminishing privacy changes as the port grid becomes dense;
- validate quadrature trends with Monte Carlo marks.

The current experiment compares:

```text
N = 2, 4, 8, 16, 32
W = 0.5 lambda, 1 lambda, 2 lambda
```

Theoretical curves are fading-averaged quadrature results. Monte Carlo marks are also averaged over channel realizations.

---

## Figure 4: Finite-SNR Distinguishability Bounds

Configuration:

```text
configs/figures/figure_04_finite_snr_bounds.yaml
```

Runner:

```text
scripts/run_figure_04.py
```

Run:

```powershell
python scripts\run_figure_04.py
```

Purpose:

- validate the exact binary total-variation identity for BPSK;
- compare the exact QPSK vulnerability with finite-SNR lower and upper bounds;
- validate exact curves using Monte Carlo marks.

The `.dat` exporter must write missing quantities explicitly as `nan`. The QPSK table contains no binary total-variation identity, so an empty field must never be written.

For LaTeX, `pgfplotstable` should load comment lines using:

```latex
\pgfplotstableread[
    col sep=space,
    comment chars={\#}
]{\figurefourdatafile}\figurefourdata
```

---

## Figure 5: Privacy-Reliability Tradeoff

Configuration:

```text
configs/figures/figure_05_privacy_reliability.yaml
```

Runner:

```text
scripts/run_figure_05.py
```

Run:

```powershell
python scripts\run_figure_05.py
```

Purpose:

- minimize Eve's quadrature vulnerability;
- constrain Bob's average symbol-error probability;
- trace the privacy-reliability tradeoff;
- inspect optimized port-selection probabilities.

The quadrature epigraph problem is solved as a linear program using `scipy.optimize.linprog`.

The experiment produces two data files:

```text
figure_05_privacy_reliability.dat
figure_05_port_probabilities.dat
```

The first contains the tradeoff curve. The second contains the optimized probability vectors for selected reliability thresholds.

---

## Figure 6: Directional Vulnerability

Configuration:

```text
configs/figures/figure_06_directional_vulnerability.yaml
```

Runner:

```text
scripts/run_figure_06.py
```

Run:

```powershell
python scripts\run_figure_06.py
```

Purpose:

- sweep Eve's direction over 360 degrees;
- compare several Eve-side Rician factors;
- display directional privacy in a polar plot;
- mark Bob's fixed direction;
- validate quadrature using Monte Carlo marks.

The current experiment compares:

```text
kappa_E = 0, 5, 20.
```

The same Eve diffuse realization is retained across the angular sweep. Directional variation is therefore primarily generated by the deterministic spatial signature.

For a quick development run:

```yaml
sweep:
  eve_angle_deg:
    step: 15.0

quadrature:
  points_per_axis: 41

monte_carlo:
  number_of_trials: 5000
```

---

## Running Experiments from Python

The runners expose a `run` function and can be imported from a Python console.

### Figure 1

```python
from scripts.run_figure_01 import run

result = run("configs/figures/figure_01_validation.yaml")
print(result.keys())
```

### Figure 2

```python
from scripts.run_figure_02 import run

result = run("configs/figures/figure_02_eve_position_map.yaml")
print(result.keys())
```

### Figure 3

```python
from scripts.run_figure_03 import run

result = run("configs/figures/figure_03_port_density_saturation.yaml")
print(result.keys())
```

### Figure 4

```python
from scripts.run_figure_04 import run

result = run("configs/figures/figure_04_finite_snr_bounds.yaml")
print(result.keys())
```

### Figure 5

```python
from scripts.run_figure_05 import run

result = run("configs/figures/figure_05_privacy_reliability.yaml")
print(result.keys())
```

### Figure 6

```python
from scripts.run_figure_06 import run

result = run("configs/figures/figure_06_directional_vulnerability.yaml")
print(result.keys())
```

---

## Computational Cost

The most expensive operations are:

- two-dimensional quadrature over Eve's complex observation plane;
- likelihood evaluation for multiple symbols and ports;
- spatial maps with many receiver positions;
- fading averages involving many channel realizations;
- linear-program optimization with one epigraph variable per quadrature node.

### Recommended development strategy

Start with reduced settings:

```yaml
quadrature:
  points_per_axis: 31
  noise_margin_sigma: 4.0

monte_carlo:
  number_of_trials: 1000
```

For spatial maps:

```yaml
map:
  x_points: 11
  y_points: 11
```

For fading averages:

```yaml
fading:
  number_of_realizations: 5
```

After verifying the pipeline, restore the final numerical settings and run from a clean Git commit.

---

## Reproducibility Guidelines

Every final result should satisfy the following checklist:

1. The repository has no unintended uncommitted changes.
2. The YAML file is committed.
3. The runner and scientific modules are committed.
4. The random seed is explicitly stored in the YAML.
5. The result folder contains the Git commit hash.
6. The exact YAML is copied as `config_used.yaml`.
7. The `.dat` file has descriptive column names.
8. Missing data are represented by `nan`.
9. The Python PDF preview has been inspected.
10. The selected `.dat` file has been copied to `fig/data/` in Overleaf.
11. The Overleaf figure uses named columns rather than numerical column indices.
12. The manuscript caption explains analytical lines, Monte Carlo marks, and confidence intervals.

---

## Git Workflow

Before generating final results:

```powershell
git status
git add .
git commit -m "Describe the completed experiment"
git push
```

Run the experiment:

```powershell
python scripts\run_figure_XX.py
```

Inspect the new result folder:

```text
results/<experiment>/<date_commit>/
```

If the folder includes `_dirty`, inspect the uncommitted changes before treating the results as final.

Commit selected result files if the repository policy is to preserve numerical outputs:

```powershell
git add results
git commit -m "Add numerical results for Figure XX"
git push
```

---

## Common Problems

### `ModuleNotFoundError: No module named 'fa_ddm'`

Install the repository in editable mode:

```powershell
python -m pip install -e . --no-build-isolation
```

Verify that `fa_ddm/__init__.py` exists.

### PyCharm uses the wrong interpreter

Verify:

```powershell
python -c "import sys; print(sys.executable)"
```

The path should point to:

```text
<repository>/fa-ddm-env/python.exe
```

### Permission errors under `C:\ProgramData\Anaconda3`

Do not install project packages into the shared Anaconda base environment. Use the project-local Conda environment.

### Editable install requires `setup.py`

The repository includes `setup.py` for compatibility with older packaging tools.

### GitHub rejects a password

GitHub HTTPS pushes require a browser-based credential manager or a personal access token. Password authentication is not supported.

### PGFPlots reports an unbalanced number of columns

Check whether a missing value was exported as an empty field. The exporter must use:

```python
na_rep="nan"
```

Every data row must contain the same number of whitespace-separated fields as the header.

### PGFPlots cannot find a column

Check the first noncomment line of the `.dat` file. Verify that the requested named column exists exactly, including capitalization.

### Overleaf reads metadata as data

When using `pgfplotstable`, specify:

```latex
comment chars={\#}
```

### Result directory contains `_dirty`

The code or configuration had uncommitted changes when the experiment started. Commit the changes and repeat the final run.

---

## Extending the Repository

When adding a new figure:

1. Add reusable scientific functions under `fa_ddm/`.
2. Add focused unit tests under `tests/`.
3. Add one YAML configuration under `configs/figures/`.
4. Add one runner under `scripts/`.
5. Export named `.dat` columns.
6. Generate a Python PDF preview.
7. Add a standalone PGFPlots file under `overleaf/figures/`.
8. Use the standard versioned result-directory function.
9. Update this README.

Suggested naming convention:

```text
configs/figures/figure_07_<topic>.yaml
scripts/run_figure_07.py
overleaf/figures/figure_07_<topic>.tex
results/figure_07_<topic>/<run_reference>/
```

---

## Modeling Notes and Limitations

The repository implements a research model with explicit assumptions:

- Alice knows Bob's instantaneous channel at every candidate port.
- The instantaneous active port is not disclosed to Eve.
- Eve knows the protocol, distributions, channel model, and optimal likelihood rule.
- The channel is narrowband.
- The switching operation is ideal.
- Mutual coupling, impedance mismatch, switching transients, and hardware impairments are not currently modeled.
- Clarke-Jakes correlation represents isotropic diffuse scattering.
- Direction enters primarily through the deterministic Rician component.
- A spatial map should preferably use a continuous underlying random field if co-located Bob and Eve behavior is studied.
- Independent diffuse samples for Bob and Eve do not enforce `h_E = h_B` at identical positions.
- Posterior vulnerability is a one-guess identity-recovery metric. Broader gain functions are outside the current implementation.

These assumptions should be considered when interpreting directional maps, high-SNR limits, and practical privacy claims.

---

## Recommended Maintenance Workflow

Whenever a scientific formula changes:

1. Update the corresponding function under `fa_ddm/`.
2. Add or update a unit test.
3. Run the complete test suite.
4. Regenerate affected results.
5. Compare the new `.dat` files against previous runs.
6. Update the manuscript text and figure captions.
7. Commit code and configuration before the final run.

Whenever the output format changes:

1. Update `fa_ddm/io.py`.
2. Add an export regression test.
3. Update the relevant PGFPlots files.
4. Verify that headers and row lengths are consistent.
5. Update this README.

Whenever configuration keys change:

1. Update all affected YAML files.
2. Update the corresponding runner.
3. Update this README.
4. Regenerate `config_used.yaml` through a fresh experiment run.

---

## Citation

A formal citation entry should be added after the associated manuscript is publicly available. Until then, cite the repository by its title and Git commit hash when reporting generated numerical results.

Suggested temporary form:

```text
FA-DDM Information Leakage, numerical simulation repository,
commit <short_git_commit>.
```

---

## License

Add the repository license before public release. The selected license should be consistent with the manuscript, institutional requirements, and third-party dependencies.

---

## Contact and Contributions

The repository is intended primarily for reproducible manuscript experiments. Contributions should be focused, tested, and accompanied by:

- a clear scientific motivation;
- a YAML configuration when applicable;
- unit tests;
- PGFPlots-compatible output;
- documentation updates.

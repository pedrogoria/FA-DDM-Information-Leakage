# FA-DDM Information Leakage

Numerical framework for vulnerability and information-leakage analysis of fluid-antenna-assisted dynamic directional modulation.

## Repository workflow

Each manuscript figure has:

1. one YAML configuration in `configs/figures/`;
2. one Python runner in `scripts/`;
3. one result directory in `data/results/`;
4. one `.dat` file for Overleaf/PGFPlots;
5. one PDF generated directly by Python;
6. one standalone LaTeX figure in `overleaf/figures/`.

## Installation in PyCharm

1. Open this repository as a PyCharm project.
2. Create a Python 3.10 or newer virtual environment.
3. Open the PyCharm terminal and run:

```bash
python -m pip install --upgrade pip
pip install -e .
```

## First figure

The first implementation will validate numerical quadrature against Monte Carlo estimation of Eve's posterior vulnerability.

```bash
python scripts/run_figure_01.py
```

Generated files will be written to:

```text
data/results/figure_01_validation/
```

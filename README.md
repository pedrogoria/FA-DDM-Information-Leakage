# Step 13: Fixed-aperture port-density saturation

Place files:

- `figure_03_port_density_saturation.yaml` -> `configs/figures/figure_03_port_density_saturation.yaml`
- `run_figure_03.py` -> `scripts/run_figure_03.py`
- `figure_03_port_density_saturation.tex` -> `overleaf/figures/figure_03_port_density_saturation.tex`

Quick test settings may be reduced to 5 fading realizations, 41 quadrature points per axis, and 1000 Monte Carlo trials per realization.

Run:

```powershell
python scripts\run_figure_03.py
```

Copy the selected data file to:

```text
fig/data/figure_03_port_density_saturation.dat
```

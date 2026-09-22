"""Helpers for spatial vulnerability maps."""

import numpy as np

from fa_ddm.geometry import receiver_geometry


def planar_receiver_grid(x_values, y_values, z_value):
    """Return a Cartesian receiver-position grid.

    The ordering is row-major: all x values are traversed for the first y
    value, then for the second y value, and so forth. This ordering is suitable
    for PGFPlots matrix plots when ``mesh_cols=len(x_values)`` is specified.
    """
    x_values = np.asarray(x_values, dtype=float)
    y_values = np.asarray(y_values, dtype=float)

    if x_values.ndim != 1 or x_values.size == 0:
        raise ValueError("x_values must be a nonempty 1-D array.")
    if y_values.ndim != 1 or y_values.size == 0:
        raise ValueError("y_values must be a nonempty 1-D array.")
    if not np.all(np.isfinite(x_values)) or not np.all(np.isfinite(y_values)):
        raise ValueError("grid coordinates must be finite.")
    if not np.isfinite(z_value):
        raise ValueError("z_value must be finite.")

    x_grid, y_grid = np.meshgrid(x_values, y_values, indexing="xy")
    z_grid = np.full_like(x_grid, float(z_value), dtype=float)
    positions = np.column_stack(
        (x_grid.ravel(), y_grid.ravel(), z_grid.ravel())
    )
    return positions


def position_descriptors(position):
    """Return distance, azimuth, elevation, and unit direction for a position."""
    distance, direction = receiver_geometry(position)
    azimuth = np.rad2deg(np.arctan2(direction[1], direction[0]))
    elevation = np.rad2deg(
        np.arctan2(direction[2], np.hypot(direction[0], direction[1]))
    )
    return distance, float(azimuth), float(elevation), direction


def min_entropy_leakage(vulnerability, constellation_order):
    """Return log2(V/V_prior) for a uniform secret."""
    if constellation_order <= 0:
        raise ValueError("constellation_order must be positive.")
    vulnerability = np.asarray(vulnerability, dtype=float)
    prior = 1.0 / float(constellation_order)
    if np.any(vulnerability < prior - 1e-12) or np.any(vulnerability > 1.0 + 1e-12):
        raise ValueError("vulnerability must lie between the prior and one.")
    return np.log2(np.maximum(vulnerability, prior) / prior)

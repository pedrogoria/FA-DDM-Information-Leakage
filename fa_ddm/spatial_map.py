"""
Spatial receiver-grid construction and information-leakage descriptors for
fluid-antenna-assisted dynamic directional modulation.

Author
------
Pedro E. Goria Silva

Module
------
fa_ddm.spatial_map

Purpose
-------
This module implements helper functions used to construct spatial vulnerability
maps for the fluid-antenna-assisted dynamic directional modulation (FA-DDM)
numerical framework.

The module provides reusable utilities for:

1. constructing a Cartesian grid of receiver positions over a plane in
   three-dimensional space;
2. converting each receiver position into distance, azimuth, elevation, and a
   unit propagation direction relative to the aperture origin; and
3. converting posterior vulnerability into min-entropy information leakage for
   a uniformly distributed confidential symbol.

The functions are intended for reproducible evaluation of Eve's vulnerability
over spatial grids and for exporting position-indexed data to plotting tools
such as TikZ and PGFPlots. The module contains no channel generation,
likelihood evaluation, numerical quadrature, plotting, or file-output logic.

Spatial-map interpretation
--------------------------
A spatial vulnerability map evaluates the operational disclosure metric at a
collection of candidate Eve positions while the transmitter geometry and Bob's
position remain fixed.

Each point in the generated grid represents one receiver location

    p_E = [x_E, y_E, z_E],

with all coordinates normalized by the carrier wavelength. The corresponding
channel, effective port coefficients, Gaussian-mixture likelihoods, posterior
vulnerability, and information leakage are calculated by other modules or
experiment runners.

The grid helper only constructs and orders the receiver positions. It does not
assume that channel samples at neighboring positions form a continuous random
field. Spatial consistency across positions depends on how the experiment
runner generates or couples the underlying channel realizations.

Planar receiver grid
--------------------
The function ``planar_receiver_grid`` constructs the Cartesian product of
one-dimensional x and y coordinate arrays at one fixed z coordinate.

For

    x_values = [x_1, ..., x_Nx],
    y_values = [y_1, ..., y_Ny],
    z_value  = z_0,

the output contains Nx times Ny receiver positions of the form

    [x_i, y_j, z_0].

The implementation uses ``numpy.meshgrid`` with ``indexing="xy"`` and flattens
the resulting grids in NumPy's default row-major order. Consequently, all x
values are traversed for the first y value, followed by all x values for the
second y value, and so forth.

The resulting ordering is

    [x_1, y_1, z_0],
    [x_2, y_1, z_0],
    ...,
    [x_Nx, y_1, z_0],
    [x_1, y_2, z_0],
    ...,
    [x_Nx, y_Ny, z_0].

This ordering is suitable for PGFPlots matrix and surface plots when the number
of mesh columns is set to

    mesh_cols = len(x_values).

Changing the flattening order or mesh-grid indexing convention would alter the
interpretation of the exported spatial map and must therefore be coordinated
with the plotting code.

Position descriptors
--------------------
The function ``position_descriptors`` characterizes one receiver position
relative to the default aperture origin used by ``receiver_geometry``.

For receiver position p and origin p_0, the displacement is

    d = p - p_0.

The returned distance and unit direction are

    r = ||d||_2,
    u = d / r.

The azimuth angle is calculated as

    azimuth = atan2(u_y, u_x),

and the elevation angle is calculated as

    elevation = atan2(u_z, sqrt(u_x^2 + u_y^2)).

Both angular descriptors are returned in degrees.

Under this convention:

- azimuth is measured in the x-y plane from the positive x-axis;
- positive azimuth rotates toward the positive y-axis;
- elevation is measured from the x-y plane;
- positive elevation points toward the positive z-axis; and
- negative elevation points toward the negative z-axis.

The function delegates receiver-position validation and direction
normalization to ``fa_ddm.geometry.receiver_geometry``. A receiver located at
the aperture origin is invalid because the propagation direction is undefined.

Posterior vulnerability
-----------------------
Posterior Bayes vulnerability is Eve's maximum probability of identifying the
confidential symbol correctly in one attempt after observing the channel
output.

For a uniformly distributed M-ary secret, the prior vulnerability is

    V(S) = 1 / M.

A valid posterior vulnerability therefore satisfies

    1 / M <= V(S | Y_E) <= 1.

The lower endpoint represents an observation that provides no one-guess
advantage over the prior. The upper endpoint represents complete one-guess
disclosure.

Min-entropy leakage
-------------------
The function ``min_entropy_leakage`` converts posterior vulnerability into the
min-entropy leakage

    L_inf(S -> Y_E)
        = log2(V(S | Y_E) / V(S)).

For a uniformly distributed M-ary secret, this becomes

    L_inf(S -> Y_E)
        = log2(M V(S | Y_E)).

The leakage range is

    0 <= L_inf(S -> Y_E) <= log2(M).

Zero leakage occurs when posterior vulnerability equals the prior
vulnerability. Maximum leakage occurs when posterior vulnerability equals one.

The implementation accepts either a scalar vulnerability or an array of
vulnerability values. It validates the values using a numerical tolerance and
applies

    maximum(vulnerability, prior)

before evaluating the logarithm. This operation prevents a negligible
floating-point deviation below the theoretical prior from producing a small
negative leakage value.

Public functions
----------------
planar_receiver_grid
    Constructs a row-major Cartesian grid of three-dimensional receiver
    positions over a plane with fixed z coordinate.

position_descriptors
    Returns receiver distance, azimuth, elevation, and unit propagation
    direction relative to the aperture origin.

min_entropy_leakage
    Converts scalar or array-valued posterior vulnerability into min-entropy
    leakage for a uniformly distributed finite secret.

Input conventions
-----------------
x_values
    Nonempty one-dimensional array of finite x coordinates normalized by
    wavelength.

y_values
    Nonempty one-dimensional array of finite y coordinates normalized by
    wavelength.

z_value
    Finite scalar z coordinate normalized by wavelength and shared by every
    point in the planar grid.

position
    Finite three-element receiver coordinate normalized by wavelength.

vulnerability
    Scalar or array-like posterior vulnerability. Values must lie between the
    uniform prior vulnerability and one, within the implemented numerical
    tolerance.

constellation_order
    Positive number of confidential symbols. The value determines the uniform
    prior vulnerability 1/M and maximum min-entropy leakage log2(M).

Returned quantities
-------------------
``planar_receiver_grid`` returns

    positions
        Real array with shape ``(Nx * Ny, 3)``. Rows are ordered by y value,
        with x varying fastest.

``position_descriptors`` returns

    distance
        Euclidean receiver distance from the aperture origin, normalized by
        wavelength.

    azimuth
        Azimuth angle in degrees, measured from the positive x-axis in the x-y
        plane.

    elevation
        Elevation angle in degrees, measured from the x-y plane.

    direction
        Three-element unit vector pointing from the aperture origin toward the
        receiver.

``min_entropy_leakage`` returns

    leakage
        NumPy scalar or array containing min-entropy leakage in bits.

PGFPlots ordering
-----------------
The receiver-grid ordering is designed for a table containing columns such as

    grid_row
    grid_col
    eve_x_lambda
    eve_y_lambda
    eve_z_lambda
    vulnerability

When plotted as matrix data, the x coordinate must vary fastest and the plotting
configuration must use the number of x coordinates as the mesh-column count.

For example, if ``x_values`` contains 41 entries, the corresponding PGFPlots
matrix plot should use

    mesh/cols=41

or the equivalent option expected by the selected plot type.

If a runner sorts the exported table after grid construction, the sort order
must preserve this matrix arrangement or the spatial image may appear
transposed, scrambled, or discontinuous.

Validation and numerical safeguards
-----------------------------------
The grid function validates that x and y coordinates are nonempty,
one-dimensional, and finite. The fixed z coordinate must also be finite.

The position descriptor relies on ``receiver_geometry`` to validate that the
position is a finite three-vector and does not coincide with the aperture
origin.

The leakage function requires a positive constellation order and validates the
posterior vulnerability interval using a tolerance of 1e-12. Values below the
prior by more than the tolerance or above one by more than the tolerance raise
a ``ValueError``.

The numerical floor at the prior is applied only after validation. It corrects
small floating-point deviations but does not silently accept materially invalid
vulnerability values.

Reproducibility
---------------
All functions in this module are deterministic and contain no random-number
generation. Identical coordinate arrays, receiver positions, vulnerability
values, and constellation orders produce identical outputs.

Spatial-map reproducibility additionally depends on the experiment runner,
channel random seed, channel-field construction, modulation, port-selection
law, quadrature settings, and exported row ordering.

Computational complexity
------------------------
For Nx x-coordinates and Ny y-coordinates, planar-grid construction requires
O(Nx Ny) operations and storage.

Position descriptors require constant-cost vector operations for one receiver
position. Applying the function independently over a complete spatial grid
therefore requires O(Nx Ny) operations.

Min-entropy leakage evaluation is elementwise and requires O(K) operations for
K vulnerability values.

Modeling assumptions and limitations
------------------------------------
- Spatial coordinates and distances are normalized by carrier wavelength.
- The receiver grid is planar and uses one fixed z coordinate.
- Geometry alone does not define path loss, line-of-sight signatures, diffuse
  fading, or spatial channel correlation.
- The azimuth convention is referenced to the positive x-axis rather than a
  communications-specific broadside angle.
- The elevation convention is referenced to the x-y plane.
- The min-entropy leakage helper assumes a uniformly distributed confidential
  symbol.
- Broader gain functions, nonuniform priors, multiple guesses, list recovery,
  and semantic losses are not represented.
- A map generated from independent channel samples at each receiver position
  is not a continuous spatial random field.
- Co-located Bob and Eve behavior depends on the complete channel-generation
  model and cannot be inferred from position descriptors alone.

Notes
-----
The min-entropy leakage calculation uses base-two logarithms and therefore
returns leakage in bits.

For a uniform M-ary secret, the maximum possible leakage is log2(M), not one,
unless M = 2.

The row-major grid order is part of the data interface between the scientific
runner and the PGFPlots figure. Changes to this ordering should be accompanied
by updated tests and plotting code.

See Also
--------
fa_ddm.geometry
    Three-dimensional aperture coordinates, receiver geometry, far-field
    signatures, and normalized large-scale gains.

fa_ddm.channel
    Spatial correlation, Rician channel generation, phase precoding, and
    effective Eve-side channels.

fa_ddm.vulnerability
    Gaussian-mixture likelihoods and posterior-vulnerability estimation.

fa_ddm.quadrature
    Deterministic integration of Eve's posterior vulnerability.

scripts.run_figure_02
    Reproducible spatial vulnerability-map experiment.
"""


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

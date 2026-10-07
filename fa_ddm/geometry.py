"""
Three-dimensional geometry and propagation utilities for
fluid-antenna-assisted dynamic directional modulation.

Author
------
Pedro E. Goria Silva

Module
------
fa_ddm.geometry

Purpose
-------
This module implements reusable three-dimensional geometry functions for the
fluid-antenna-assisted dynamic directional modulation (FA-DDM) simulation and
numerical-analysis framework.

The module provides tools for:

1. constructing centered linear fluid-antenna apertures in three-dimensional
   Cartesian coordinates;
2. converting receiver positions into propagation distances and unit direction
   vectors relative to an aperture reference point;
3. evaluating deterministic three-dimensional far-field spatial signatures;
   and
4. computing normalized distance-dependent large-scale power gains.

The functions are designed to support spatial vulnerability maps, directional
channel models, Bob and Eve placement, and reproducible manuscript
experiments. Geometry and propagation calculations are kept independent of
channel realization, vulnerability evaluation, plotting, and file-output
logic.

Coordinate system
-----------------
All spatial coordinates are expressed in Cartesian form as

    r = [x, y, z],

and are normalized by the carrier wavelength. Consequently, distances returned
by this module are also measured in wavelengths.

The aperture origin is the reference point from which receiver displacement,
distance, and direction are evaluated. Unless specified otherwise, the origin
is

    [0, 0, 0].

The coordinate system itself does not impose a specific broadside direction.
The physical interpretation of the x, y, and z axes is determined by the
experiment configuration and the selected aperture orientation.

Centered linear aperture
------------------------
The function ``linear_aperture_coordinates`` creates N uniformly spaced ports
along one Cartesian axis. The aperture is centered at the origin, and the
first and last ports are separated by the normalized aperture length W/lambda.

For an aperture aligned with the x-axis, the coordinates are

    r_n = [x_n, 0, 0],

where

    x_n in [-W/(2 lambda), W/(2 lambda)].

Equivalent constructions are available along the y-axis and z-axis. When only
one port is requested, the port is placed at the origin.

This centered-coordinate convention is particularly convenient for spatial
maps and far-field phase calculations because it prevents an arbitrary global
phase shift caused by placing one aperture endpoint at the origin.

Receiver geometry
-----------------
For receiver position p_z and aperture reference point p_0, the displacement is

    d_z = p_z - p_0.

The propagation distance in wavelengths is

    r_z = ||d_z||_2,

and the corresponding unit direction vector is

    u_z = d_z / r_z.

The function ``receiver_geometry`` returns the pair

    (r_z, u_z).

The receiver position must not coincide with the aperture origin because the
direction vector would otherwise be undefined.

Three-dimensional far-field signature
-------------------------------------
For candidate-port coordinate r_n and unit propagation direction u_z, the
deterministic far-field spatial signature is

    a_z[n] = exp(-j 2 pi r_n^T u_z).

Because the port coordinates are normalized by wavelength, the wave number
appears as 2 pi without an explicit wavelength denominator.

The function ``far_field_signature_3d`` accepts any nonzero finite direction
vector and normalizes the vector internally. This permits direct use of either
a unit direction returned by ``receiver_geometry`` or an arbitrary vector
pointing toward the receiver.

The output contains one unit-magnitude complex coefficient per candidate port.
The phase variation across the aperture captures the deterministic directional
structure used by the Rician line-of-sight channel model.

Large-scale propagation gain
----------------------------
The function ``free_space_large_scale_gain`` implements the normalized power-
law model

    beta(d) = beta_ref (d_ref / d)^eta,

where:

    d           is the receiver distance in wavelengths;
    beta_ref    is the power gain at the reference distance;
    d_ref       is the reference distance in wavelengths;
    eta         is the path-loss exponent.

The default exponent is eta = 2, corresponding to free-space power decay. The
function is intentionally normalized and does not include an explicit Friis
constant, antenna gains, carrier-frequency conversion, shadowing, or additional
link-budget terms.

Public functions
----------------
linear_aperture_coordinates
    Returns centered three-dimensional coordinates for uniformly spaced ports
    on a linear aperture aligned with the x, y, or z axis.

receiver_geometry
    Computes receiver distance and unit direction from the aperture origin or
    another user-specified reference point.

far_field_signature_3d
    Evaluates the deterministic unit-modulus far-field signature
    exp(-j 2 pi r_n^T u) for every port.

free_space_large_scale_gain
    Evaluates a normalized distance-dependent large-scale power gain using a
    configurable reference gain, reference distance, and path-loss exponent.

Input conventions
-----------------
number_of_ports
    Positive integer giving the number of candidate fluid-antenna ports.

aperture_wavelengths
    Nonnegative aperture length normalized by the carrier wavelength.

axis
    Cartesian aperture axis specified as ``"x"``, ``"y"``, or ``"z"``.

receiver_position_wavelengths
    Finite three-element receiver position normalized by wavelength.

aperture_origin
    Optional finite three-element aperture reference point. The default is the
    Cartesian origin.

port_coordinates_wavelengths
    Finite array with shape ``(N, 3)`` containing one normalized Cartesian
    coordinate per candidate port.

direction_vector
    Finite nonzero three-element vector. The function normalizes the vector
    internally before evaluating the far-field phase.

distance_wavelengths
    Strictly positive receiver distance normalized by wavelength.

reference_gain
    Strictly positive large-scale power gain defined at the reference
    distance.

reference_distance_wavelengths
    Strictly positive reference distance normalized by wavelength.

path_loss_exponent
    Strictly positive power-law attenuation exponent.

Returned quantities
-------------------
``linear_aperture_coordinates`` returns

    coordinates
        Real array with shape ``(N, 3)``. The aperture is centered at the
        origin and varies only along the selected axis.

``receiver_geometry`` returns

    distance_wavelengths
        Scalar Euclidean distance from the aperture origin to the receiver.

    direction_vector
        Three-element unit vector pointing from the aperture origin toward the
        receiver.

``far_field_signature_3d`` returns

    signature
        Complex array with one unit-magnitude coefficient per candidate port.

``free_space_large_scale_gain`` returns

    gain
        Scalar normalized large-scale channel power gain.

Validation and numerical safeguards
-----------------------------------
The module validates integer port counts, nonnegative aperture lengths, valid
axis names, finite Cartesian coordinates, nonzero receiver displacement,
nonzero direction vectors, positive distances, positive reference gains, and
positive path-loss exponents.

Direction vectors are normalized before use. This prevents the far-field phase
from depending on the arbitrary magnitude of a user-supplied direction vector.

A receiver located exactly at the aperture origin is rejected because both the
propagation direction and the power-law path-loss model are singular there.

Reproducibility
---------------
All functions in this module are deterministic and contain no random-number
generation. Identical inputs produce identical coordinates, distances,
directions, signatures, and large-scale gains.

Experiment runners are responsible for recording spatial-grid parameters,
receiver locations, aperture orientation, and propagation constants in their
YAML configuration files.

Computational complexity
------------------------
The aperture-coordinate and far-field-signature functions require O(N)
operations for N candidate ports. Receiver geometry and large-scale gain are
constant-cost scalar computations.

The module does not construct dense pairwise-distance or correlation matrices.
Spatial correlation is handled separately by ``fa_ddm.channel``.

Modeling assumptions and limitations
------------------------------------
- Spatial coordinates and distances are normalized by carrier wavelength.
- The aperture is linear and aligned with one Cartesian axis.
- The far-field model assumes a plane wave across the complete aperture.
- Near-field spherical-wave curvature is not modeled.
- The large-scale gain uses a normalized power law rather than a complete link
  budget.
- Antenna gains, polarization, blockage, shadowing, atmospheric loss, mutual
  coupling, and hardware impairments are not included.
- A spatial map constructed from independent channel draws does not by itself
  represent a continuous physical random field across receiver positions.
- Co-located Bob and Eve behavior depends on how the experiment couples their
  channel realizations and is not determined by geometry alone.

Notes
-----
The coordinate convention in this module differs from the endpoint-based
one-dimensional coordinates returned by ``fa_ddm.channel.linear_port_positions``.
The geometry module centers the aperture at the origin, whereas the channel
helper places the ports over the interval from zero to the normalized aperture
length. Experiments should use one convention consistently when combining
coordinates, signatures, and correlation matrices.

The deterministic signature captures directional phase structure only. The
complete channel model may additionally include large-scale gain, a Rician
factor, a diffuse spatial correlation matrix, and random fading samples.

See Also
--------
fa_ddm.channel
    One-dimensional aperture positions, Clarke-Jakes correlation, Rician
    channel generation, phase precoding, and Eve-side effective channels.

fa_ddm.spatial_map
    Utilities for evaluating vulnerability over receiver-position grids.

fa_ddm.vulnerability
    Gaussian-mixture likelihood and posterior-vulnerability evaluation.

scripts.run_figure_02
    Reproducible Eve-position vulnerability-map experiment.

scripts.run_figure_06
    Reproducible directional-vulnerability experiment.
"""


import numpy as np


def linear_aperture_coordinates(number_of_ports, aperture_wavelengths, axis="x"):
    """Return centered 3-D port coordinates normalized by wavelength.

    The linear aperture is centered at the origin. The first and last ports are
    separated by ``aperture_wavelengths``.
    """
    if not isinstance(number_of_ports, (int, np.integer)):
        raise TypeError("number_of_ports must be an integer.")
    if number_of_ports <= 0:
        raise ValueError("number_of_ports must be positive.")
    if aperture_wavelengths < 0.0:
        raise ValueError("aperture_wavelengths must be nonnegative.")
    if axis not in ("x", "y", "z"):
        raise ValueError("axis must be 'x', 'y', or 'z'.")

    coordinates = np.zeros((number_of_ports, 3), dtype=float)
    if number_of_ports == 1:
        return coordinates

    values = np.linspace(
        -0.5 * float(aperture_wavelengths),
        0.5 * float(aperture_wavelengths),
        number_of_ports,
    )
    axis_index = {"x": 0, "y": 1, "z": 2}[axis]
    coordinates[:, axis_index] = values
    return coordinates


def receiver_geometry(receiver_position_wavelengths, aperture_origin=None):
    """Return receiver distance and direction from the aperture origin.

    Parameters
    ----------
    receiver_position_wavelengths : array_like
        Receiver position [x, y, z], normalized by wavelength.
    aperture_origin : array_like or None
        Aperture reference point. The default is [0, 0, 0].

    Returns
    -------
    tuple
        ``(distance_wavelengths, direction_vector)``.
    """
    receiver = np.asarray(receiver_position_wavelengths, dtype=float)
    if receiver.shape != (3,) or not np.all(np.isfinite(receiver)):
        raise ValueError("receiver_position_wavelengths must be a finite 3-vector.")

    if aperture_origin is None:
        origin = np.zeros(3, dtype=float)
    else:
        origin = np.asarray(aperture_origin, dtype=float)
        if origin.shape != (3,) or not np.all(np.isfinite(origin)):
            raise ValueError("aperture_origin must be a finite 3-vector.")

    displacement = receiver - origin
    distance = float(np.linalg.norm(displacement))
    if distance <= 0.0:
        raise ValueError("receiver must not coincide with the aperture origin.")
    return distance, displacement / distance


def far_field_signature_3d(port_coordinates_wavelengths, direction_vector):
    """Return exp(-j*2*pi*r_n^T*u) for a 3-D far-field direction."""
    coordinates = np.asarray(port_coordinates_wavelengths, dtype=float)
    direction = np.asarray(direction_vector, dtype=float)

    if coordinates.ndim != 2 or coordinates.shape[1] != 3:
        raise ValueError("port_coordinates_wavelengths must have shape (N, 3).")
    if coordinates.shape[0] == 0 or not np.all(np.isfinite(coordinates)):
        raise ValueError("port coordinates must be nonempty and finite.")
    if direction.shape != (3,) or not np.all(np.isfinite(direction)):
        raise ValueError("direction_vector must be a finite 3-vector.")

    norm = np.linalg.norm(direction)
    if norm <= 0.0:
        raise ValueError("direction_vector must be nonzero.")
    unit_direction = direction / norm
    phase = -2.0 * np.pi * coordinates.dot(unit_direction)
    return np.exp(1j * phase).astype(np.complex128)


def free_space_large_scale_gain(distance_wavelengths, reference_gain=1.0,
                                reference_distance_wavelengths=1.0,
                                path_loss_exponent=2.0):
    """Return a normalized distance-dependent large-scale power gain."""
    if distance_wavelengths <= 0.0:
        raise ValueError("distance_wavelengths must be positive.")
    if reference_gain <= 0.0 or reference_distance_wavelengths <= 0.0:
        raise ValueError("reference gain and distance must be positive.")
    if path_loss_exponent <= 0.0:
        raise ValueError("path_loss_exponent must be positive.")
    ratio = float(reference_distance_wavelengths) / float(distance_wavelengths)
    return float(reference_gain) * ratio ** float(path_loss_exponent)

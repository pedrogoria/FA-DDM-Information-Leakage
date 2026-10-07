"""
Fluid-antenna geometry, spatial correlation, and Rician channel models.

Author
------
Pedro E. Goria Silva

Module
------
fa_ddm.channel

Purpose
-------
This module implements the physical-layer channel primitives used by the
fluid-antenna-assisted dynamic directional modulation (FA-DDM) simulation and
numerical-analysis framework.

The module provides reusable functions for:

1. generating uniformly spaced candidate-port positions over a linear fluid-
   antenna aperture;
2. constructing Clarke-Jakes spatial correlation matrices for diffuse
   scattering;
3. evaluating the minimum eigenvalue of real symmetric matrices;
4. constructing deterministic far-field signatures for linear apertures;
5. computing numerically stable square roots of Hermitian positive-
   semidefinite matrices;
6. generating spatially correlated Rician channel realizations;
7. constructing Bob-oriented phase-only precoding coefficients; and
8. deriving Eve's effective port-dependent channel after Bob-oriented phase
   compensation.

The functions are designed to support reproducible manuscript experiments and
to separate scientific channel modeling from figure-specific simulation code.

Physical model
--------------
Alice employs one radio-frequency chain connected to a reconfigurable
radiating element that can access multiple candidate positions over a bounded
linear aperture. Port coordinates are normalized by the carrier wavelength.
For N candidate ports and normalized aperture length W/lambda, the default
geometry is

    x_n in [0, W/lambda],

with uniformly spaced candidate positions, including both endpoints when
N > 1.

Spatial correlation
-------------------
Under two-dimensional isotropic diffuse scattering, the correlation between
ports n and m is modeled by the Clarke-Jakes expression

    R[n, m] = J_0(2 pi |x_n - x_m|),

where:

    J_0(.)      is the Bessel function of the first kind and order zero;
    x_n, x_m    are port positions normalized by wavelength.

The implementation explicitly symmetrizes the resulting matrix and restores a
unit diagonal to reduce numerical asymmetries caused by floating-point
evaluation.

The Clarke-Jakes matrix describes correlation among the diffuse channel
components. Receiver direction is introduced separately through the
deterministic far-field spatial signature.

Far-field directional signature
-------------------------------
For a linear aperture and an angle theta measured from broadside, the
normalized deterministic signature is

    a[n] = exp(-j 2 pi x_n sin(theta)).

The convention implies that theta = 0 degrees corresponds to broadside, for
which every port has the same deterministic phase. Nonzero angles create a
linear phase variation across the aperture.

The positions are dimensionless because they are normalized by wavelength,
and the input angle is specified in degrees.

Spatially correlated Rician channel
-----------------------------------
For receiver z, the generated channel follows

    h_z = sqrt(beta_z) [
              sqrt(kappa_z / (kappa_z + 1)) a_z
              + sqrt(1 / (kappa_z + 1)) R_z^(1/2) w_z
          ],

where:

    beta_z      is the large-scale channel power gain;
    kappa_z     is the Rician factor;
    a_z         is the deterministic unit-modulus far-field signature;
    R_z         is the diffuse spatial correlation matrix;
    w_z         contains independent unit-variance circularly symmetric
                complex Gaussian samples.

The normalization is selected so that

    E[|h_z,n|^2] = beta_z

for every candidate port when the supplied correlation matrix has unit
diagonal.

The special cases are:

    kappa_z = 0
        Spatially correlated Rayleigh fading.

    kappa_z -> infinity
        Deterministic line-of-sight propagation governed by the far-field
        signature.

Matrix square root
------------------
The correlated diffuse component requires a square root of the spatial
correlation matrix. The function ``hermitian_square_root`` first constructs the
Hermitian part of the input matrix, performs an eigenvalue decomposition, and
clips small negative eigenvalues caused by numerical roundoff.

A matrix is rejected when its smallest eigenvalue is below the negative
user-specified tolerance. This behavior prevents an invalid correlation matrix
from being silently converted into a positive-semidefinite matrix.

Bob-oriented phase precoding
----------------------------
For Bob's channel coefficient h_B,n at port n, the phase-only coefficient is

    q_n = conjugate(h_B,n) / |h_B,n|.

The coefficient has unit magnitude and removes the phase of Bob's channel.
Consequently,

    h_B,n q_n = |h_B,n|,

so Bob observes a phase-aligned confidential symbol with a port-dependent
amplitude.

The implementation rejects zero or numerically negligible Bob-channel
coefficients because their phase is undefined. The threshold is controlled by
``zero_tolerance``.

Eve-side effective channel
--------------------------
Eve's effective coefficient at port n is

    g_n = h_E,n q_n
        = h_E,n conjugate(h_B,n) / |h_B,n|.

This coefficient retains the interaction between Bob-oriented phase
compensation and Eve's spatial channel. When the active port is undisclosed,
the collection of coefficients {g_n} induces the port-dependent Gaussian-
mixture observation used by the vulnerability-analysis modules.

Public functions
----------------
linear_port_positions
    Returns uniformly spaced one-dimensional candidate-port positions
    normalized by wavelength.

clarke_jakes_correlation
    Constructs the Clarke-Jakes diffuse spatial correlation matrix from
    normalized port coordinates.

minimum_eigenvalue
    Returns the smallest eigenvalue of a real symmetric matrix. This helper is
    useful for validation and regression testing of correlation matrices.

far_field_signature
    Evaluates the deterministic far-field signature of a linear aperture for
    an angle measured from broadside.

hermitian_square_root
    Computes a stable Hermitian positive-semidefinite matrix square root using
    eigenvalue decomposition.

generate_correlated_rician_channel
    Generates one correlated Rician channel realization using the supplied
    NumPy random-number generator.

phase_precoder
    Computes Bob-oriented unit-modulus phase coefficients for all candidate
    ports.

effective_eve_channel
    Combines Eve's channel with Bob's phase precoder to obtain the effective
    port-dependent coefficients observed by Eve.

Input conventions
-----------------
Port positions
    One-dimensional real values normalized by carrier wavelength.

Angles
    Real values in degrees, measured from broadside.

Channel vectors
    One-dimensional complex arrays with one entry per candidate port.

Correlation matrices
    Square Hermitian positive-semidefinite arrays whose dimensions equal the
    number of candidate ports. Unit diagonal is expected for the channel-power
    normalization used in the manuscript.

Large-scale channel gain
    Strictly positive scalar representing average received channel power.

Rician factor
    Nonnegative scalar in linear scale, not decibels.

Random-number generator
    An explicit ``numpy.random.Generator`` instance. Requiring the generator as
    an input makes channel generation reproducible and prevents hidden global
    random-state dependencies.

Validation and numerical safeguards
-----------------------------------
The module validates array dimensions, nonempty inputs, finite coordinates,
positive gains, nonnegative Rician factors, compatible channel shapes, and
positive-semidefinite correlation matrices.

The correlation matrix is symmetrized before its eigendecomposition. Small
negative eigenvalues within the specified tolerance are interpreted as
floating-point roundoff and clipped to zero. Larger negative eigenvalues raise
an exception.

The phase precoder validates Bob's channel magnitude at every port because the
phase coefficient is undefined when a channel coefficient is zero.

Reproducibility
---------------
All deterministic geometry, correlation, signature, matrix, and precoding
functions return identical outputs for identical inputs.

The Rician channel generator is reproducible when called with the same inputs
and an equivalently initialized ``numpy.random.Generator``. Experiment runners
are responsible for defining and recording the random seed in their YAML
configuration files.

Dependencies
------------
numpy
    Array construction, validation, linear algebra, complex arithmetic, and
    random-sample manipulation.

scipy.special.j0
    Evaluation of the order-zero Bessel function used by the Clarke-Jakes
    correlation model.

Modeling assumptions and limitations
------------------------------------
- The implemented aperture geometry is one-dimensional and linear.
- Candidate-port coordinates are normalized by wavelength.
- The Clarke-Jakes model assumes isotropic two-dimensional diffuse scattering.
- Receiver direction affects the deterministic Rician component through a
  far-field plane-wave signature.
- The same correlation model is used for the diffuse component unless a custom
  matrix is supplied explicitly.
- Mutual coupling, impedance mismatch, radiation-pattern variation, switching
  transients, and hardware impairments are not modeled.
- Independent calls for Bob and Eve generate independent diffuse channel
  samples unless the experiment explicitly introduces shared randomness.
- Co-located Bob and Eve are therefore not forced to have identical diffuse
  channel realizations.
- The effective Eve-side channel assumes that Alice applies Bob-oriented
  phase-only compensation and that the relevant channels remain constant over
  the considered coherence interval.

Notes
-----
The module contains no figure-specific plotting, file output, or experiment
configuration logic. Such operations belong in the scripts and I/O modules.

When a custom correlation matrix is supplied, the caller is responsible for
ensuring that the matrix represents the intended physical model and uses the
same port ordering as ``positions_wavelengths``.

See Also
--------
fa_ddm.geometry
    General receiver-coordinate and directional-geometry utilities.

fa_ddm.vulnerability
    Gaussian-mixture likelihoods and posterior-vulnerability evaluation.

fa_ddm.quadrature
    Deterministic integration over Eve's complex observation plane.

fa_ddm.optimization
    Privacy-aware optimization of fluid-antenna port-selection probabilities.

fa_ddm.bounds
    Total-variation and permutation-based vulnerability bounds.
"""


import numpy as np
from scipy.special import j0


def linear_port_positions(number_of_ports, aperture_wavelengths):
    """Return uniformly spaced 1-D port positions normalized by wavelength."""
    if not isinstance(number_of_ports, (int, np.integer)):
        raise TypeError("number_of_ports must be an integer.")
    if number_of_ports <= 0:
        raise ValueError("number_of_ports must be positive.")
    if aperture_wavelengths < 0.0:
        raise ValueError("aperture_wavelengths must be nonnegative.")

    if number_of_ports == 1:
        return np.array([0.0], dtype=float)

    return np.linspace(
        0.0,
        float(aperture_wavelengths),
        number_of_ports,
        dtype=float,
    )


def clarke_jakes_correlation(positions_wavelengths):
    """Construct the Clarke--Jakes diffuse correlation matrix."""
    positions = np.asarray(positions_wavelengths, dtype=float)

    if positions.ndim != 1:
        raise ValueError("positions_wavelengths must be one-dimensional.")
    if positions.size == 0:
        raise ValueError("positions_wavelengths must not be empty.")
    if not np.all(np.isfinite(positions)):
        raise ValueError("positions_wavelengths must contain finite values.")

    pairwise_distance = np.abs(positions[:, None] - positions[None, :])
    correlation = j0(2.0 * np.pi * pairwise_distance)
    correlation = 0.5 * (correlation + correlation.T)
    np.fill_diagonal(correlation, 1.0)
    return correlation


def minimum_eigenvalue(matrix):
    """Return the smallest eigenvalue of a real symmetric matrix."""
    values = np.linalg.eigvalsh(np.asarray(matrix, dtype=float))
    return float(values[0])


def far_field_signature(positions_wavelengths, angle_deg):
    """Return the deterministic far-field signature of a linear aperture.

    The angle is measured from broadside. For normalized position x_n/lambda,
    the n-th entry is exp(-j*2*pi*x_n*sin(theta)).
    """
    positions = np.asarray(positions_wavelengths, dtype=float)
    if positions.ndim != 1 or positions.size == 0:
        raise ValueError("positions_wavelengths must be a nonempty 1-D array.")
    if not np.all(np.isfinite(positions)):
        raise ValueError("positions_wavelengths must contain finite values.")
    if not np.isfinite(angle_deg):
        raise ValueError("angle_deg must be finite.")

    angle_rad = np.deg2rad(float(angle_deg))
    phase = -2.0 * np.pi * positions * np.sin(angle_rad)
    return np.exp(1j * phase).astype(np.complex128)


def hermitian_square_root(matrix, tolerance=1e-12):
    """Return a stable Hermitian positive-semidefinite matrix square root."""
    matrix = np.asarray(matrix, dtype=np.complex128)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("matrix must be square.")

    hermitian = 0.5 * (matrix + matrix.conj().T)
    eigenvalues, eigenvectors = np.linalg.eigh(hermitian)
    if eigenvalues[0] < -tolerance:
        raise ValueError("matrix must be positive semidefinite.")

    clipped = np.clip(eigenvalues, 0.0, None)
    return (eigenvectors * np.sqrt(clipped)) @ eigenvectors.conj().T


def generate_correlated_rician_channel(
    positions_wavelengths,
    angle_deg,
    large_scale_gain,
    rician_factor,
    rng,
    correlation_matrix=None,
):
    """Generate one spatially correlated Rician channel realization.

    Parameters use the paper normalization E[|h_n|^2] = large_scale_gain.
    The supplied ``rng`` must be a NumPy random Generator.
    """
    positions = np.asarray(positions_wavelengths, dtype=float)
    if large_scale_gain <= 0.0:
        raise ValueError("large_scale_gain must be positive.")
    if rician_factor < 0.0:
        raise ValueError("rician_factor must be nonnegative.")
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be an instance of numpy.random.Generator.")

    if correlation_matrix is None:
        correlation = clarke_jakes_correlation(positions)
    else:
        correlation = np.asarray(correlation_matrix, dtype=np.complex128)

    number_of_ports = positions.size
    if correlation.shape != (number_of_ports, number_of_ports):
        raise ValueError("correlation_matrix has an incompatible shape.")

    signature = far_field_signature(positions, angle_deg)
    root = hermitian_square_root(correlation)
    white = (
        rng.standard_normal(number_of_ports)
        + 1j * rng.standard_normal(number_of_ports)
    ) / np.sqrt(2.0)
    diffuse = root @ white

    los_scale = np.sqrt(rician_factor / (rician_factor + 1.0))
    diffuse_scale = np.sqrt(1.0 / (rician_factor + 1.0))
    channel = np.sqrt(large_scale_gain) * (
        los_scale * signature + diffuse_scale * diffuse
    )
    return channel.astype(np.complex128)


def phase_precoder(bob_channel, zero_tolerance=1e-15):
    """Return q_n = conj(h_B,n)/|h_B,n| for every candidate port."""
    bob_channel = np.asarray(bob_channel, dtype=np.complex128)
    if bob_channel.ndim != 1 or bob_channel.size == 0:
        raise ValueError("bob_channel must be a nonempty 1-D array.")
    magnitudes = np.abs(bob_channel)
    if np.any(magnitudes <= zero_tolerance):
        raise ValueError("bob_channel contains a zero-magnitude coefficient.")
    return np.conj(bob_channel) / magnitudes


def effective_eve_channel(bob_channel, eve_channel):
    """Return g_n = h_E,n * conj(h_B,n)/|h_B,n|."""
    bob_channel = np.asarray(bob_channel, dtype=np.complex128)
    eve_channel = np.asarray(eve_channel, dtype=np.complex128)
    if bob_channel.shape != eve_channel.shape:
        raise ValueError("bob_channel and eve_channel must have equal shapes.")
    return eve_channel * phase_precoder(bob_channel)

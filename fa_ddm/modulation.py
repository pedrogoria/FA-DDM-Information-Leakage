"""
Digital modulation constellations and symbol-probability utilities for the
FA-DDM numerical framework.

Author
------
Pedro E. Goria Silva

Module
------
fa_ddm.modulation

Purpose
-------
This module implements the digital modulation alphabets used by the
fluid-antenna-assisted dynamic directional modulation (FA-DDM) simulations,
analytical validations, and privacy-aware port-selection experiments.

The module provides reusable functions for:

1. normalizing arbitrary complex constellations to unit average symbol energy;
2. constructing binary phase-shift keying (BPSK);
3. constructing quadrature phase-shift keying (QPSK);
4. constructing general M-ary phase-shift keying (M-PSK);
5. constructing square M-ary quadrature-amplitude modulation (M-QAM);
6. selecting a supported constellation through a case-insensitive name; and
7. constructing uniform symbol-prior probability vectors.

The functions are intentionally independent of channel generation, port
selection, likelihood evaluation, plotting, and file output. This separation
allows all experiments to use a common and consistently normalized modulation
model.

Constellation normalization
---------------------------
All supported constellations follow the unit-average-energy convention

    E[|S|^2] = (1 / M) sum_m |s_m|^2 = 1,

where M is the constellation order and s_m is the m-th complex symbol.

The private helper ``_normalize_average_energy`` computes the empirical average
symbol energy and scales the complete constellation by its square root. A
constellation with zero or negative numerical average energy is rejected.

Unit-energy normalization ensures that the transmit-power parameter used by the
channel, vulnerability, quadrature, and optimization modules has the same
interpretation across modulation formats. Comparisons among BPSK, PSK, and QAM
therefore do not unintentionally include different average constellation
energies.

BPSK constellation
------------------
The BPSK alphabet is

    S_BPSK = {+1, -1}.

Both symbols have unit magnitude, so the alphabet already has unit average
energy and does not require additional normalization.

QPSK constellation
------------------
The QPSK alphabet is initially constructed from the four points

    {1 + j, -1 + j, -1 - j, 1 - j}

and is then normalized to unit average energy. The resulting points have
magnitude one and are located at phases separated by pi/2.

The explicit ordering is retained by the implementation and may affect the
meaning of a reference-symbol index used by distinguishability or
permutation-bound experiments. Experiments should therefore treat symbol
indices as implementation-defined rather than assuming a different Gray,
clockwise, or counterclockwise ordering.

M-PSK constellation
-------------------
For integer order M >= 2, the general PSK factory constructs

    s_m = exp(j 2 pi m / M),
    m = 0, ..., M - 1.

Every symbol has unit magnitude, so every M-PSK alphabet automatically has unit
average energy.

The first symbol is located at phase zero. No additional phase rotation or Gray
labeling is applied. The function represents the complex symbol geometry only;
it does not assign bit labels or implement encoding and decoding operations.

Square M-QAM constellation
--------------------------
For a square modulation order M = L^2 with integer L >= 2, the QAM routine
constructs equally spaced in-phase and quadrature levels

    -(L - 1), -(L - 3), ..., L - 3, L - 1.

The Cartesian product of these real and imaginary levels forms an L by L square
QAM grid. The flattened complex alphabet is then normalized to unit average
energy.

Examples include:

    M = 4       4-QAM;
    M = 16      16-QAM;
    M = 64      64-QAM.

The routine accepts any square integer of at least four, even when the name is
not explicitly listed in the repository documentation. Non-square orders and
orders below four are rejected.

As with PSK, the function defines constellation locations only. Bit mapping,
Gray labeling, pulse shaping, coding, and demodulator implementation are
outside the scope of this module.

Named constellation factory
---------------------------
The function ``get_constellation`` provides a common string-based interface.
Input names are converted to uppercase and stripped of spaces and hyphens.
Consequently, names such as

    "16QAM"
    "16-QAM"
    "16 QAM"

are interpreted identically.

The factory recognizes the dedicated names ``BPSK`` and ``QPSK`` and also
parses names ending in ``PSK`` or ``QAM`` when the preceding characters form an
integer order.

Typical documented names are:

    BPSK
    QPSK
    8PSK
    16PSK
    16QAM
    64QAM

Unsupported names raise a ``ValueError`` containing the original input.

Uniform symbol probabilities
----------------------------
The function ``uniform_symbol_probabilities`` returns

    p_S(s_m) = 1 / M,
    m = 1, ..., M,

for a finite alphabet containing M symbols.

Uniform priors are used by the principal vulnerability bounds and most
manuscript experiments. The returned vector is a one-dimensional floating-
point array whose entries sum to one up to normal floating-point precision.

Public functions
----------------
bpsk_constellation
    Returns the two-symbol unit-average-energy BPSK alphabet.

qpsk_constellation
    Returns the explicitly ordered and normalized four-symbol QPSK alphabet.

psk_constellation
    Returns an M-PSK alphabet with equally spaced phases and unit magnitude.

square_qam_constellation
    Returns a normalized square M-QAM alphabet for a valid square order.

get_constellation
    Parses a modulation name and returns the corresponding normalized complex
    alphabet.

uniform_symbol_probabilities
    Returns a uniform prior-probability vector for a finite symbol alphabet.

Private functions
-----------------
_normalize_average_energy
    Converts an input to a complex NumPy array and scales the symbols to unit
    empirical average energy.

Input conventions
-----------------
symbols
    Array-like collection of complex constellation points.

order
    Integer modulation order. PSK requires order >= 2. Square QAM requires a
    square integer of at least four.

name
    String-like modulation identifier. Matching is case-insensitive and ignores
    spaces and hyphens.

number_of_symbols
    Positive number of entries in the requested uniform probability vector.

Returned quantities
-------------------
Constellation functions return

    symbols
        One-dimensional ``numpy.complex128`` array of complex constellation
        points with unit average symbol energy.

``uniform_symbol_probabilities`` returns

    probabilities
        One-dimensional floating-point array with equal entries 1/M.

Validation and numerical safeguards
-----------------------------------
The normalization helper rejects constellations whose empirical average energy
is not strictly positive.

The PSK generator requires an integer order of at least two. The square-QAM
generator requires an integer order whose square root is an integer of at least
two.

The named factory delegates order validation to the corresponding PSK or QAM
constructor. A syntactically valid name such as ``3QAM`` is parsed but rejected
because the order is not a valid square QAM size.

The uniform-prior helper requires a positive symbol count. Callers should pass
an integer symbol count, normally obtained from ``symbols.size``.

Reproducibility
---------------
All functions in this module are deterministic and contain no random-number
generation. Constellation ordering, normalization, and probability vectors are
fully determined by the input arguments.

Because reference-symbol indices are used in some analytical bounds,
reproducible experiments should preserve the symbol ordering returned by these
functions rather than sorting or rotating the alphabet after construction.

Computational complexity
------------------------
Constellation construction and normalization require O(M) operations and O(M)
storage for an alphabet of size M. Square-QAM grid construction additionally
creates two intermediate L by L real grids, where L is the square root of M.

These costs are negligible relative to Gaussian-mixture quadrature, Monte Carlo
simulation, channel averaging, and linear-program optimization.

Modeling assumptions and limitations
------------------------------------
- Every supported constellation is normalized to unit average symbol energy.
- Symbol probabilities are uniform unless another module receives an explicit
  nonuniform prior.
- The module represents complex symbols only and does not define bit labels.
- Gray coding, differential encoding, channel coding, interleaving, pulse
  shaping, filtering, synchronization, and waveform generation are not
  implemented.
- The PSK factory applies no constellation rotation beyond the zero-phase first
  symbol.
- The QAM factory supports square constellations only.
- Cross-QAM, rectangular QAM, APSK, PAM, and custom irregular constellations are
  not provided by the named factory.
- The module does not calculate symbol-error probability, bit-error
  probability, minimum distance, or decision regions.

Notes
-----
The QPSK-specific function and the general ``psk_constellation(4)`` function
produce constellations with different phase rotations and potentially different
symbol ordering. Both have unit average energy and equivalent Euclidean
geometry, but experiments that depend on the explicit symbol index or on
channel-dependent centroid alignment should use one convention consistently.

Likewise, constellation rotation can affect the interaction between symbols and
one fixed channel realization even when the modulation has the same minimum
distance. Reproducible comparisons should therefore use the same factory and
ordering across all evaluated methods.

See Also
--------
fa_ddm.vulnerability
    Secret-conditioned Gaussian-mixture likelihoods and optimal symbol
    inference.

fa_ddm.quadrature
    Deterministic posterior-vulnerability integration.

fa_ddm.distinguishability
    Finite-SNR total-variation bounds for uniform symbol priors.

fa_ddm.bounds
    Total-variation and permutation-based vulnerability-bound validation.

fa_ddm.optimization
    Bob-side symbol-error models and privacy-aware port selection.

scripts.run_figure_01
    Multi-modulation quadrature and Monte Carlo validation.
"""


import numpy as np


def _normalize_average_energy(symbols):
    """Normalize a complex constellation to unit average energy."""
    symbols = np.asarray(symbols, dtype=np.complex128)
    average_energy = np.mean(np.abs(symbols) ** 2)
    if average_energy <= 0.0:
        raise ValueError("Constellation average energy must be positive.")
    return symbols / np.sqrt(average_energy)


def bpsk_constellation():
    """Return the unit-average-energy BPSK constellation."""
    return np.array([1.0, -1.0], dtype=np.complex128)


def qpsk_constellation():
    """Return a unit-average-energy QPSK constellation."""
    symbols = np.array(
        [1.0 + 1.0j, -1.0 + 1.0j, -1.0 - 1.0j, 1.0 - 1.0j],
        dtype=np.complex128,
    )
    return _normalize_average_energy(symbols)


def psk_constellation(order):
    """Return a unit-average-energy M-PSK constellation."""
    if not isinstance(order, (int, np.integer)):
        raise TypeError("PSK order must be an integer.")
    if order < 2:
        raise ValueError("PSK order must be at least two.")
    phases = 2.0 * np.pi * np.arange(order) / order
    return np.exp(1j * phases).astype(np.complex128)


def square_qam_constellation(order):
    """Return a unit-average-energy square M-QAM constellation."""
    if not isinstance(order, (int, np.integer)):
        raise TypeError("QAM order must be an integer.")
    side = int(np.sqrt(order))
    if side * side != order or side < 2:
        raise ValueError("QAM order must be a square integer of at least four.")
    levels = np.arange(-(side - 1), side, 2, dtype=float)
    real_grid, imag_grid = np.meshgrid(levels, levels, indexing="xy")
    symbols = (real_grid + 1j * imag_grid).ravel()
    return _normalize_average_energy(symbols)


def get_constellation(name):
    """Return a supported unit-average-energy constellation by name.

    Supported names are BPSK, QPSK, 8PSK, 16PSK, 16QAM, and 64QAM.
    Names are case-insensitive and may contain hyphens or spaces.
    """
    normalized = str(name).upper().replace("-", "").replace(" ", "")
    if normalized == "BPSK":
        return bpsk_constellation()
    if normalized == "QPSK":
        return qpsk_constellation()
    if normalized.endswith("PSK") and normalized[:-3].isdigit():
        return psk_constellation(int(normalized[:-3]))
    if normalized.endswith("QAM") and normalized[:-3].isdigit():
        return square_qam_constellation(int(normalized[:-3]))
    raise ValueError("Unsupported modulation: {0}".format(name))


def uniform_symbol_probabilities(number_of_symbols):
    """Return a uniform probability vector for a finite constellation."""
    if number_of_symbols <= 0:
        raise ValueError("number_of_symbols must be positive.")
    return np.full(number_of_symbols, 1.0 / number_of_symbols, dtype=float)

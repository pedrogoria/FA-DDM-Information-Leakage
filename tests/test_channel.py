"""Tests for fluid-antenna geometry and spatial correlation."""

import numpy as np
import pytest

from fa_ddm.channel import (
    clarke_jakes_correlation,
    linear_port_positions,
    minimum_eigenvalue,
)


def test_linear_port_positions_include_aperture_endpoints():
    positions = linear_port_positions(5, 2.0)
    expected = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
    np.testing.assert_allclose(positions, expected)


def test_single_port_is_located_at_zero():
    positions = linear_port_positions(1, 2.0)
    np.testing.assert_allclose(positions, np.array([0.0]))


def test_invalid_geometry_parameters_are_rejected():
    with pytest.raises(ValueError):
        linear_port_positions(0, 1.0)
    with pytest.raises(ValueError):
        linear_port_positions(4, -1.0)
    with pytest.raises(TypeError):
        linear_port_positions(4.5, 1.0)


def test_correlation_matrix_is_symmetric_with_unit_diagonal():
    positions = linear_port_positions(8, 1.0)
    correlation = clarke_jakes_correlation(positions)

    np.testing.assert_allclose(correlation, correlation.T, atol=1e-14)
    np.testing.assert_allclose(np.diag(correlation), np.ones(8))


def test_correlation_matrix_is_positive_semidefinite_numerically():
    positions = linear_port_positions(12, 2.0)
    correlation = clarke_jakes_correlation(positions)

    # A small negative tolerance is allowed for floating-point roundoff.
    assert minimum_eigenvalue(correlation) >= -1e-10


def test_closer_ports_have_larger_local_correlation():
    close_positions = np.array([0.0, 0.05])
    separated_positions = np.array([0.0, 0.20])

    close_correlation = clarke_jakes_correlation(close_positions)[0, 1]
    separated_correlation = clarke_jakes_correlation(separated_positions)[0, 1]

    # Both distances lie before the first zero of J_0, where correlation
    # decreases with distance.
    assert close_correlation > separated_correlation


def test_invalid_position_array_is_rejected():
    with pytest.raises(ValueError):
        clarke_jakes_correlation([])
    with pytest.raises(ValueError):
        clarke_jakes_correlation([[0.0, 0.1]])
    with pytest.raises(ValueError):
        clarke_jakes_correlation([0.0, np.nan])

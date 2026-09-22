"""Tests for deterministic posterior-vulnerability quadrature."""

import numpy as np
import pytest

from fa_ddm.modulation import qpsk_constellation, uniform_symbol_probabilities
from fa_ddm.quadrature import (
    integration_bounds,
    quadrature_vulnerability,
    rectangular_quadrature_grid,
    trapezoidal_axis,
)
from fa_ddm.vulnerability import monte_carlo_vulnerability


def test_trapezoidal_weights_integrate_constant_function():
    _, weights = trapezoidal_axis(-2.0, 3.0, 101)
    assert np.sum(weights) == pytest.approx(5.0)


def test_rectangular_weights_equal_rectangle_area():
    observations, weights = rectangular_quadrature_grid(
        (-2.0, 3.0, -1.0, 2.0), points_per_axis=51
    )
    assert observations.shape == weights.shape
    assert np.sum(weights) == pytest.approx(15.0)


def test_integration_bounds_contain_all_centroids():
    centroids = np.array([[1.0 + 2.0j, -3.0 - 1.0j]])
    bounds = integration_bounds(
        centroids, noise_variance=0.25, noise_margin_sigma=4.0
    )
    assert bounds == pytest.approx((-5.0, 3.0, -3.0, 4.0))


def test_zero_effective_channel_equals_prior_vulnerability():
    symbols = qpsk_constellation()
    value = quadrature_vulnerability(
        symbols=symbols,
        effective_eve_channel=np.zeros(3, dtype=complex),
        port_probabilities=np.full(3, 1.0 / 3.0),
        symbol_probabilities=uniform_symbol_probabilities(4),
        transmit_power=1.0,
        noise_variance=1.0,
        points_per_axis=121,
        noise_margin_sigma=6.0,
    )
    assert value == pytest.approx(0.25, abs=2e-5)


def test_high_snr_single_port_qpsk_is_nearly_perfect():
    symbols = qpsk_constellation()
    value = quadrature_vulnerability(
        symbols=symbols,
        effective_eve_channel=np.array([1.0 + 0.0j]),
        port_probabilities=np.array([1.0]),
        symbol_probabilities=uniform_symbol_probabilities(4),
        transmit_power=1.0,
        noise_variance=0.02,
        points_per_axis=151,
        noise_margin_sigma=6.0,
    )
    assert value > 0.999
    assert value <= 1.0001


def test_quadrature_agrees_with_monte_carlo():
    symbols = qpsk_constellation()
    effective_channel = np.array([1.0 + 0.0j, 0.35 + 0.75j])
    rho = np.array([0.4, 0.6])
    priors = uniform_symbol_probabilities(4)

    quadrature = quadrature_vulnerability(
        symbols=symbols,
        effective_eve_channel=effective_channel,
        port_probabilities=rho,
        symbol_probabilities=priors,
        transmit_power=1.0,
        noise_variance=0.5,
        points_per_axis=151,
        noise_margin_sigma=6.0,
    )
    monte_carlo, standard_error = monte_carlo_vulnerability(
        symbols=symbols,
        effective_eve_channel=effective_channel,
        port_probabilities=rho,
        symbol_probabilities=priors,
        transmit_power=1.0,
        noise_variance=0.5,
        number_of_trials=100000,
        rng=np.random.default_rng(20260921),
    )
    tolerance = max(5.0 * standard_error, 0.01)
    assert abs(quadrature - monte_carlo) < tolerance


def test_quadrature_is_between_prior_and_one():
    symbols = qpsk_constellation()
    value = quadrature_vulnerability(
        symbols=symbols,
        effective_eve_channel=np.array([1.0 + 0.0j, -0.5 + 0.3j]),
        port_probabilities=np.array([0.5, 0.5]),
        symbol_probabilities=uniform_symbol_probabilities(4),
        transmit_power=1.0,
        noise_variance=1.0,
        points_per_axis=101,
    )
    assert 0.25 <= value <= 1.0001

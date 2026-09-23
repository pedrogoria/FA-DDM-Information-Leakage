"""Tests for privacy-aware port-selection optimization."""

import numpy as np
import pytest

from fa_ddm.modulation import get_constellation, uniform_symbol_probabilities
from fa_ddm.optimization import (
    optimize_port_probabilities,
    qpsk_symbol_error_probabilities,
)


def test_qpsk_error_decreases_with_channel_gain():
    errors = qpsk_symbol_error_probabilities(
        np.array([0.2 + 0.0j, 1.0 + 0.0j, 2.0 + 0.0j]),
        transmit_power=1.0,
        noise_variance_bob=0.1,
    )
    assert errors[0] > errors[1] > errors[2]


def test_optimizer_returns_probability_vector_and_respects_reliability():
    symbols = get_constellation("QPSK")
    bob_errors = np.array([0.01, 0.03, 0.08])
    result = optimize_port_probabilities(
        symbols=symbols,
        effective_eve_channel=np.array([1.0, 0.2 + 0.8j, -0.6 + 0.1j]),
        symbol_probabilities=uniform_symbol_probabilities(4),
        transmit_power=1.0,
        noise_variance_eve=0.5,
        bob_port_error_probabilities=bob_errors,
        maximum_bob_error=0.04,
        points_per_axis=31,
        noise_margin_sigma=4.0,
    )
    assert np.sum(result["rho"]) == pytest.approx(1.0, abs=1e-8)
    assert np.all(result["rho"] >= -1e-10)
    assert result["bob_error"] <= 0.04 + 1e-8
    assert 0.25 <= result["vulnerability"] <= 1.0


def test_infeasible_reliability_is_rejected():
    with pytest.raises(ValueError):
        optimize_port_probabilities(
            symbols=get_constellation("QPSK"),
            effective_eve_channel=np.ones(2, dtype=complex),
            symbol_probabilities=uniform_symbol_probabilities(4),
            transmit_power=1.0,
            noise_variance_eve=1.0,
            bob_port_error_probabilities=np.array([0.1, 0.2]),
            maximum_bob_error=0.05,
            points_per_axis=11,
        )

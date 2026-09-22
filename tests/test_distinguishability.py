"""Tests for finite-SNR distinguishability bounds."""

import numpy as np
import pytest

from fa_ddm.distinguishability import finite_snr_metrics
from fa_ddm.modulation import get_constellation


def test_binary_tv_identity_matches_vulnerability():
    symbols = get_constellation("BPSK")
    result = finite_snr_metrics(
        symbols=symbols,
        effective_eve_channel=np.array([1.0 + 0.0j]),
        port_probabilities=np.array([1.0]),
        transmit_power=1.0,
        noise_variance=0.5,
        points_per_axis=151,
    )
    assert result["vulnerability"] == pytest.approx(
        result["binary_tv_identity"], abs=2e-4
    )


def test_mary_bounds_contain_exact_vulnerability():
    symbols = get_constellation("QPSK")
    result = finite_snr_metrics(
        symbols=symbols,
        effective_eve_channel=np.array([1.0 + 0.0j, 0.3 + 0.7j]),
        port_probabilities=np.array([0.5, 0.5]),
        transmit_power=1.0,
        noise_variance=0.7,
        points_per_axis=151,
    )
    assert result["lower_bound"] <= result["vulnerability"] + 3e-4
    assert result["vulnerability"] <= result["upper_bound"] + 3e-4


def test_zero_channel_gives_prior_and_zero_tv():
    symbols = get_constellation("QPSK")
    result = finite_snr_metrics(
        symbols=symbols,
        effective_eve_channel=np.zeros(2, dtype=complex),
        port_probabilities=np.array([0.5, 0.5]),
        transmit_power=1.0,
        noise_variance=1.0,
        points_per_axis=121,
    )
    assert result["vulnerability"] == pytest.approx(0.25, abs=2e-5)
    np.testing.assert_allclose(
        result["pairwise_total_variation"], np.zeros(4), atol=1e-10
    )

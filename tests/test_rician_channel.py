"""Tests for directional and spatially correlated Rician channels."""

import numpy as np
import pytest

from fa_ddm.channel import (
    clarke_jakes_correlation,
    effective_eve_channel,
    far_field_signature,
    generate_correlated_rician_channel,
    linear_port_positions,
    phase_precoder,
)


def test_broadside_signature_is_all_ones():
    positions = linear_port_positions(8, 1.0)
    signature = far_field_signature(positions, angle_deg=0.0)
    np.testing.assert_allclose(signature, np.ones(8, dtype=complex))


def test_far_field_signature_has_unit_modulus():
    positions = linear_port_positions(8, 1.0)
    signature = far_field_signature(positions, angle_deg=35.0)
    np.testing.assert_allclose(np.abs(signature), np.ones(8))


def test_channel_is_reproducible_with_fixed_seed():
    positions = linear_port_positions(6, 1.0)
    rng_1 = np.random.default_rng(1234)
    rng_2 = np.random.default_rng(1234)

    channel_1 = generate_correlated_rician_channel(
        positions, 20.0, 1.0, 5.0, rng_1
    )
    channel_2 = generate_correlated_rician_channel(
        positions, 20.0, 1.0, 5.0, rng_2
    )
    np.testing.assert_allclose(channel_1, channel_2)


def test_rayleigh_channel_has_nearly_zero_sample_mean():
    positions = linear_port_positions(3, 0.5)
    correlation = clarke_jakes_correlation(positions)
    rng = np.random.default_rng(2026)
    samples = np.array([
        generate_correlated_rician_channel(
            positions,
            angle_deg=40.0,
            large_scale_gain=1.0,
            rician_factor=0.0,
            rng=rng,
            correlation_matrix=correlation,
        )
        for _ in range(20000)
    ])
    assert np.max(np.abs(np.mean(samples, axis=0))) < 0.03


def test_average_port_power_matches_large_scale_gain():
    positions = linear_port_positions(3, 0.5)
    correlation = clarke_jakes_correlation(positions)
    rng = np.random.default_rng(2027)
    beta = 2.5
    samples = np.array([
        generate_correlated_rician_channel(
            positions,
            angle_deg=-25.0,
            large_scale_gain=beta,
            rician_factor=4.0,
            rng=rng,
            correlation_matrix=correlation,
        )
        for _ in range(25000)
    ])
    average_power = np.mean(np.abs(samples) ** 2, axis=0)
    np.testing.assert_allclose(average_power, beta, rtol=0.035, atol=0.035)


def test_phase_precoder_removes_bob_channel_phase():
    bob_channel = np.array([1.0 + 1.0j, -2.0 + 0.5j, 0.4 - 1.2j])
    precoder = phase_precoder(bob_channel)
    aligned = bob_channel * precoder
    np.testing.assert_allclose(aligned, np.abs(bob_channel), atol=1e-14)


def test_effective_eve_channel_matches_definition():
    bob_channel = np.array([1.0 + 1.0j, 0.5 - 0.25j])
    eve_channel = np.array([-0.5 + 0.1j, 2.0 + 1.0j])
    expected = eve_channel * np.conj(bob_channel) / np.abs(bob_channel)
    actual = effective_eve_channel(bob_channel, eve_channel)
    np.testing.assert_allclose(actual, expected)


def test_invalid_channel_parameters_are_rejected():
    positions = linear_port_positions(4, 1.0)
    rng = np.random.default_rng(1)
    with pytest.raises(ValueError):
        generate_correlated_rician_channel(positions, 0.0, 0.0, 1.0, rng)
    with pytest.raises(ValueError):
        generate_correlated_rician_channel(positions, 0.0, 1.0, -1.0, rng)
    with pytest.raises(ValueError):
        effective_eve_channel(np.ones(3), np.ones(4))

"""Tests for three-dimensional receiver and aperture geometry."""

import numpy as np
import pytest

from fa_ddm.geometry import (
    far_field_signature_3d,
    free_space_large_scale_gain,
    linear_aperture_coordinates,
    receiver_geometry,
)


def test_linear_aperture_is_centered():
    coordinates = linear_aperture_coordinates(5, 2.0, axis="x")
    np.testing.assert_allclose(coordinates[:, 0], [-1.0, -0.5, 0.0, 0.5, 1.0])
    np.testing.assert_allclose(coordinates[:, 1:], 0.0)


def test_receiver_geometry_returns_distance_and_unit_direction():
    distance, direction = receiver_geometry([3.0, 4.0, 0.0])
    assert distance == pytest.approx(5.0)
    np.testing.assert_allclose(direction, [0.6, 0.8, 0.0])


def test_broadside_direction_gives_equal_phase_for_x_aperture():
    coordinates = linear_aperture_coordinates(7, 1.0, axis="x")
    signature = far_field_signature_3d(coordinates, [0.0, 1.0, 0.0])
    np.testing.assert_allclose(signature, np.ones(7, dtype=complex), atol=1e-14)


def test_endfire_direction_generates_spatial_phase():
    coordinates = linear_aperture_coordinates(3, 1.0, axis="x")
    signature = far_field_signature_3d(coordinates, [1.0, 0.0, 0.0])
    expected = np.exp(-1j * 2.0 * np.pi * coordinates[:, 0])
    np.testing.assert_allclose(signature, expected)


def test_large_scale_gain_decreases_with_distance():
    near_gain = free_space_large_scale_gain(10.0)
    far_gain = free_space_large_scale_gain(20.0)
    assert near_gain > far_gain
    assert near_gain / far_gain == pytest.approx(4.0)


def test_invalid_receiver_at_origin_is_rejected():
    with pytest.raises(ValueError):
        receiver_geometry([0.0, 0.0, 0.0])

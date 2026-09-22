"""Tests for spatial vulnerability-map helpers."""

import numpy as np
import pytest

from fa_ddm.spatial_map import (
    min_entropy_leakage,
    planar_receiver_grid,
    position_descriptors,
)


def test_planar_grid_ordering():
    positions = planar_receiver_grid([-1.0, 0.0, 1.0], [2.0, 3.0], 0.5)
    expected = np.array([
        [-1.0, 2.0, 0.5],
        [0.0, 2.0, 0.5],
        [1.0, 2.0, 0.5],
        [-1.0, 3.0, 0.5],
        [0.0, 3.0, 0.5],
        [1.0, 3.0, 0.5],
    ])
    np.testing.assert_allclose(positions, expected)


def test_position_descriptors():
    distance, azimuth, elevation, direction = position_descriptors([1.0, 1.0, 1.0])
    assert distance == pytest.approx(np.sqrt(3.0))
    assert azimuth == pytest.approx(45.0)
    assert elevation == pytest.approx(np.rad2deg(np.arctan2(1.0, np.sqrt(2.0))))
    assert np.linalg.norm(direction) == pytest.approx(1.0)


def test_min_entropy_leakage_limits():
    values = min_entropy_leakage([0.25, 0.5, 1.0], constellation_order=4)
    np.testing.assert_allclose(values, [0.0, 1.0, 2.0])


def test_invalid_vulnerability_is_rejected():
    with pytest.raises(ValueError):
        min_entropy_leakage([0.20], constellation_order=4)

"""Three-dimensional geometry for Bob and Eve positions."""

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

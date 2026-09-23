"""Privacy-aware optimization of fluid-antenna port probabilities."""

import numpy as np
from scipy.optimize import linprog
from scipy.special import erfc

from fa_ddm.quadrature import integration_bounds, rectangular_quadrature_grid
from fa_ddm.vulnerability import mixture_centroids


def q_function(x):
    """Return the Gaussian Q-function."""
    return 0.5 * erfc(np.asarray(x, dtype=float) / np.sqrt(2.0))


def qpsk_symbol_error_probabilities(
    bob_channel, transmit_power, noise_variance_bob
):
    """Return coherent QPSK symbol-error probability for every port.

    For port n, gamma_n = P_t |h_B,n|^2 / sigma_B^2 and
    P_e,n = 2 Q(sqrt(gamma_n)) - Q^2(sqrt(gamma_n)).
    """
    bob_channel = np.asarray(bob_channel, dtype=np.complex128)
    if bob_channel.ndim != 1 or bob_channel.size == 0:
        raise ValueError("bob_channel must be a nonempty 1-D array.")
    if transmit_power <= 0.0 or noise_variance_bob <= 0.0:
        raise ValueError("power and noise variance must be positive.")
    gamma = transmit_power * np.abs(bob_channel) ** 2 / noise_variance_bob
    q_values = q_function(np.sqrt(gamma))
    return 2.0 * q_values - q_values ** 2


def likelihood_coefficients(
    observations,
    symbols,
    effective_eve_channel,
    symbol_probabilities,
    transmit_power,
    noise_variance_eve,
):
    """Return coefficients C[r,m,n] such that ell_m(y_r)=sum_n C[r,m,n]rho_n."""
    observations = np.asarray(observations, dtype=np.complex128)
    symbols = np.asarray(symbols, dtype=np.complex128)
    effective_channel = np.asarray(effective_eve_channel, dtype=np.complex128)
    priors = np.asarray(symbol_probabilities, dtype=float)
    if observations.ndim != 1:
        raise ValueError("observations must be one-dimensional.")
    if priors.shape != (symbols.size,) or not np.isclose(priors.sum(), 1.0):
        raise ValueError("symbol_probabilities are invalid.")
    if noise_variance_eve <= 0.0:
        raise ValueError("noise_variance_eve must be positive.")

    centroids = mixture_centroids(symbols, effective_channel, transmit_power)
    displacement = observations[:, None, None] - centroids[None, :, :]
    densities = np.exp(
        -(np.abs(displacement) ** 2) / noise_variance_eve
    ) / (np.pi * noise_variance_eve)
    return densities * priors[None, :, None]


def optimize_port_probabilities(
    symbols,
    effective_eve_channel,
    symbol_probabilities,
    transmit_power,
    noise_variance_eve,
    bob_port_error_probabilities,
    maximum_bob_error,
    points_per_axis=61,
    noise_margin_sigma=5.0,
):
    """Solve the quadrature epigraph LP for privacy-aware port selection.

    Returns a dictionary containing the optimal probability vector and the
    quadrature objective value.
    """
    symbols = np.asarray(symbols, dtype=np.complex128)
    effective_channel = np.asarray(effective_eve_channel, dtype=np.complex128)
    bob_errors = np.asarray(bob_port_error_probabilities, dtype=float)
    number_of_ports = effective_channel.size
    if bob_errors.shape != (number_of_ports,):
        raise ValueError("bob_port_error_probabilities has an invalid shape.")
    if maximum_bob_error < np.min(bob_errors) - 1e-12:
        raise ValueError("maximum_bob_error makes the problem infeasible.")

    centroids = mixture_centroids(symbols, effective_channel, transmit_power)
    bounds = integration_bounds(
        centroids, noise_variance_eve, noise_margin_sigma
    )
    observations, weights = rectangular_quadrature_grid(
        bounds, points_per_axis
    )
    coefficients = likelihood_coefficients(
        observations,
        symbols,
        effective_channel,
        symbol_probabilities,
        transmit_power,
        noise_variance_eve,
    )

    number_of_nodes = observations.size
    number_of_symbols = symbols.size
    number_of_variables = number_of_ports + number_of_nodes

    objective = np.zeros(number_of_variables, dtype=float)
    objective[number_of_ports:] = weights

    # For every (r,m): sum_n C[r,m,n] rho_n - t_r <= 0.
    rows = number_of_nodes * number_of_symbols
    a_ub = np.zeros((rows + 1, number_of_variables), dtype=float)
    b_ub = np.zeros(rows + 1, dtype=float)
    row = 0
    for node_index in range(number_of_nodes):
        for symbol_index in range(number_of_symbols):
            a_ub[row, :number_of_ports] = coefficients[
                node_index, symbol_index, :
            ]
            a_ub[row, number_of_ports + node_index] = -1.0
            row += 1

    # Bob reliability: sum_n rho_n P_e,n <= epsilon_B.
    a_ub[-1, :number_of_ports] = bob_errors
    b_ub[-1] = float(maximum_bob_error)

    a_eq = np.zeros((1, number_of_variables), dtype=float)
    a_eq[0, :number_of_ports] = 1.0
    b_eq = np.array([1.0])
    variable_bounds = [(0.0, 1.0)] * number_of_ports + [
        (0.0, None)
    ] * number_of_nodes

    result = linprog(
        objective,
        A_ub=a_ub,
        b_ub=b_ub,
        A_eq=a_eq,
        b_eq=b_eq,
        bounds=variable_bounds,
        method="highs",
    )
    if not result.success:
        raise RuntimeError("Port-selection LP failed: {0}".format(result.message))

    rho = result.x[:number_of_ports]
    return {
        "rho": rho,
        "vulnerability": float(result.fun),
        "bob_error": float(np.dot(rho, bob_errors)),
        "status": result.message,
    }

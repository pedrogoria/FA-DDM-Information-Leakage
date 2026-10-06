"""Figure 5: privacy-reliability tradeoff and optimized port probabilities."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from fa_ddm.channel import (
    clarke_jakes_correlation,
    effective_eve_channel,
    generate_correlated_rician_channel,
    linear_port_positions,
)
from fa_ddm.io import (
    copy_configuration,
    load_yaml,
    prepare_experiment_directory,
    save_dat,
)
from fa_ddm.modulation import get_constellation, uniform_symbol_probabilities
from fa_ddm.optimization import (
    optimize_port_probabilities,
    qpsk_symbol_error_probabilities,
)
from fa_ddm.quadrature import quadrature_vulnerability


def threshold_key(value):
    """Return a column-safe reliability-threshold identifier."""
    return str(float(value)).replace(".", "p")


def feasible_value(value, bob_error, maximum_bob_error, tolerance=1e-12):
    """Return a benchmark value when feasible and NaN otherwise."""
    if bob_error <= maximum_bob_error + tolerance:
        return float(value)
    return np.nan


def evaluate_vulnerability(
    symbols,
    effective_channel,
    rho,
    priors,
    system,
    quadrature,
):
    """Evaluate posterior vulnerability for a fixed port distribution."""
    return quadrature_vulnerability(
        symbols=symbols,
        effective_eve_channel=effective_channel,
        port_probabilities=rho,
        symbol_probabilities=priors,
        transmit_power=float(system["transmit_power"]),
        noise_variance=float(system["noise_variance_eve"]),
        points_per_axis=int(quadrature["points_per_axis"]),
        noise_margin_sigma=float(quadrature["noise_margin_sigma"]),
    )


def run(config_path="configs/figures/figure_05_privacy_reliability.yaml"):
    """Run the privacy-reliability tradeoff experiment."""
    config_path = Path(config_path)
    config = load_yaml(config_path)
    system = config["system"]
    tradeoff = config["tradeoff"]
    quadrature = config["quadrature"]
    output = config["output"]
    seed = int(config["random"]["seed"])

    symbols = get_constellation(system["modulation"])
    priors = uniform_symbol_probabilities(symbols.size)
    number_of_ports = int(system["number_of_ports"])

    positions = linear_port_positions(
        number_of_ports,
        float(system["aperture_wavelengths"]),
    )
    correlation = clarke_jakes_correlation(positions)

    rng = np.random.default_rng(seed)
    bob_channel = generate_correlated_rician_channel(
        positions,
        float(system["bob_angle_deg"]),
        float(system["large_scale_gain_bob"]),
        float(system["rician_factor_bob"]),
        rng,
        correlation,
    )
    eve_channel = generate_correlated_rician_channel(
        positions,
        float(system["eve_angle_deg"]),
        float(system["large_scale_gain_eve"]),
        float(system["rician_factor_eve"]),
        rng,
        correlation,
    )
    effective_channel = effective_eve_channel(bob_channel, eve_channel)

    bob_errors = qpsk_symbol_error_probabilities(
        bob_channel,
        float(system["transmit_power"]),
        float(system["noise_variance_bob"]),
    )

    thresholds = np.asarray(
        tradeoff["reliability_thresholds"],
        dtype=float,
    )
    selected_thresholds = [
        float(value)
        for value in tradeoff["selected_thresholds_for_probabilities"]
    ]

    optimized_vulnerabilities = []
    optimized_bob_errors = []
    solutions = {}

    for threshold in thresholds:
        result = optimize_port_probabilities(
            symbols=symbols,
            effective_eve_channel=effective_channel,
            symbol_probabilities=priors,
            transmit_power=float(system["transmit_power"]),
            noise_variance_eve=float(system["noise_variance_eve"]),
            bob_port_error_probabilities=bob_errors,
            maximum_bob_error=float(threshold),
            points_per_axis=int(quadrature["points_per_axis"]),
            noise_margin_sigma=float(quadrature["noise_margin_sigma"]),
        )
        optimized_vulnerabilities.append(result["vulnerability"])
        optimized_bob_errors.append(result["bob_error"])
        solutions[float(threshold)] = result["rho"]

        print(
            "epsilon_B={0:.4f} | V_E={1:.6f} | Bob error={2:.6f}".format(
                threshold,
                result["vulnerability"],
                result["bob_error"],
            )
        )

    optimized_vulnerabilities = np.asarray(
        optimized_vulnerabilities,
        dtype=float,
    )
    optimized_bob_errors = np.asarray(optimized_bob_errors, dtype=float)

    uniform_rho = np.full(number_of_ports, 1.0 / number_of_ports)
    uniform_vulnerability = evaluate_vulnerability(
        symbols,
        effective_channel,
        uniform_rho,
        priors,
        system,
        quadrature,
    )
    uniform_bob_error = float(np.dot(uniform_rho, bob_errors))

    reliability_weights = np.abs(bob_channel) ** 2
    reliability_weighted_rho = reliability_weights / np.sum(reliability_weights)
    reliability_weighted_vulnerability = evaluate_vulnerability(
        symbols,
        effective_channel,
        reliability_weighted_rho,
        priors,
        system,
        quadrature,
    )
    reliability_weighted_bob_error = float(
        np.dot(reliability_weighted_rho, bob_errors)
    )

    best_bob_port = int(np.argmax(np.abs(bob_channel) ** 2))
    best_bob_rho = np.zeros(number_of_ports)
    best_bob_rho[best_bob_port] = 1.0
    best_bob_vulnerability = evaluate_vulnerability(
        symbols,
        effective_channel,
        best_bob_rho,
        priors,
        system,
        quadrature,
    )
    best_bob_error = float(bob_errors[best_bob_port])

    fixed_port_vulnerabilities = np.empty(number_of_ports)
    for port_index in range(number_of_ports):
        fixed_port_rho = np.zeros(number_of_ports)
        fixed_port_rho[port_index] = 1.0
        fixed_port_vulnerabilities[port_index] = evaluate_vulnerability(
            symbols,
            effective_channel,
            fixed_port_rho,
            priors,
            system,
            quadrature,
        )

    best_privacy_port = int(np.argmin(fixed_port_vulnerabilities))
    best_privacy_vulnerability = float(
        fixed_port_vulnerabilities[best_privacy_port]
    )
    best_privacy_error = float(bob_errors[best_privacy_port])

    uniform_feasible = np.asarray(
        [
            feasible_value(
                uniform_vulnerability,
                uniform_bob_error,
                threshold,
            )
            for threshold in thresholds
        ]
    )
    reliability_weighted_feasible = np.asarray(
        [
            feasible_value(
                reliability_weighted_vulnerability,
                reliability_weighted_bob_error,
                threshold,
            )
            for threshold in thresholds
        ]
    )
    best_bob_feasible = np.asarray(
        [
            feasible_value(
                best_bob_vulnerability,
                best_bob_error,
                threshold,
            )
            for threshold in thresholds
        ]
    )
    best_privacy_feasible = np.asarray(
        [
            feasible_value(
                best_privacy_vulnerability,
                best_privacy_error,
                threshold,
            )
            for threshold in thresholds
        ]
    )

    run_directory = prepare_experiment_directory(
        output["results_root"],
        output["experiment_name"],
        repository_root=".",
    )
    tradeoff_path = run_directory / output["dat_file"]
    probability_path = run_directory / output["probabilities_dat_file"]
    pdf_path = run_directory / output["pdf_file"]

    save_dat(
        {
            "reliability_threshold": thresholds,
            "optimized_vulnerability": optimized_vulnerabilities,
            "optimized_bob_error": optimized_bob_errors,
            "uniform_vulnerability_feasible": uniform_feasible,
            "uniform_vulnerability_all": np.full(
                thresholds.size,
                uniform_vulnerability,
            ),
            "uniform_bob_error": np.full(
                thresholds.size,
                uniform_bob_error,
            ),
            "reliability_weighted_vulnerability_feasible": (
                reliability_weighted_feasible
            ),
            "reliability_weighted_vulnerability_all": np.full(
                thresholds.size,
                reliability_weighted_vulnerability,
            ),
            "reliability_weighted_bob_error": np.full(
                thresholds.size,
                reliability_weighted_bob_error,
            ),
            "best_bob_vulnerability_feasible": best_bob_feasible,
            "best_bob_vulnerability_all": np.full(
                thresholds.size,
                best_bob_vulnerability,
            ),
            "best_bob_error": np.full(
                thresholds.size,
                best_bob_error,
            ),
            "best_privacy_vulnerability_feasible": best_privacy_feasible,
            "best_privacy_vulnerability_all": np.full(
                thresholds.size,
                best_privacy_vulnerability,
            ),
            "best_privacy_error": np.full(
                thresholds.size,
                best_privacy_error,
            ),
        },
        tradeoff_path,
        comments=[
            "Privacy-reliability tradeoff for one channel realization.",
            "Feasible benchmark columns contain NaN below their Bob-error thresholds.",
            "Columns ending in _all retain unconstrained benchmark values.",
            "Best-Bob and best-privacy port indices are one-based: {0} and {1}.".format(
                best_bob_port + 1,
                best_privacy_port + 1,
            ),
        ],
    )

    probability_data = {
        "port_index": np.arange(1, number_of_ports + 1),
        "port_position_lambda": positions,
        "bob_port_error": bob_errors,
        "fixed_port_vulnerability": fixed_port_vulnerabilities,
        "rho_uniform": uniform_rho,
        "rho_reliability_weighted": reliability_weighted_rho,
    }
    for threshold in selected_thresholds:
        probability_data[
            "rho_epsilon_{0}".format(threshold_key(threshold))
        ] = solutions[threshold]

    save_dat(
        probability_data,
        probability_path,
        comments=[
            "Optimized and benchmark port probabilities.",
            "Best-Bob port index: {0}.".format(best_bob_port + 1),
            "Best-privacy port index: {0}.".format(best_privacy_port + 1),
        ],
    )
    copy_configuration(
        config_path,
        run_directory / output["config_copy"],
    )

    figure, axes = plt.subplots(1, 2, figsize=(10.0, 3.7))

    axes[0].plot(
        thresholds,
        optimized_vulnerabilities,
        marker="o",
        color="tab:blue",
        label="Optimized",
    )
    axes[0].plot(
        thresholds,
        uniform_feasible,
        linestyle="--",
        color="black",
        label="Uniform",
    )
    axes[0].plot(
        thresholds,
        reliability_weighted_feasible,
        linestyle="-.",
        marker="s",
        color="tab:green",
        label="Reliability-weighted",
    )
    axes[0].plot(
        thresholds,
        best_bob_feasible,
        linestyle=":",
        marker="^",
        color="tab:red",
        label="Best-Bob fixed port",
    )
    axes[0].plot(
        thresholds,
        best_privacy_feasible,
        linestyle=(0, (5, 2)),
        marker="D",
        color="tab:purple",
        label="Best-privacy fixed port",
    )
    axes[0].set_xlabel("Bob reliability threshold")
    axes[0].set_ylabel("Posterior vulnerability")
    axes[0].grid(True)
    axes[0].legend()

    styles = ["o-", "s--", "^-." ]
    for style, threshold in zip(styles, selected_thresholds):
        axes[1].plot(
            np.arange(1, number_of_ports + 1),
            solutions[threshold],
            style,
            label="epsilon={0}".format(threshold),
        )
    axes[1].set_xlabel("Port index")
    axes[1].set_ylabel("Selection probability")
    axes[1].grid(True)
    axes[1].legend()

    figure.tight_layout()
    figure.savefig(pdf_path, bbox_inches="tight")
    plt.close(figure)

    print("Best-Bob port: {0}".format(best_bob_port + 1))
    print("Best-privacy port: {0}".format(best_privacy_port + 1))
    print("Uniform Bob error: {0:.6f}".format(uniform_bob_error))
    print(
        "Reliability-weighted Bob error: {0:.6f}".format(
            reliability_weighted_bob_error
        )
    )
    print("Best-Bob error: {0:.6f}".format(best_bob_error))
    print("Best-privacy error: {0:.6f}".format(best_privacy_error))
    print("Tradeoff data saved to: {0}".format(tradeoff_path.resolve()))
    print("Probability data saved to: {0}".format(probability_path.resolve()))
    print("Preview saved to: {0}".format(pdf_path.resolve()))

    return {
        "vulnerability": optimized_vulnerabilities,
        "bob_error": optimized_bob_errors,
        "solutions": solutions,
        "uniform_vulnerability": uniform_vulnerability,
        "reliability_weighted_vulnerability": reliability_weighted_vulnerability,
        "best_bob_vulnerability": best_bob_vulnerability,
        "best_privacy_vulnerability": best_privacy_vulnerability,
        "best_bob_port": best_bob_port,
        "best_privacy_port": best_privacy_port,
    }


if __name__ == "__main__":
    run()

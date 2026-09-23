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

    vulnerabilities = []
    achieved_errors = []
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
        vulnerabilities.append(result["vulnerability"])
        achieved_errors.append(result["bob_error"])
        solutions[float(threshold)] = result["rho"]

        print(
            "epsilon_B={0:.4f} | V_E={1:.6f} | Bob error={2:.6f}".format(
                threshold,
                result["vulnerability"],
                result["bob_error"],
            )
        )

    uniform_rho = np.full(number_of_ports, 1.0 / number_of_ports)
    uniform_vulnerability = quadrature_vulnerability(
        symbols=symbols,
        effective_eve_channel=effective_channel,
        port_probabilities=uniform_rho,
        symbol_probabilities=priors,
        transmit_power=float(system["transmit_power"]),
        noise_variance=float(system["noise_variance_eve"]),
        points_per_axis=int(quadrature["points_per_axis"]),
        noise_margin_sigma=float(quadrature["noise_margin_sigma"]),
    )
    uniform_bob_error = float(np.dot(uniform_rho, bob_errors))

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
            "optimized_vulnerability": vulnerabilities,
            "achieved_bob_error": achieved_errors,
            "uniform_vulnerability": np.full(
                thresholds.size,
                uniform_vulnerability,
            ),
            "uniform_bob_error": np.full(
                thresholds.size,
                uniform_bob_error,
            ),
        },
        tradeoff_path,
        comments=[
            "Optimized privacy-reliability tradeoff for one channel realization.",
            "Uniform benchmark columns are repeated for PGFPlots convenience.",
        ],
    )

    probability_data = {
        "port_index": np.arange(1, number_of_ports + 1),
        "port_position_lambda": positions,
        "bob_port_error": bob_errors,
        "rho_uniform": uniform_rho,
    }
    for threshold in selected_thresholds:
        probability_data[
            "rho_epsilon_{0}".format(threshold_key(threshold))
        ] = solutions[threshold]

    save_dat(
        probability_data,
        probability_path,
        comments=[
            "Optimized port probabilities for selected reliability thresholds."
        ],
    )
    copy_configuration(
        config_path,
        run_directory / output["config_copy"],
    )

    figure, axes = plt.subplots(1, 2, figsize=(10.0, 3.7))

    axes[0].plot(
        thresholds,
        vulnerabilities,
        marker="o",
        color="tab:blue",
        label="Optimized",
    )
    axes[0].axhline(
        uniform_vulnerability,
        color="black",
        linestyle="--",
        label="Uniform",
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

    print("Tradeoff data saved to: {0}".format(tradeoff_path.resolve()))
    print("Probability data saved to: {0}".format(probability_path.resolve()))
    print("Preview saved to: {0}".format(pdf_path.resolve()))

    return {
        "vulnerability": np.asarray(vulnerabilities),
        "bob_error": np.asarray(achieved_errors),
        "solutions": solutions,
    }


if __name__ == "__main__":
    run()

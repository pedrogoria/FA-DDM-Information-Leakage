"""Figure 1: validate quadrature vulnerability against Monte Carlo."""

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
from fa_ddm.quadrature import quadrature_vulnerability
from fa_ddm.vulnerability import monte_carlo_vulnerability


def run(config_path="configs/figures/figure_01_validation.yaml"):
    """Run Figure 1 and return the exported numerical data."""
    config_path = Path(config_path)
    config = load_yaml(config_path)
    system = config["system"]
    quadrature = config["quadrature"]
    monte_carlo = config["monte_carlo"]
    output = config["output"]

    seed = int(config["random"]["seed"])
    channel_rng = np.random.default_rng(seed)

    symbols = get_constellation(system["modulation"])
    symbol_probabilities = uniform_symbol_probabilities(symbols.size)
    number_of_ports = int(system["number_of_ports"])
    positions = linear_port_positions(
        number_of_ports, float(system["aperture_wavelengths"])
    )
    correlation = clarke_jakes_correlation(positions)

    bob_channel = generate_correlated_rician_channel(
        positions,
        float(system["bob_angle_deg"]),
        float(system["large_scale_gain_bob"]),
        float(system["rician_factor_bob"]),
        channel_rng,
        correlation,
    )
    eve_channel = generate_correlated_rician_channel(
        positions,
        float(system["eve_angle_deg"]),
        float(system["large_scale_gain_eve"]),
        float(system["rician_factor_eve"]),
        channel_rng,
        correlation,
    )
    effective_channel = effective_eve_channel(bob_channel, eve_channel)

    if system["port_selection"].lower() != "uniform":
        raise ValueError("Figure 1 currently supports uniform port selection only.")
    rho = np.full(number_of_ports, 1.0 / number_of_ports)

    transmit_power = float(system["transmit_power"])
    beta_eve = float(system["large_scale_gain_eve"])
    snr_values_db = np.asarray(config["sweep"]["eve_snr_db"], dtype=float)

    vulnerability_quad = []
    vulnerability_mc = []
    standard_error_mc = []

    for index, snr_db in enumerate(snr_values_db):
        snr_linear = 10.0 ** (snr_db / 10.0)
        noise_variance = transmit_power * beta_eve / snr_linear

        value_quad = quadrature_vulnerability(
            symbols=symbols,
            effective_eve_channel=effective_channel,
            port_probabilities=rho,
            symbol_probabilities=symbol_probabilities,
            transmit_power=transmit_power,
            noise_variance=noise_variance,
            points_per_axis=int(quadrature["points_per_axis"]),
            noise_margin_sigma=float(quadrature["noise_margin_sigma"]),
            batch_size=int(quadrature["batch_size"]),
        )
        value_mc, standard_error = monte_carlo_vulnerability(
            symbols=symbols,
            effective_eve_channel=effective_channel,
            port_probabilities=rho,
            symbol_probabilities=symbol_probabilities,
            transmit_power=transmit_power,
            noise_variance=noise_variance,
            number_of_trials=int(monte_carlo["number_of_trials"]),
            rng=np.random.default_rng(seed + 1000 + index),
        )
        vulnerability_quad.append(value_quad)
        vulnerability_mc.append(value_mc)
        standard_error_mc.append(standard_error)
        print(
            "SNR={0:6.1f} dB | quadrature={1:.6f} | MC={2:.6f}".format(
                snr_db, value_quad, value_mc
            )
        )

    vulnerability_quad = np.asarray(vulnerability_quad)
    vulnerability_mc = np.asarray(vulnerability_mc)
    standard_error_mc = np.asarray(standard_error_mc)
    confidence_95 = 1.96 * standard_error_mc
    prior_vulnerability = 1.0 / symbols.size

    data = {
        "eve_snr_db": snr_values_db,
        "vulnerability_quadrature": vulnerability_quad,
        "vulnerability_monte_carlo": vulnerability_mc,
        "mc_ci95_lower": np.maximum(
            prior_vulnerability, vulnerability_mc - confidence_95
        ),
        "mc_ci95_upper": np.minimum(1.0, vulnerability_mc + confidence_95),
        "prior_vulnerability": np.full(snr_values_db.size, prior_vulnerability),
    }

    run_directory = prepare_experiment_directory(
        output["results_root"],
        output["experiment_name"],
        repository_root=".",
    )
    dat_path = run_directory / output["dat_file"]
    config_copy_path = run_directory / output["config_copy"]
    pdf_path = run_directory / output["pdf_file"]

    comments = [
        "Experiment: {0}".format(output["experiment_name"]),
        "Modulation: {0}".format(system["modulation"]),
        "Columns use descriptive names and are intended for PGFPlots.",
        "Monte Carlo confidence bounds are exported for validation only.",
    ]
    save_dat(data, dat_path, comments=comments)
    copy_configuration(config_path, config_copy_path)

    figure, axis = plt.subplots(figsize=(5.3, 3.5))
    axis.plot(snr_values_db, vulnerability_quad, marker="o", label="Quadrature")
    axis.plot(
        snr_values_db,
        vulnerability_mc,
        marker="s",
        linestyle="--",
        label="Monte Carlo",
    )
    axis.axhline(prior_vulnerability, linestyle=":", color="black", label="Prior")
    axis.set_xlabel("Eve-side SNR (dB)")
    axis.set_ylabel("Posterior vulnerability")
    axis.set_ylim(max(0.0, prior_vulnerability - 0.05), 1.02)
    axis.grid(True)
    axis.legend()
    figure.tight_layout()
    figure.savefig(pdf_path, bbox_inches="tight")
    plt.close(figure)

    print("Data saved to: {0}".format(dat_path.resolve()))
    print("Preview saved to: {0}".format(pdf_path.resolve()))
    return data


if __name__ == "__main__":
    run()

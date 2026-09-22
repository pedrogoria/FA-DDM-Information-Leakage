"""Figure 1: quadrature and Monte Carlo validation for several modulations."""

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


def modulation_key(name):
    """Return a lowercase column-safe modulation identifier."""
    return str(name).lower().replace("-", "").replace(" ", "")


def normal_quantile_95(confidence_level):
    """Return the normal multiplier used for the supported 95% interval."""
    if not np.isclose(float(confidence_level), 0.95):
        raise ValueError("Figure 1 currently supports confidence_level=0.95 only.")
    return 1.96


def run(config_path="configs/figures/figure_01_validation.yaml"):
    """Run the multi-modulation validation experiment."""
    config_path = Path(config_path)
    config = load_yaml(config_path)
    system = config["system"]
    quadrature = config["quadrature"]
    monte_carlo = config["monte_carlo"]
    output = config["output"]

    seed = int(config["random"]["seed"])
    channel_rng = np.random.default_rng(seed)
    number_of_ports = int(system["number_of_ports"])
    positions = linear_port_positions(
        number_of_ports, float(system["aperture_wavelengths"])
    )
    correlation = clarke_jakes_correlation(positions)

    # The same channel realization is used for every modulation and SNR point.
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
    modulations = list(system["modulations"])
    confidence_multiplier = normal_quantile_95(
        monte_carlo["confidence_level"]
    )

    data = {"eve_snr_db": snr_values_db}
    preview_results = {}

    for modulation_index, modulation_name in enumerate(modulations):
        symbols = get_constellation(modulation_name)
        priors = uniform_symbol_probabilities(symbols.size)
        key = modulation_key(modulation_name)
        prior = 1.0 / symbols.size

        values_quad = []
        values_mc = []
        errors_mc = []

        print("\nModulation: {0}".format(modulation_name))
        for snr_index, snr_db in enumerate(snr_values_db):
            snr_linear = 10.0 ** (snr_db / 10.0)
            noise_variance = transmit_power * beta_eve / snr_linear

            value_quad = quadrature_vulnerability(
                symbols=symbols,
                effective_eve_channel=effective_channel,
                port_probabilities=rho,
                symbol_probabilities=priors,
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
                symbol_probabilities=priors,
                transmit_power=transmit_power,
                noise_variance=noise_variance,
                number_of_trials=int(monte_carlo["number_of_trials"]),
                rng=np.random.default_rng(
                    seed + 10000 * (modulation_index + 1) + snr_index
                ),
            )
            values_quad.append(value_quad)
            values_mc.append(value_mc)
            errors_mc.append(confidence_multiplier * standard_error)
            print(
                "SNR={0:6.1f} dB | quadrature={1:.6f} | MC={2:.6f}".format(
                    snr_db, value_quad, value_mc
                )
            )

        values_quad = np.asarray(values_quad)
        values_mc = np.asarray(values_mc)
        errors_mc = np.asarray(errors_mc)

        data["vulnerability_{0}_theory".format(key)] = values_quad
        data["vulnerability_{0}_mc".format(key)] = values_mc
        data["vulnerability_{0}_ci95_lower".format(key)] = np.maximum(
            prior, values_mc - errors_mc
        )
        data["vulnerability_{0}_ci95_upper".format(key)] = np.minimum(
            1.0, values_mc + errors_mc
        )
        data["prior_{0}".format(key)] = np.full(snr_values_db.size, prior)
        preview_results[key] = (modulation_name, values_quad, values_mc, errors_mc)

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
        "Modulations: {0}".format(", ".join(modulations)),
        "Theory columns contain deterministic quadrature results.",
        "MC columns contain Monte Carlo estimates; CI columns are for validation.",
        "All modulations use the same Bob and Eve channel realization.",
    ]
    save_dat(data, dat_path, comments=comments)
    copy_configuration(config_path, config_copy_path)

    colors = ["tab:blue", "tab:red", "tab:green", "black"]
    line_styles = ["-", "--", "-.", ":"]
    markers = ["o", "s", "^", "D"]

    figure, axis = plt.subplots(figsize=(5.5, 3.7))
    for curve_index, key in enumerate(preview_results):
        name, theory, monte_carlo_values, confidence = preview_results[key]
        color = colors[curve_index % len(colors)]
        axis.plot(
            snr_values_db,
            theory,
            color=color,
            linestyle=line_styles[curve_index % len(line_styles)],
            linewidth=1.1,
            label=name,
        )
        axis.errorbar(
            snr_values_db,
            monte_carlo_values,
            yerr=confidence,
            color=color,
            marker=markers[curve_index % len(markers)],
            linestyle="none",
            markersize=4.0,
            capsize=2.0,
            label="_nolegend_",
        )

    axis.set_xlabel("Eve-side SNR (dB)")
    axis.set_ylabel("Posterior vulnerability")
    axis.set_ylim(0.0, 1.02)
    axis.grid(True)
    axis.legend()
    figure.tight_layout()
    figure.savefig(pdf_path, bbox_inches="tight")
    plt.close(figure)

    print("\nData saved to: {0}".format(dat_path.resolve()))
    print("Preview saved to: {0}".format(pdf_path.resolve()))
    return data


if __name__ == "__main__":
    run()

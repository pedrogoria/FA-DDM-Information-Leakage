"""Finite-SNR and permutation-based vulnerability-bound validation."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from fa_ddm.bounds import (
    pairwise_total_variation_bounds,
    permutation_vulnerability_bound,
)
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


def run(config_path="configs/figures/figure_07_bounds.yaml"):
    """Run the finite-SNR vulnerability-bound experiment."""
    config_path = Path(config_path)
    config = load_yaml(config_path)
    system = config["system"]
    sweep = config["sweep"]
    quadrature = config["quadrature"]
    bounds_config = config["bounds"]
    output = config["output"]

    seed = int(config["random"]["seed"])
    rng = np.random.default_rng(seed)
    number_of_ports = int(system["number_of_ports"])
    positions = linear_port_positions(
        number_of_ports,
        float(system["aperture_wavelengths"]),
    )
    correlation = clarke_jakes_correlation(positions)

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

    symbols = get_constellation(system["modulation"])
    priors = uniform_symbol_probabilities(symbols.size)
    if system["port_selection"].lower() != "uniform":
        raise ValueError("Figure 7 currently supports uniform selection only.")
    rho = np.full(number_of_ports, 1.0 / number_of_ports)

    transmit_power = float(system["transmit_power"])
    beta_eve = float(system["large_scale_gain_eve"])
    snr_values_db = np.asarray(sweep["eve_snr_db"], dtype=float)
    reference_symbol_index = int(bounds_config["reference_symbol_index"])

    exact = []
    tv_lower = []
    tv_upper_raw = []
    tv_upper = []
    permutation_upper_raw = []
    permutation_upper = []

    for snr_db in snr_values_db:
        snr_linear = 10.0 ** (snr_db / 10.0)
        noise_variance = transmit_power * beta_eve / snr_linear

        tv_result = pairwise_total_variation_bounds(
            symbols=symbols,
            effective_eve_channel=effective_channel,
            port_probabilities=rho,
            symbol_probabilities=priors,
            transmit_power=transmit_power,
            noise_variance=noise_variance,
            reference_symbol_index=reference_symbol_index,
            points_per_axis=int(quadrature["points_per_axis"]),
            noise_margin_sigma=float(quadrature["noise_margin_sigma"]),
            batch_size=int(quadrature["batch_size"]),
        )
        permutation_result = permutation_vulnerability_bound(
            symbols=symbols,
            effective_eve_channel=effective_channel,
            port_probabilities=rho,
            symbol_probabilities=priors,
            transmit_power=transmit_power,
            noise_variance=noise_variance,
            reference_symbol_index=reference_symbol_index,
        )

        exact.append(tv_result["vulnerability"])
        tv_lower.append(tv_result["lower_bound"])
        tv_upper_raw.append(tv_result["upper_bound"])
        tv_upper.append(tv_result["upper_bound_capped"])
        permutation_upper_raw.append(permutation_result["upper_bound"])
        permutation_upper.append(permutation_result["upper_bound_capped"])

        print(
            "SNR={0:6.1f} dB | exact={1:.6f} | TV-L={2:.6f} | "
            "TV-U={3:.6f} | Perm-U={4:.6f}".format(
                snr_db,
                exact[-1],
                tv_lower[-1],
                tv_upper_raw[-1],
                permutation_upper_raw[-1],
            )
        )

    data = {
        "eve_snr_db": snr_values_db,
        "vulnerability_exact": np.asarray(exact),
        "tv_lower_bound": np.asarray(tv_lower),
        "tv_upper_bound_raw": np.asarray(tv_upper_raw),
        "tv_upper_bound_capped": np.asarray(tv_upper),
        "permutation_upper_bound_raw": np.asarray(permutation_upper_raw),
        "permutation_upper_bound_capped": np.asarray(permutation_upper),
        "prior_vulnerability": np.full(snr_values_db.size, 1.0 / symbols.size),
    }

    run_directory = prepare_experiment_directory(
        output["results_root"],
        output["experiment_name"],
        repository_root=".",
    )
    dat_path = run_directory / output["dat_file"]
    pdf_path = run_directory / output["pdf_file"]
    copy_path = run_directory / output["config_copy"]

    save_dat(
        data,
        dat_path,
        comments=[
            "Finite-SNR vulnerability-bound validation.",
            "Upper bounds ending in _capped use min(1, analytical bound).",
            "Reference symbol index is zero-based: {0}.".format(
                reference_symbol_index
            ),
        ],
    )
    copy_configuration(config_path, copy_path)

    figure, axis = plt.subplots(figsize=(5.5, 3.7))
    axis.plot(snr_values_db, exact, "o-", label="Exact vulnerability")
    axis.plot(snr_values_db, tv_lower, "s--", label="TV lower bound")
    axis.plot(snr_values_db, tv_upper, "^-.", label="TV upper bound")
    axis.plot(
        snr_values_db,
        permutation_upper,
        "D:",
        label="Permutation upper bound",
    )
    axis.set_xlabel("Eve-side SNR (dB)")
    axis.set_ylabel("Posterior vulnerability")
    axis.set_ylim(0.0, 1.02)
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

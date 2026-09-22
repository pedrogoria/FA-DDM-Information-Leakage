"""Figure 4: exact vulnerability and finite-SNR TV bounds."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from fa_ddm.channel import (
    clarke_jakes_correlation,
    effective_eve_channel,
    generate_correlated_rician_channel,
    linear_port_positions,
)
from fa_ddm.distinguishability import finite_snr_metrics
from fa_ddm.io import (
    copy_configuration,
    load_yaml,
    prepare_experiment_directory,
    save_dat,
)
from fa_ddm.modulation import get_constellation, uniform_symbol_probabilities
from fa_ddm.vulnerability import monte_carlo_vulnerability


def modulation_key(name):
    return str(name).lower().replace("-", "").replace(" ", "")


def run(config_path="configs/figures/figure_04_finite_snr_bounds.yaml"):
    config_path = Path(config_path)
    config = load_yaml(config_path)
    system = config["system"]
    quadrature = config["quadrature"]
    monte_carlo = config["monte_carlo"]
    output = config["output"]
    seed = int(config["random"]["seed"])

    number_of_ports = int(system["number_of_ports"])
    positions = linear_port_positions(
        number_of_ports, float(system["aperture_wavelengths"])
    )
    correlation = clarke_jakes_correlation(positions)
    channel_rng = np.random.default_rng(seed)
    bob_channel = generate_correlated_rician_channel(
        positions, float(system["bob_angle_deg"]),
        float(system["large_scale_gain_bob"]),
        float(system["rician_factor_bob"]), channel_rng, correlation)
    eve_channel = generate_correlated_rician_channel(
        positions, float(system["eve_angle_deg"]),
        float(system["large_scale_gain_eve"]),
        float(system["rician_factor_eve"]), channel_rng, correlation)
    effective_channel = effective_eve_channel(bob_channel, eve_channel)
    rho = np.full(number_of_ports, 1.0 / number_of_ports)

    snr_db_values = np.asarray(config["sweep"]["eve_snr_db"], dtype=float)
    transmit_power = float(system["transmit_power"])
    beta_eve = float(system["large_scale_gain_eve"])
    data = {"eve_snr_db": snr_db_values}
    preview = {}

    for modulation_index, modulation_name in enumerate(system["modulations"]):
        symbols = get_constellation(modulation_name)
        priors = uniform_symbol_probabilities(symbols.size)
        key = modulation_key(modulation_name)
        exact, lower, upper, tv_identity = [], [], [], []
        mc_values, mc_lower, mc_upper = [], [], []

        print("\nModulation: {}".format(modulation_name))
        for snr_index, snr_db in enumerate(snr_db_values):
            noise_variance = transmit_power * beta_eve / (10.0 ** (snr_db / 10.0))
            result = finite_snr_metrics(
                symbols=symbols,
                effective_eve_channel=effective_channel,
                port_probabilities=rho,
                transmit_power=transmit_power,
                noise_variance=noise_variance,
                reference_symbol_index=int(system["reference_symbol_index"]),
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
            exact.append(result["vulnerability"])
            lower.append(result["lower_bound"])
            upper.append(result["upper_bound"])
            tv_identity.append(result.get("binary_tv_identity", np.nan))
            half_width = 1.96 * standard_error
            mc_values.append(value_mc)
            mc_lower.append(max(1.0 / symbols.size, value_mc - half_width))
            mc_upper.append(min(1.0, value_mc + half_width))
            print("SNR={:6.1f} | exact={:.6f} | lower={:.6f} | upper={:.6f}".format(
                snr_db, exact[-1], lower[-1], upper[-1]))

        data["vulnerability_{}_exact".format(key)] = exact
        data["vulnerability_{}_lower".format(key)] = lower
        data["vulnerability_{}_upper".format(key)] = upper
        data["vulnerability_{}_tv_identity".format(key)] = tv_identity
        data["vulnerability_{}_mc".format(key)] = mc_values
        data["vulnerability_{}_ci95_lower".format(key)] = mc_lower
        data["vulnerability_{}_ci95_upper".format(key)] = mc_upper
        preview[key] = (modulation_name, np.asarray(exact), np.asarray(lower),
                        np.asarray(upper), np.asarray(tv_identity),
                        np.asarray(mc_values), np.asarray(mc_lower),
                        np.asarray(mc_upper))

    run_directory = prepare_experiment_directory(
        output["results_root"], output["experiment_name"], repository_root="."
    )
    dat_path = run_directory / output["dat_file"]
    pdf_path = run_directory / output["pdf_file"]
    save_dat(data, dat_path, comments=[
        "Experiment: {}".format(output["experiment_name"]),
        "Same channel realization is used for BPSK and QPSK.",
        "Monte Carlo columns validate exact quadrature only.",
    ])
    copy_configuration(config_path, run_directory / output["config_copy"])

    figure, axes = plt.subplots(1, 2, figsize=(10.0, 3.7), sharex=True)
    for axis, key in zip(axes, [modulation_key(v) for v in system["modulations"]]):
        name, exact, lower, upper, identity, mc_value, ci_lower, ci_upper = preview[key]
        axis.plot(snr_db_values, exact, color="tab:blue", label="Exact")
        if name.upper() == "BPSK":
            axis.plot(snr_db_values, identity, color="tab:red", linestyle="--", label="TV identity")
        else:
            axis.plot(snr_db_values, lower, color="tab:red", linestyle="--", label="Lower bound")
            axis.plot(snr_db_values, upper, color="black", linestyle="-.", label="Upper bound")
        axis.errorbar(snr_db_values, mc_value,
                      yerr=[mc_value-ci_lower, ci_upper-mc_value],
                      marker="o", linestyle="none", color="tab:blue",
                      capsize=2, label="_nolegend_")
        axis.set_title(name)
        axis.set_xlabel("Eve-side SNR (dB)")
        axis.grid(True)
        axis.legend()
    axes[0].set_ylabel("Posterior vulnerability")
    figure.tight_layout()
    figure.savefig(pdf_path, bbox_inches="tight")
    plt.close(figure)
    print("\nData saved to: {}".format(dat_path.resolve()))
    return data


if __name__ == "__main__":
    run()

"""Figure 3: vulnerability saturation as port density increases."""
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from fa_ddm.channel import (clarke_jakes_correlation, effective_eve_channel,
    generate_correlated_rician_channel, linear_port_positions)
from fa_ddm.io import copy_configuration, load_yaml, prepare_experiment_directory, save_dat
from fa_ddm.modulation import get_constellation, uniform_symbol_probabilities
from fa_ddm.quadrature import quadrature_vulnerability
from fa_ddm.vulnerability import monte_carlo_vulnerability


def aperture_key(value):
    return str(float(value)).replace(".", "p")


def mean_ci95(values):
    values = np.asarray(values, dtype=float)
    mean = float(np.mean(values))
    if values.size <= 1:
        return mean, mean, mean
    half = 1.96 * float(np.std(values, ddof=1) / np.sqrt(values.size))
    return mean, max(0.0, mean - half), min(1.0, mean + half)


def run(config_path="configs/figures/figure_03_port_density_saturation.yaml"):
    config_path = Path(config_path)
    cfg = load_yaml(config_path)
    system, fading = cfg["system"], cfg["fading"]
    quad, mc, output = cfg["quadrature"], cfg["monte_carlo"], cfg["output"]
    seed = int(cfg["random"]["seed"])
    symbols = get_constellation(system["modulation"])
    priors = uniform_symbol_probabilities(symbols.size)
    port_counts = np.asarray(system["number_of_ports"], dtype=int)
    apertures = [float(v) for v in system["aperture_wavelengths"]]
    if system["port_selection"].lower() != "uniform":
        raise ValueError("Only uniform port selection is supported in Figure 3.")

    data = {"number_of_ports": port_counts}
    preview = {}
    for ia, aperture in enumerate(apertures):
        key = aperture_key(aperture)
        theory_means, mc_means, mc_lowers, mc_uppers = [], [], [], []
        print("\nAperture: {} wavelength(s)".format(aperture))
        for i_n, n_ports in enumerate(port_counts):
            positions = linear_port_positions(int(n_ports), aperture)
            corr = clarke_jakes_correlation(positions)
            rho = np.full(int(n_ports), 1.0 / float(n_ports))
            theory_samples, mc_samples = [], []
            for realization in range(int(fading["number_of_realizations"])):
                run_seed = seed + 1000000 * ia + 10000 * i_n + realization
                rng = np.random.default_rng(run_seed)
                h_b = generate_correlated_rician_channel(
                    positions, float(system["bob_angle_deg"]),
                    float(system["large_scale_gain_bob"]),
                    float(system["rician_factor_bob"]), rng, corr)
                h_e = generate_correlated_rician_channel(
                    positions, float(system["eve_angle_deg"]),
                    float(system["large_scale_gain_eve"]),
                    float(system["rician_factor_eve"]), rng, corr)
                g = effective_eve_channel(h_b, h_e)
                theory_samples.append(quadrature_vulnerability(
                    symbols=symbols, effective_eve_channel=g,
                    port_probabilities=rho, symbol_probabilities=priors,
                    transmit_power=float(system["transmit_power"]),
                    noise_variance=float(system["noise_variance_eve"]),
                    points_per_axis=int(quad["points_per_axis"]),
                    noise_margin_sigma=float(quad["noise_margin_sigma"]),
                    batch_size=int(quad["batch_size"])))
                if bool(mc["enabled"]):
                    value, _ = monte_carlo_vulnerability(
                        symbols=symbols, effective_eve_channel=g,
                        port_probabilities=rho, symbol_probabilities=priors,
                        transmit_power=float(system["transmit_power"]),
                        noise_variance=float(system["noise_variance_eve"]),
                        number_of_trials=int(mc["number_of_trials_per_realization"]),
                        rng=np.random.default_rng(run_seed + 50000000))
                    mc_samples.append(value)
            theory_mean = float(np.mean(theory_samples))
            mc_mean, lower, upper = mean_ci95(mc_samples) if mc_samples else (np.nan,) * 3
            theory_means.append(theory_mean)
            mc_means.append(mc_mean)
            mc_lowers.append(lower)
            mc_uppers.append(upper)
            print("N={:2d} | quadrature={:.6f} | MC={:.6f}".format(
                int(n_ports), theory_mean, mc_mean))
        data["vulnerability_aperture_{}_theory".format(key)] = theory_means
        data["vulnerability_aperture_{}_mc".format(key)] = mc_means
        data["vulnerability_aperture_{}_ci95_lower".format(key)] = mc_lowers
        data["vulnerability_aperture_{}_ci95_upper".format(key)] = mc_uppers
        preview[key] = (aperture, np.asarray(theory_means), np.asarray(mc_means),
                        np.asarray(mc_lowers), np.asarray(mc_uppers))
    data["prior_vulnerability"] = np.full(port_counts.size, 1.0 / symbols.size)

    run_dir = prepare_experiment_directory(output["results_root"],
        output["experiment_name"], repository_root=".")
    dat_path, pdf_path = run_dir / output["dat_file"], run_dir / output["pdf_file"]
    save_dat(data, dat_path, comments=[
        "Experiment: {}".format(output["experiment_name"]),
        "Modulation: {}".format(system["modulation"]),
        "Theory: fading-averaged quadrature; marks: Monte Carlo."])
    copy_configuration(config_path, run_dir / output["config_copy"])

    colors, styles, markers = ["tab:blue", "tab:red", "black"], ["-", "--", "-."], ["o", "s", "^"]
    fig, ax = plt.subplots(figsize=(5.3, 3.6))
    for i, key in enumerate(preview):
        aperture, theory, sim, lower, upper = preview[key]
        ax.plot(port_counts, theory, color=colors[i], linestyle=styles[i],
                label="Aperture = {} wavelength".format(aperture))
        ax.errorbar(port_counts, sim, yerr=[sim-lower, upper-sim], color=colors[i],
                    marker=markers[i], linestyle="none", capsize=2, label="_nolegend_")
    ax.axhline(1.0 / symbols.size, color="gray", linestyle=":")
    ax.set_xscale("log", base=2)
    ax.set_xticks(port_counts); ax.set_xticklabels([str(v) for v in port_counts])
    ax.set_xlabel("Number of candidate ports")
    ax.set_ylabel("Average posterior vulnerability")
    ax.grid(True); ax.legend(); fig.tight_layout()
    fig.savefig(pdf_path, bbox_inches="tight"); plt.close(fig)
    print("\nData saved to: {}".format(dat_path.resolve()))
    return data


if __name__ == "__main__":
    run()

"""Figure 6: directional posterior vulnerability around Alice."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from fa_ddm.channel import (
    clarke_jakes_correlation,
    effective_eve_channel,
    far_field_signature,
    generate_correlated_rician_channel,
    hermitian_square_root,
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


def rician_key(value):
    """Return a column-safe Rician-factor identifier."""
    return str(float(value)).replace(".", "p")


def correlated_diffuse_sample(correlation, rng):
    """Generate one unit-power correlated diffuse port vector."""
    number_of_ports = correlation.shape[0]
    white = (
        rng.standard_normal(number_of_ports)
        + 1j * rng.standard_normal(number_of_ports)
    ) / np.sqrt(2.0)
    return hermitian_square_root(correlation).dot(white)


def rician_channel_from_components(
    positions,
    angle_deg,
    large_scale_gain,
    rician_factor,
    diffuse_sample,
):
    """Build a Rician channel while keeping the diffuse sample fixed."""
    signature = far_field_signature(positions, angle_deg)
    los_scale = np.sqrt(rician_factor / (rician_factor + 1.0))
    diffuse_scale = np.sqrt(1.0 / (rician_factor + 1.0))
    return np.sqrt(large_scale_gain) * (
        los_scale * signature + diffuse_scale * diffuse_sample
    )


def angle_grid(sweep):
    """Return an inclusive angular grid."""
    start = float(sweep["start"])
    stop = float(sweep["stop"])
    step = float(sweep["step"])
    if step <= 0.0 or stop < start:
        raise ValueError("The angular sweep is invalid.")
    count = int(round((stop - start) / step)) + 1
    return start + step * np.arange(count)


def run(config_path="configs/figures/figure_06_directional_vulnerability.yaml"):
    """Run the angular vulnerability experiment."""
    config_path = Path(config_path)
    config = load_yaml(config_path)
    system = config["system"]
    quadrature = config["quadrature"]
    monte_carlo = config["monte_carlo"]
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
    rho = np.full(number_of_ports, 1.0 / number_of_ports)

    rng = np.random.default_rng(seed)
    bob_channel = generate_correlated_rician_channel(
        positions,
        float(system["bob_angle_deg"]),
        float(system["large_scale_gain_bob"]),
        float(system["rician_factor_bob"]),
        rng,
        correlation,
    )
    eve_diffuse_sample = correlated_diffuse_sample(correlation, rng)

    angles = angle_grid(config["sweep"]["eve_angle_deg"])
    rician_factors = [float(value) for value in system["rician_factors_eve"]]
    transmit_power = float(system["transmit_power"])
    noise_variance = float(system["noise_variance_eve"])

    data = {"eve_angle_deg": angles}
    preview = {}

    for factor_index, rician_factor in enumerate(rician_factors):
        key = rician_key(rician_factor)
        theory_values = []
        mc_values = []
        ci_lower = []
        ci_upper = []

        print("\nEve Rician factor: {0}".format(rician_factor))
        for angle_index, angle_deg in enumerate(angles):
            eve_channel = rician_channel_from_components(
                positions=positions,
                angle_deg=float(angle_deg),
                large_scale_gain=float(system["large_scale_gain_eve"]),
                rician_factor=rician_factor,
                diffuse_sample=eve_diffuse_sample,
            )
            effective_channel = effective_eve_channel(
                bob_channel,
                eve_channel,
            )

            theory = quadrature_vulnerability(
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
            theory_values.append(theory)

            if bool(monte_carlo["enabled"]):
                estimate, standard_error = monte_carlo_vulnerability(
                    symbols=symbols,
                    effective_eve_channel=effective_channel,
                    port_probabilities=rho,
                    symbol_probabilities=priors,
                    transmit_power=transmit_power,
                    noise_variance=noise_variance,
                    number_of_trials=int(monte_carlo["number_of_trials"]),
                    rng=np.random.default_rng(
                        seed + 100000 * (factor_index + 1) + angle_index
                    ),
                )
                half_width = 1.96 * standard_error
                mc_values.append(estimate)
                ci_lower.append(max(1.0 / symbols.size, estimate - half_width))
                ci_upper.append(min(1.0, estimate + half_width))
            else:
                mc_values.append(np.nan)
                ci_lower.append(np.nan)
                ci_upper.append(np.nan)

        theory_values = np.asarray(theory_values)
        mc_values = np.asarray(mc_values)
        ci_lower = np.asarray(ci_lower)
        ci_upper = np.asarray(ci_upper)

        data["vulnerability_kappa_{0}_theory".format(key)] = theory_values
        data["vulnerability_kappa_{0}_mc".format(key)] = mc_values
        data["vulnerability_kappa_{0}_ci95_lower".format(key)] = ci_lower
        data["vulnerability_kappa_{0}_ci95_upper".format(key)] = ci_upper
        data["vulnerability_kappa_{0}_ci95_plus".format(key)] = (
            ci_upper - mc_values
        )
        data["vulnerability_kappa_{0}_ci95_minus".format(key)] = (
            mc_values - ci_lower
        )
        preview[key] = (
            rician_factor,
            theory_values,
            mc_values,
            ci_lower,
            ci_upper,
        )

    run_directory = prepare_experiment_directory(
        output["results_root"],
        output["experiment_name"],
        repository_root=".",
    )
    dat_path = run_directory / output["dat_file"]
    pdf_path = run_directory / output["pdf_file"]

    save_dat(
        data,
        dat_path,
        comments=[
            "Directional vulnerability for fixed Bob direction.",
            "Bob angle: {0} degree.".format(system["bob_angle_deg"]),
            "One Eve diffuse realization is retained for all angles and Rician factors.",
            "Theory columns are quadrature; marks are Monte Carlo.",
        ],
    )
    copy_configuration(config_path, run_directory / output["config_copy"])

    figure = plt.figure(figsize=(6.0, 5.0))
    axis = figure.add_subplot(111, projection="polar")
    colors = ["tab:blue", "tab:red", "black"]
    line_styles = ["-", "--", "-."]
    markers = ["o", "s", "^"]

    radians = np.deg2rad(angles)
    for index, key in enumerate(preview):
        factor, theory, simulation, lower, upper = preview[key]
        color = colors[index % len(colors)]
        axis.plot(
            radians,
            theory,
            color=color,
            linestyle=line_styles[index % len(line_styles)],
            linewidth=1.1,
            label="K_E={0}".format(factor),
        )
        axis.errorbar(
            radians,
            simulation,
            yerr=[simulation - lower, upper - simulation],
            color=color,
            marker=markers[index % len(markers)],
            linestyle="none",
            markersize=2.8,
            capsize=1.5,
            label="_nolegend_",
        )

    axis.set_theta_zero_location("N")
    axis.set_theta_direction(-1)
    axis.set_ylim(0.0, 1.0)
    axis.grid(True)
    axis.legend(loc="upper right", bbox_to_anchor=(1.25, 1.10))
    figure.tight_layout()
    figure.savefig(pdf_path, bbox_inches="tight")
    plt.close(figure)

    print("Data saved to: {0}".format(dat_path.resolve()))
    print("Preview saved to: {0}".format(pdf_path.resolve()))
    return data


if __name__ == "__main__":
    run()

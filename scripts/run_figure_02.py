"""Figure 2: posterior vulnerability over Eve's planar position."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from fa_ddm.channel import (
    clarke_jakes_correlation,
    effective_eve_channel,
    generate_correlated_rician_channel,
)
from fa_ddm.geometry import (
    far_field_signature_3d,
    free_space_large_scale_gain,
    linear_aperture_coordinates,
    receiver_geometry,
)
from fa_ddm.io import (
    copy_configuration,
    load_yaml,
    prepare_experiment_directory,
    save_dat,
)
from fa_ddm.modulation import get_constellation, uniform_symbol_probabilities
from fa_ddm.quadrature import quadrature_vulnerability
from fa_ddm.spatial_map import (
    min_entropy_leakage,
    planar_receiver_grid,
    position_descriptors,
)


def modulation_key(name):
    """Return a lowercase column-safe modulation identifier."""
    return str(name).lower().replace("-", "").replace(" ", "")


def correlated_diffuse_sample(correlation, rng):
    """Generate one correlated unit-power complex Gaussian port vector."""
    eigenvalues, eigenvectors = np.linalg.eigh(
        0.5 * (correlation + correlation.T)
    )
    root = (eigenvectors * np.sqrt(np.clip(eigenvalues, 0.0, None))) @ eigenvectors.T
    white = (
        rng.standard_normal(correlation.shape[0])
        + 1j * rng.standard_normal(correlation.shape[0])
    ) / np.sqrt(2.0)
    return root.dot(white)


def rician_channel_from_position(
    port_coordinates,
    position,
    correlation,
    diffuse_sample,
    rician_factor,
    large_scale_gain,
):
    """Build a Rician channel with position-dependent LOS and fixed diffuse sample."""
    _, direction = receiver_geometry(position)
    signature = far_field_signature_3d(port_coordinates, direction)
    los_scale = np.sqrt(rician_factor / (rician_factor + 1.0))
    diffuse_scale = np.sqrt(1.0 / (rician_factor + 1.0))
    return np.sqrt(large_scale_gain) * (
        los_scale * signature + diffuse_scale * diffuse_sample
    )


def run(config_path="configs/figures/figure_02_eve_position_map.yaml"):
    """Run the spatial vulnerability-map experiment."""
    config_path = Path(config_path)
    config = load_yaml(config_path)
    system = config["system"]
    map_config = config["map"]
    quadrature = config["quadrature"]
    output = config["output"]

    seed = int(config["random"]["seed"])
    rng = np.random.default_rng(seed)
    number_of_ports = int(system["number_of_ports"])
    aperture = float(system["aperture_wavelengths"])
    port_coordinates = linear_aperture_coordinates(
        number_of_ports, aperture, axis=system["aperture_axis"]
    )
    one_dimensional_positions = port_coordinates[:, {"x": 0, "y": 1, "z": 2}[system["aperture_axis"]]]
    correlation = clarke_jakes_correlation(one_dimensional_positions)

    bob_position = np.asarray(system["bob_position_wavelengths"], dtype=float)
    _, bob_direction = receiver_geometry(bob_position)
    # Convert the 3-D direction to the broadside angle expected by the existing
    # channel generator for the x-axis aperture.
    bob_angle_deg = np.rad2deg(np.arcsin(bob_direction[0]))
    bob_channel = generate_correlated_rician_channel(
        one_dimensional_positions,
        bob_angle_deg,
        float(system["large_scale_gain_bob"]),
        float(system["rician_factor_bob"]),
        rng,
        correlation,
    )

    # One diffuse realization is retained over the complete Eve-position map.
    # The position dependence enters through LOS direction and path loss.
    eve_diffuse_sample = correlated_diffuse_sample(correlation, rng)

    if system["port_selection"].lower() != "uniform":
        raise ValueError("Figure 2 currently supports uniform port selection only.")
    rho = np.full(number_of_ports, 1.0 / number_of_ports)

    x_values = np.linspace(
        float(map_config["x_min_wavelengths"]),
        float(map_config["x_max_wavelengths"]),
        int(map_config["x_points"]),
    )
    y_values = np.linspace(
        float(map_config["y_min_wavelengths"]),
        float(map_config["y_max_wavelengths"]),
        int(map_config["y_points"]),
    )
    positions = planar_receiver_grid(
        x_values, y_values, float(map_config["z_wavelengths"])
    )

    data = {
        "grid_row": np.repeat(np.arange(y_values.size), x_values.size),
        "grid_col": np.tile(np.arange(x_values.size), y_values.size),
        "eve_x_lambda": positions[:, 0],
        "eve_y_lambda": positions[:, 1],
        "eve_z_lambda": positions[:, 2],
    }
    descriptors = [position_descriptors(position) for position in positions]
    distances = np.asarray([item[0] for item in descriptors])
    data["eve_distance_lambda"] = distances
    data["eve_azimuth_deg"] = np.asarray([item[1] for item in descriptors])
    data["eve_elevation_deg"] = np.asarray([item[2] for item in descriptors])

    modulations = list(system["modulations"])
    vulnerabilities = {
        modulation_key(name): np.empty(positions.shape[0], dtype=float)
        for name in modulations
    }

    for position_index, position in enumerate(positions):
        beta_eve = free_space_large_scale_gain(
            distances[position_index],
            reference_gain=float(system["eve_reference_gain"]),
            reference_distance_wavelengths=float(
                system["eve_reference_distance_wavelengths"]
            ),
            path_loss_exponent=float(system["path_loss_exponent_eve"]),
        )
        eve_channel = rician_channel_from_position(
            port_coordinates,
            position,
            correlation,
            eve_diffuse_sample,
            float(system["rician_factor_eve"]),
            beta_eve,
        )
        effective_channel = effective_eve_channel(bob_channel, eve_channel)

        for modulation_name in modulations:
            key = modulation_key(modulation_name)
            symbols = get_constellation(modulation_name)
            priors = uniform_symbol_probabilities(symbols.size)
            vulnerabilities[key][position_index] = quadrature_vulnerability(
                symbols=symbols,
                effective_eve_channel=effective_channel,
                port_probabilities=rho,
                symbol_probabilities=priors,
                transmit_power=float(system["transmit_power"]),
                noise_variance=float(system["noise_variance_eve"]),
                points_per_axis=int(quadrature["points_per_axis"]),
                noise_margin_sigma=float(quadrature["noise_margin_sigma"]),
                batch_size=int(quadrature["batch_size"]),
            )

        if (position_index + 1) % max(1, positions.shape[0] // 20) == 0:
            print("Completed {0}/{1} positions".format(
                position_index + 1, positions.shape[0]
            ))

    for modulation_name in modulations:
        key = modulation_key(modulation_name)
        data["vulnerability_{0}".format(key)] = vulnerabilities[key]
        data["min_entropy_leakage_{0}".format(key)] = min_entropy_leakage(
            vulnerabilities[key], get_constellation(modulation_name).size
        )

    run_directory = prepare_experiment_directory(
        output["results_root"], output["experiment_name"], repository_root="."
    )
    dat_path = run_directory / output["dat_file"]
    config_copy_path = run_directory / output["config_copy"]
    pdf_path = run_directory / output["pdf_file"]

    comments = [
        "Experiment: {0}".format(output["experiment_name"]),
        "Rows are ordered by y first and x second; mesh cols = {0}.".format(
            x_values.size
        ),
        "Coordinates and distances are normalized by wavelength.",
        "One diffuse channel realization is fixed over the full map.",
        "Position dependence enters through LOS direction and path loss.",
    ]
    save_dat(data, dat_path, comments=comments)
    copy_configuration(config_path, config_copy_path)

    figure, axes = plt.subplots(1, len(modulations), figsize=(5.4 * len(modulations), 4.2))
    if len(modulations) == 1:
        axes = [axes]
    for axis, modulation_name in zip(axes, modulations):
        key = modulation_key(modulation_name)
        matrix = vulnerabilities[key].reshape(y_values.size, x_values.size)
        image = axis.imshow(
            matrix,
            origin="lower",
            extent=[x_values[0], x_values[-1], y_values[0], y_values[-1]],
            aspect="equal",
            vmin=1.0 / get_constellation(modulation_name).size,
            vmax=1.0,
        )
        axis.plot(bob_position[0], bob_position[1], marker="*", color="red", markersize=9)
        axis.set_title(modulation_name)
        axis.set_xlabel("Eve x / wavelength")
        axis.set_ylabel("Eve y / wavelength")
        figure.colorbar(image, ax=axis, label="Posterior vulnerability")
    figure.tight_layout()
    figure.savefig(pdf_path, bbox_inches="tight")
    plt.close(figure)

    print("Data saved to: {0}".format(dat_path.resolve()))
    print("Preview saved to: {0}".format(pdf_path.resolve()))
    return data


if __name__ == "__main__":
    run()

"""Tests for versioned experiment-result directories."""

from fa_ddm.io import prepare_experiment_directory, save_dat


def test_prepare_versioned_experiment_directory(tmp_path):
    run_directory = prepare_experiment_directory(
        tmp_path / "results",
        "figure_01_validation",
        run_reference="20260922_a1b2c3d",
    )
    assert run_directory.is_dir()
    assert run_directory.name == "20260922_a1b2c3d"
    assert run_directory.parent.name == "figure_01_validation"


def test_save_dat_has_comments_and_header(tmp_path):
    output = tmp_path / "result.dat"
    save_dat(
        {"eve_snr_db": [0.0], "vulnerability": [0.25]},
        output,
        comments=["Git commit: a1b2c3d"],
    )
    lines = output.read_text(encoding="utf-8").splitlines()
    assert lines[0] == "# Git commit: a1b2c3d"
    assert lines[1] == "eve_snr_db vulnerability"

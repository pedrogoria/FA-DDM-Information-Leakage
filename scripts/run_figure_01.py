"""Runner for Fig. 1: quadrature versus Monte Carlo validation."""

from pathlib import Path


def main() -> None:
    config = Path("configs/figures/figure_01_validation.yaml")
    print(f"Configuration found: {config.resolve()}")
    print("Next step: implement the shared channel and vulnerability functions.")


if __name__ == "__main__":
    main()

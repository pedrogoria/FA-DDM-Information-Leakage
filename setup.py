from setuptools import find_packages, setup


setup(
    name="fa-ddm-information-leakage",
    version="0.1.0",
    description=(
        "Numerical framework for information-leakage "
        "analysis of FA-assisted DDM"
    ),
    packages=find_packages(),
    python_requires=">=3.8",
)
"""Setup of the `cvp` and `env` packages."""
from pathlib import Path

from setuptools import find_packages, setup

CWD = Path(__file__).absolute().parent
REQUIREMENTS = [
    line.strip() for line in (CWD / "requirements.txt").read_text().splitlines()
    if line.strip() and not line.startswith("#")
]

setup(
    name="cvp",
    version="0.1.0",
    description="Compositional Visual Planning via Inference-Time Diffusion Scaling",
    long_description=(CWD / "README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    author="Yixin Zhang",
    license="MIT",
    python_requires=">=3.10",
    packages=find_packages(include=["cvp", "cvp.*", "env", "env.*"]),
    package_data={"env": ["assets/**/*", "eval_states/*.json"]},
    install_requires=REQUIREMENTS,
)

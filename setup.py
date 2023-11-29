# Runs the installation. See the following for more detail:
# https://docs.python.org/3/distutils/setupscript.html

from setuptools import find_packages, setup

# Avoids duplication of requirements
with open("requirements.txt") as file:
    requirements = file.read().splitlines()

setup(
    name="band_torch",
    author="Nina Kudryashova",
    author_email="kudryashova.nina@gmail.com",
    description="A PyTorch implementation of "
    "Behavior aligned neural dynamics (BAND)"
    " based on Latent Factor Analysis via Dynamical Systems (LFADS)",
    url="https://github.com/NinelK/BAND-torch",
    install_requires=requirements,
    packages=find_packages(),
)

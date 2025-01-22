"""Installation script for the 'defnder' python package."""

import os
import toml

from setuptools import setup, find_packages  # Use find_packages to include sub-packages

# Obtain the extension data from the extension.toml file
EXTENSION_PATH = os.path.dirname(os.path.realpath(__file__))
# Read the extension.toml file
EXTENSION_TOML_DATA = toml.load(os.path.join(EXTENSION_PATH, "config", "extension.toml"))

# Minimum dependencies required prior to installation
INSTALL_REQUIRES = [
    "psutil",
    "toml",  # Ensure toml is included as well if it's a required dependency
]

# Installation operation
setup(
    name="Isaac_drone",
    packages=find_packages(include=["Isaac_drone", "Isaac_drone.*"]),  # Include sub-packages
    author=EXTENSION_TOML_DATA["package"]["author"],
    maintainer=EXTENSION_TOML_DATA["package"]["maintainer"],
    url=EXTENSION_TOML_DATA["package"]["repository"],
    version=EXTENSION_TOML_DATA["package"]["version"],
    description=EXTENSION_TOML_DATA["package"]["description"],
    keywords=EXTENSION_TOML_DATA["package"]["keywords"],
    install_requires=INSTALL_REQUIRES,
    license="MIT",
    include_package_data=True,  # Necessary for package_data to work
    package_data={
        "Isaac_drone": [
            "tasks/Isaac_drone/agents/*.yaml", 
            "tasks/Isaac_drone_v1/agents/*.yaml",  
            "tasks/Isaac_drone_v2/agents/*.yaml",  
        ]
    },
    python_requires=">=3.10",
    classifiers=[
        "Natural Language :: English",
        "Programming Language :: Python :: 3.10",
        "Isaac Sim :: 2023.1.1",
        "Isaac Sim :: 4.2.0.2",
    ],
    zip_safe=False,
)



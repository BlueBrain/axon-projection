# LICENSE HEADER MANAGED BY add-license-header
#
# Copyright (c) 2023-2024 Blue Brain Project, EPFL.
#
# This file is part of axon-projection.
# See https://github.com/BlueBrain/axon-projection for further info.
#
# SPDX-License-Identifier: Apache-2.0
#

"""Setup for the axon-projection package."""

from pathlib import Path

from setuptools import find_namespace_packages
from setuptools import setup

reqs = [
    "axon-synthesis>=0.1.5",
    "click>=8",
    "matplotlib>=3.9.0",
    "networkx>=3.1",
    "neurom>=3.2.11",
    "nexusforge>=0.8.1",
    "numpy>=1.26.4",
    "pandas>=1.5",
    "plotly>=5.17.0",
    "plotly-helper>=0.0.8",
    "pycirclize>=1.6.0",
    "scikit-learn>=1.3.0",
    "tabulate>=0.9.0",
    "voxcell>=3.1.9",
]

doc_reqs = [
    "docutils<0.21",
    "m2r2",
    "sphinx",
    "sphinx-bluebrain-theme",
    "sphinx-click",
]

test_reqs = [
    "mock>=3",
    "pytest>=8.3",
    "pytest-click>=1.1",
    "pytest-console-scripts>=1.4",
    "pytest-cov>=5.0",
    "pytest-html>=4.1",
]

setup(
    name="axon-projection",
    author="Blue Brain Project, EPFL",
    description=(
        "A code that analyses long-range axons provided as input, and "
        "classify them based on the brain regions they project to."
    ),
    long_description=Path("README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    url="https://axon-projection.readthedocs.io",
    project_urls={
        "Tracker": "https://github.com/BlueBrain/axon-projection/issues",
        "Source": "https://github.com/BlueBrain/axon-projection",
    },
    license="Apache License 2.0",
    packages=find_namespace_packages(include=["axon_projection*"]),
    python_requires=">=3.9",
    use_scm_version=True,
    setup_requires=[
        "setuptools_scm",
    ],
    install_requires=reqs,
    extras_require={
        "docs": doc_reqs,
        "test": test_reqs,
    },
    entry_points={
        "console_scripts": [
            "axon-projection=axon_projection.cli:main",
        ],
    },
    include_package_data=True,
    classifiers=[
        # TODO: Update to relevant classifiers
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
    ],
)

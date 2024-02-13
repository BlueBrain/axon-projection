"""Setup for the axon-projection package."""
import importlib.util
from pathlib import Path

from setuptools import find_namespace_packages
from setuptools import setup

spec = importlib.util.spec_from_file_location(
    "axon_projection.version",
    "axon_projection/version.py",
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
VERSION = module.VERSION

reqs = [
    "click>=7",
    "matplotlib",
    "networkx>=3.1",
    "neurom>=3.2.4",
    "nexusforge>=0.8.1",
    "numpy",
    "pandas>=1.5.3",
    "scikit-learn>=1.3.0",
    "voxcell>=3.1.5",
    "plotly>=5.17.0",
    "plotly-helper>=0.0.8",
    "axon-synthesis>=0.1.0.dev0",
    "synthesis_workflow>=1.0.2",
]

doc_reqs = [
    "m2r2",
    "sphinx",
    "sphinx-bluebrain-theme",
    "sphinx-click",
]

test_reqs = [
    "mock>=3",
    "pytest>=6",
    "pytest-click>=1",
    "pytest-console-scripts>=1.3",
    "pytest-cov>=3",
    "pytest-html>=2",
]

setup(
    name="axon-projection",
    author="cells",
    author_email="cells@groupes.epfl.ch",
    description="A code that analyses long-range axons provided as input, "
    + "and classify them based on the brain regions they project to.",
    long_description=Path("README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    url="https://bbpteam.epfl.ch/documentation/projects/axon-projection",
    project_urls={
        "Tracker": "https://bbpteam.epfl.ch/project/issues/projects/CELLS/issues",
        "Source": "https://bbpgitlab.epfl.ch/neuromath/petkantc/axon-projection",
    },
    license="BBP-internal-confidential",
    packages=find_namespace_packages(include=["axon_projection*"]),
    python_requires=">=3.8",
    version=VERSION,
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
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
    ],
)

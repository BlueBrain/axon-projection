# LICENSE HEADER MANAGED BY add-license-header
#
# Copyright (c) 2023-2024 Blue Brain Project, EPFL.
#
# This file is part of axon-projection.
# See https://github.com/BlueBrain/axon-projection for further info.
#
# SPDX-License-Identifier: Apache-2.0
#

"""Spack module for the axon-projection distribution."""

# Copyright 2013-2018 Lawrence Livermore National Security, LLC and other
# Spack Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
# flake8: noqa
from spack import *


# replace all 'x-y' with 'xY' (e.g. 'Py-morph-tool' -> 'PyMorphTool')
class Py_axon_projection(PythonPackage):
    """A code that analyses long-range axons provided as input, and classify them based on the brain regions they project to."""

    homepage = "https://axon-projection.readthedocs.io"
    git = "https://github.com/BlueBrain/axon-projection"

    version("develop", branch="master")
    version("0.1.0.dev0", tag="0.1.0.dev0")

    depends_on("py-setuptools", type="build")
    # type=("build", "run") if specifying entry points in "setup.py"

    # for all "foo>=X" in "install_requires" and "extra_requires":
    # depends_on("py-foo@<min>:")

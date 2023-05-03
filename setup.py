from setuptools import find_packages, setup

from bosot import __version__

name = "bosot"
version = __version__
description = "Structural Kernel Search via BO and OT"

setup(name=name, version=version, packages=find_packages(exclude=["tests"]), description=description)

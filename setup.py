# !/usr/bin/env python
from setuptools import find_packages, setup

__version__ = "0.0.5"

if __name__ == "__main__":
    setup(
        name="hyperlpr3",
        version=__version__,
        description="vehicle license plate recognition.",
        url="https://github.com/tunmx/LPRV3.git",
        author="Tunm",
        author_email="tunmxy@163.com",
        keywords="vehicle license plate recognition",
        packages=find_packages(exclude=("resource", "utils")),
        classifiers=[
            "License :: OSI Approved :: Apache Software License",
            "Operating System :: OS Independent",
            "Programming Language :: Python :: 3",
        ],
        install_requires=[
            "opencv-python",
            "onnxruntime",
            "tqdm",
            "requests",
        ],
        license="Apache License 2.0",
        zip_safe=False,
    )
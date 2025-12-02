"""
Setup configuration for PIC3D package.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="pic3d",
    version="0.1.0",
    author="PIC3D Contributors",
    description="A 3D Particle-In-Cell Simulation Framework for Laser-Plasma Interactions",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/vvoidpointer/PIC3D",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Physics",
    ],
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.20.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "matplotlib>=3.5.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "pic3d-example=pic3d.examples.laser_plasma_example:main",
        ],
    },
)

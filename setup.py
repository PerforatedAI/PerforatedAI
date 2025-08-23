# Copyright (c) 2025 Perforated AI

from setuptools import setup

setup(
    name="perforatedai",
    version="1.0.9",
    packages=["perforatedai"],
    author="PerforatedAI",
    author_email="rorry@perforatedai.com",
    description="perforatedai",
    classifiers=[
        "Programming Language: Python :: 3",
        "License: TBD",
        "Operating System: OS Independent",
    ],
    python_requires=">=3.6",
    install_requires=[
        "torch",
        "torchvision",
        "matplotlib",
        "pandas",
        "rsa",
        "pyyaml",
        "safetensors",
    ],
)

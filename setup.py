from setuptools import setup

setup(
    name="perforatedai",
    # Remember to also edit setupCython
<<<<<<< HEAD
    version="3.0.9",
=======
    version="3.0.7",
>>>>>>> c10f400ff7 (new version for compile)
    packages=["perforatedai"],
    author="PerforatedAI",
    author_email="rorry@perforatedai.com",
    description="perforatedai baseline package",
    license="Apache 2.0",
    classifiers=[
        "Programming Language: Python :: 3",
        "License: Apache 2.0",
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
    # may need setuptools upgraded
)

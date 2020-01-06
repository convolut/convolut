# -*- coding: utf-8 -*-
import io

from setuptools import setup, find_packages

with io.open("README.md", "r", encoding="utf-8") as f:
    readme = f.read()

version = "0.0.4"

setup(
    name="convolut",
    version=version,
    description="Distributed development and modularity for deep learning",
    long_description=readme,
    long_description_content_type="text/markdown",
    author="Andrey Avdeev",
    author_email="seorazer@gmail.com",
    license="Apache 2.0",
    packages=find_packages(),
    zip_safe=False,
    python_requires=">=3.7",
    install_requires=[
        "decouple>=0.0.7",
        "torch>=1.3.1",
    ],
    keywords="convolut modularity decoupled dl",
    url="https://github.com/convolut/convolut",
)

#!/usr/bin/env python3
"""Setup script for ChurnForge synthetic dataset generator."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="churnforge",
    version="1.0.0",
    author="ChurnForge",
    description="Universal synthetic dataset generator for any business domain",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/churn-forge",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Data Scientists",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Software Development :: Testing",
    ],
    python_requires=">=3.7",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0",
            "black>=21.0",
            "flake8>=3.8",
            "mypy>=0.800",
        ],
    },
    entry_points={
        "console_scripts": [
            "churnforge=config_dataset_generator:main",
            "churnforge-batch=config_batch_generator:main",
            "churnforge-temporal=config_temporal_generator:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.json", "*.md"],
    },
    keywords="synthetic data, dataset generation, machine learning, churn prediction, data science",
    project_urls={
        "Bug Reports": "https://github.com/yourusername/churn-forge/issues",
        "Source": "https://github.com/yourusername/churn-forge",
        "Documentation": "https://github.com/yourusername/churn-forge/blob/main/README.md",
    },
) 
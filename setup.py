from setuptools import setup, find_packages
import os

# Read the README file
def read_readme():
    try:
        with open("README.md", "r", encoding="utf-8") as fh:
            return fh.read()
    except FileNotFoundError:
        return "Python implementation of the Double Post-Lasso estimator for treatment effect estimation"

# Read requirements
def read_requirements():
    try:
        with open("requirements.txt", "r", encoding="utf-8") as fh:
            return [line.strip() for line in fh if line.strip() and not line.startswith("#")]
    except FileNotFoundError:
        # Fallback to hardcoded requirements if file not found
        return [
            "numpy",
            "scikit-learn", 
            "statsmodels",
            "pandas"
        ]

setup(
    name="pydoublelasso",
    version="0.1.0",
    author="Vasco Yasenov",
    author_email="",  # Add your email if desired
    description="Python implementation of the Double Post-Lasso estimator for treatment effect estimation",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/vyasenov/pydoublelasso",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Scientific/Engineering :: Mathematics",
    ],
    python_requires=">=3.7",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.8",
            "mypy>=0.800",
        ],
        "docs": [
            "sphinx>=3.0",
            "sphinx-rtd-theme>=0.5",
        ],
    },
    keywords="lasso, treatment effects, causal inference, high-dimensional, econometrics",
    project_urls={
        "Bug Reports": "https://github.com/vyasenov/pydoublelasso/issues",
        "Source": "https://github.com/vyasenov/pydoublelasso",
        "Documentation": "https://github.com/vyasenov/pydoublelasso#readme",
    },
) 
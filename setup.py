from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="tfilterspy",
    use_scm_version=True,
    setup_requires=["setuptools_scm"],
    author="Thabang L. Mashinini-Sekgoto, Lebogang L. Sekgoto, Palesa L. Sekgoto",
    author_email="thabangline@gmail.com",
    description="A Python package for Bayesian filtering models such as Kalman and Particle Filters.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ubunye-ai-ecosystems/tfilterspy",
    project_urls={
    "Documentation": "https://ubunye-ai-ecosystems.github.io/tfilterspy/",
    "Source": "https://github.com/ubunye-ai-ecosystems/tfilterspy",
    "Tracker": "https://github.com/ubunye-ai-ecosystems/tfilterspy/issues",
    "Logo": "https://raw.githubusercontent.com/ubunye-ai-ecosystems/tfilterspy/main/branding/logo/tfilters-logo.jpeg",
    },

    packages=find_packages(),
    install_requires=[
        "numpy>=1.20",
        "scipy>=1.7",
        "dask>=2021.1.0",
    ],
    extras_require={
        "dev": ["pytest", "sphinx", "sphinx-rtd-theme", "nbsphinx"],
    },
    keywords="kalman filter, particle filter, Bayesian filtering, distributed computing, Dask, UAIE, Ubunye-AI-Ecosystems",
    python_requires=">=3.8",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_data={
        "": ["examples/notebooks/*.ipynb", "README.md"],
    },
    include_package_data=True,
    # Uncomment the following if you have command-line tools
    # entry_points={
    #     "console_scripts": [
    #         "tfilterpy=tfilterpy.cli:main",
    #     ],
    # },
)

from setuptools import setup, find_packages

setup(
    name="deepanm",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.20.0",
        "scikit-learn>=1.0.0",
        "matplotlib>=3.5.0",
        "scipy>=1.7.0",
        "pytest>=7.0.0",
        "seaborn>=0.12.0",
        "requests>=2.28.0"
    ],
    license="MIT",
)

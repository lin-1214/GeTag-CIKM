import os

from setuptools import setup, find_packages

# README and requirements live at the pipeline root (one level above Downstream/).
_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")

with open(os.path.join(_ROOT, "README.md"), "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open(os.path.join(_ROOT, "requirements.txt"), "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="getag",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A tagging system for structured data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/GeTag",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
)

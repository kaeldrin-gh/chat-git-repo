"""Setup configuration for codechat package."""

from setuptools import find_packages, setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="codechat",
    version="1.0.0",
    author="Chat Git Repo",
    author_email="contact@example.com",
    description="Chat with Your Codebase - RAG system for GitHub repositories",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/example/codechat",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "black==24.4.2",
            "isort==5.13.2", 
            "flake8==7.0.0",
            "mypy==1.10.0",
            "pytest==8.2.2",
            "pytest-cov==5.0.0",
            "coverage==7.5.3",
        ],
    },
    entry_points={
        "console_scripts": [
            "codechat=codechat.__main__:main",
        ],
    },
)

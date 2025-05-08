from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = fh.read().splitlines()

setup(
    name="ottonomiutils",
    version="0.1.0",
    author="OttonomiAI",
    # author_email="your.email@example.com",  # Update this
    description="A high-performance Python package for ONNX model inference with GPU acceleration support",
    long_description=long_description,
    long_description_content_type="text/markdown",
    # url="https://github.com/yourusername/ottonomiutils",  # Update this
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.7",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "isort>=5.0",
            "mypy>=0.910",
            "flake8>=3.9",
        ],
    },
) 
"""Setup script for EMU7800 Python port."""

from setuptools import setup, find_packages

setup(
    name="emu7800",
    version="1.0.0",
    description="Atari 7800/2600 Emulator - Python Port",
    long_description=open("README.md").read() if __import__("os").path.exists("README.md") else "",
    long_description_content_type="text/markdown",
    author="Original: Mike Murphy, Python port: reyco2000",
    license="MIT",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "pygame>=2.5.0",
        "numpy>=1.24.0",
    ],
    entry_points={
        "console_scripts": [
            "emu7800=emu7800.main:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: End Users/Desktop",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Games/Entertainment",
        "Topic :: System :: Emulators",
    ],
)

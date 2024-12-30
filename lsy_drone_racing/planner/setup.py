from setuptools import setup, find_packages

setup(
    name="planner",  # Name of your package
    version="0.1.0",  # Version
    packages=find_packages(where="external"),  # Automatically find packages in the directory
    package_dir={"": "external"},
    description="A library for my project",
    author="Your Name",
    author_email="your.email@example.com",
)

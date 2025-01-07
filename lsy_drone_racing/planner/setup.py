from setuptools import setup, find_packages

setup(
    name="planner",  # Name of your package
    version="0.1.0",  # Version
    packages=find_packages(where="external"),  # Automatically find packages in the directory
    package_dir={"": "external"},
    description="A planner library for lsy's drone racing course",
    author="Shotaro Itahara",
    author_email="shotaro.itahara@tum.de",
)

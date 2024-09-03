from setuptools import setup, find_packages
import os

# Utility function to read the requirements file
def parse_requirements(filename):
    with open(filename, 'r') as file:
        return [line.strip() for line in file if line.strip() and not line.startswith('#')]

# Utility function to read the README file
def read_readme(filename):
    with open(filename, 'r', encoding='utf-8') as file:
        return file.read()

setup(
    name="minipath",
    version="0.1.9",
    description="A tool for processing and analyzing digital pathology images stored in DICOM format.",
    long_description=read_readme('README.md'),
    long_description_content_type='text/markdown',
    author="Steven Hart",
    author_email="Hart.Steven@Mayo.edu",
    license="MIT",
    packages=find_packages(),
    install_requires=parse_requirements('requirements.txt'),
    python_requires='>=3.7',
)
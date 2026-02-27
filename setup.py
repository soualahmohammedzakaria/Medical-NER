"""Medical-NER package setup."""

from setuptools import setup, find_packages

setup(
    name="medical-ner",
    version="0.1.0",
    description="Named Entity Recognition for medical and biomedical text.",
    author="",
    python_requires=">=3.10",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
)

# Updated setup.py
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="semiauto-classification",
    version="1.0.4",
    author="Akshat-Sharma-110011",
    author_email="akshatsharma.business.1310@gmail.com",
    description="Automated classification pipeline for machine learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Akshat-Sharma-110011/SemiAuto-Classification",
    packages=find_packages(),
    package_data={
        'semiauto_classification': ['pipeline_utils.py']
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
    install_requires=[
        'numpy==1.24.4',  # Allow compatible NumPy versions
        'pandas',
        'scikit-learn',
        'scipy',
        'PyYAML',
        'cloudpickle==2.2.1',
        'fastapi',
        'uvicorn',
        'matplotlib',
        'seaborn',
        'optuna',
        'featuretools',
        'imbalanced-learn'
    ],
    entry_points={
        'console_scripts': [
            'semiauto-classification = semiauto_classification.app:main',
        ],
    },
    include_package_data=True,
)
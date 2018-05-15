#!/usr/bin/env python

from setuptools import setup
from setuptools import find_packages


setup(
    name='kerassurgeon',
    version="0.1.0",
    url='https://github.com/BenWhetton/keras-surgeon',
    license='MIT',
    description='A library for performing network surgery on trained Keras '
                'models. Useful for deep neural network pruning.',
    author='Ben Whetton',
    author_email='Ben.Whetton@gmail.com',
    python_requires='>=3',
    install_requires=['keras>=2.0.7'],
    extras_require={'examples': ['pandas'], },
    tests_require=['pytest'],
    packages=find_packages('src'),
    package_dir={'': 'src'}
)

#!/usr/bin/env python

from setuptools import setup
from setuptools import find_packages

setup(
    name='kerassurgeon',
    version="0.0.1",
    url='https://github.com/BenWhetton/keras-surgeon',
    license='MIT',
    description='A library for performing network surgery on trained Keras models.'
                'Useful for deep neural network pruning.',
    author='Ben Whetton',
    author_email='Ben.Whetton@gmail.com',
    install_requires=['keras'],
    extras_require={'pd': ['pandas'], },
    tests_require=['pytest'],
    packages=find_packages('src'),
    package_dir={'': 'src'}
)

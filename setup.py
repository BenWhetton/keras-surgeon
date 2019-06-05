#!/usr/bin/env python

from setuptools import setup
from setuptools import find_packages


setup(
    name='tfkerassurgeon',
    version="0.2.0",
    url='https://github.com/Raukk/tf-keras-surgeon',
    license='MIT',
    description='A library for performing network surgery on trained tf.Keras '
                'models. Useful for deep neural network pruning.',
    author='Ben Whetton',
    author_email='',
    python_requires='>=3',
    extras_require={'examples': ['pandas'], },
    tests_require=['pytest'],
    packages=find_packages('src'),
    package_dir={'': 'src'}
)


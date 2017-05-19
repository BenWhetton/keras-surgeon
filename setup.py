#!/usr/bin/env python

from setuptools import setup
from setuptools import find_packages

setup(
    name='kerasprune',
    version="0",
    url='',
    license='MIT',
    description='A library for pruning channels/neurons from trained Keras models '
                'while preserving the trained weights.',
    author='Ben Whetton',
    author_email='Ben.Whetton@gmail.com',
    install_requires=['numpy',
                      # 'tensorflow', # uncomment this and comment out tensorflow-gpu if not using gpu
                      'tensorflow-gpu',
                      'keras',

                      'pandas'],
    tests_require=['pytest'],
    packages=find_packages("src"),
    package_dir={'': 'src'}
)
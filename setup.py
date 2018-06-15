from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from setuptools import setup

setup(
    name='tf1',
    version='1.0.1',
    description='F1-score metric for TensorFlow',
    url='https://github.com/shkarupa-alex/tf1',
    author='Shkarupa Alex',
    author_email='shkarupa.alex@gmail.com',
    license='MIT',
    packages=['tf1'],
    install_requires=[
        'tensorflow>=1.5.0',
    ],
    test_suite='nose.collector',
    tests_require=['nose']
)

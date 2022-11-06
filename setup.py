## setup.py file for roicat

from distutils.core import setup

setup(
    name='roicat',
    version='0.1.0',
    author='Richard Hakim',
    keywords=['neuroscience', 'neuroimaging', 'machine learning', 'deep learning'],
    packages=[
        'roicat',
        'roicat.tracking',
        'roicat.classification'
    ],
    license='LICENSE',
    description='A library for classifying and tracking ROIs.',
    long_description=open('README.md').read(),
    # install_requires=open('requirements.txt').read().splitlines(),
    url='https://github.com/RichieHakim/ROICaT',
)
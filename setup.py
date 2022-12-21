## setup.py file for roicat

from distutils.core import setup

## Get install dependencies from requirements.txt
pip_deps = []
with open('requirements.txt', 'r') as f:
    for line in f.readlines():
        pip_deps.append(line.strip())

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
    install_requires=pip_deps,
    url='https://github.com/RichieHakim/ROICaT',
)
from setuptools import setup, find_packages

setup(
    name='pyretrodesign',
    version='0.0.1',
    author='Dylan M. Nielson',
    author_email='dylan.nielson@gmail.com',
    description='Translation of r-package retrodesign to python',
    url='https://github.com/nih-fmrif/pyretrodesign',
    packages=find_packages(),
    install_requires=[
        'pandas',
        'numpy',
        'scipy',
        'seaborn',
        'matplotlib',
        'jupytext'
    ],
    classifiers = [
        "Programming Language :: Python :: 3",
        "Operating System :: POSIX",
    ]
)

AstroBkgInterp: An Astronomical Background Estimation Tool
----------------------------------------------------------
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17410170.svg)](https://doi.org/10.5281/zenodo.17410170)


AstroBkgInterp is a Python tool designed to provide robust, flexible background
subtraction for sources observed on top of complex regions. It is a highly 
parameterizable Python package that works for both single images and 
3-dimensional data cubes, such as those produced by the JWST/MIRI Medium 
Resolution Spectrometer (MRS).

Installation
------------

The following will create a new Conda environment with the required Python 
packages:

```
conda env create --file environment.yml
conda activate abi_env
```

Finally install the package using:
```
pip install -e .
```

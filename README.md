AstroBkgInterp: An Astronomical Background Estimation Tool
----------------------------------------------------------

AstroBkgInterp is a Python tool designed for robust background subtraction 
for sources observed ontop of complex regions. It is a highly 
parameterizable python package that works for both single images and 
3-dimensional data cubes (e.g. MIRI Medium Resolution Spectrometer (MRS) data).

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
pip install =e .
```
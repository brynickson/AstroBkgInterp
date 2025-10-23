AstroBkgInterp: An Astronomical Background Estimation Tool
----------------------------------------------------------
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17410170.svg)](https://doi.org/10.5281/zenodo.17410170)


AstroBkgInterp is a Python tool designed to provide robust, flexible background
subtraction for sources observed on top of complex regions. It implements a 
flexible source-masking and polynomial interpolation routines to model 
spatially and spectrally varying backgrounds without relying on separate 
sky exposures. The tool supports both 2D images and 3D data cubes (e.g., 
integral field spectroscopy), offering wavelength-dependant masking and 
background modelling that account for PSF variations across spectral slices. 

AstroBkgInterp is particularly effective in scenarios where traditional 
background subtraction methods struggle, such as extended sources embedded 
in structured backgrounds, IFU data with spatial-spectral variability, and 
imaging data affected by instrumental gradients or artifacts.


Installation
------------

1. Clone the `AstroBkgInterp` repository:

```
git clone https://github.com/brynickson/AstroBkgInterp/
cd AstroBkgInterp
```

2. Create a new Conda environment with the required Python packages:

```
conda env create --file environment.yml
conda activate abi_env
```

3. Install the package using:

```
pip install -e .
```

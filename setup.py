from setuptools import setup, find_packages
import pathlib

here = pathlib.Path(__file__).parent.resolve()

# Get the long description from the README file
long_description = (here / 'README.md').read_text(encoding='utf-8')

setup(name='AstroBkgInterp',
      version='1.0',
      description='Astronomical Background Interpolation Tool',
      long_description=long_description,
      long_description_content_type='text/markdown',
#      license="",
      url='https://github.com/brynickson/astrobkginterp',
      author='B. Nickson',
      author_email='bnickson@stsci.edu',
      project_urls={
            'Documentation': 'https://github.com/brynickson/astrobkginterp/docs',
            'Source': 'https://github.com/brynickson/astrobkginterp',
            'Tracker': 'https://github.com/brynickson/astrobkginterp/issues'
      },
      packages=find_packages(), install_requires=['astropy', 'matplotlib', 'photutils'])
from setuptools import setup, find_packages

"""
From project root run
pip install -e .
(don't forget the dot)

This will install core as a package, as well as its sub-packages. 
-e means "editable", so you can edit the code in core and it will be reflected in the package.
"""

setup(
    name="citeline",
    version="0.1",
    packages=find_packages(),
)

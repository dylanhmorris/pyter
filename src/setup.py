from setuptools import setup

with open("README.md", 'r') as f:
    long_description = f.read()

setup(
   name='pyter',
   version='.1',
   description='',
   license="BSD 3-clause",
   long_description=long_description,
   author='Dylan H. Morris',
   author_email='dylan@dylanhmorris.com',
   url="https://github.com/dylanhmorris/pyter",
   packages=['pyter'],  # same as name
)

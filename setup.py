from setuptools import setup, find_packages
from os.path import join, dirname
import estraine


setup(
    name='EStraine',
    version=estraine.__version__,
    packages=find_packages(),
    long_description=open(join(dirname(__file__), 'README.txt')).read(),
    author_email='sevostyanikhin02@gmail.com',
    license="GPL"
)

from io import open

from setuptools import find_packages, setup

with open('README.md', 'r', encoding='utf-8') as f:
    readme = f.read()

REQUIRES = ["opencv-python==4.2"]

kwargs = {
    'name': 'dpfk',
    'version': '0.0.1',
    'description': '',
    'long_description': readme,
    'author': 'xvr.hlt',
    'author_email': 'xvr.hlt@gmail.com',
    'maintainer': 'xvr.hlt',
    'maintainer_email': 'xvr.hlt@gmail.com',
    'license': 'MIT/Apache-2.0',
    'install_requires': REQUIRES,
    'packages': find_packages(exclude=('tests', 'tests.*')),
}

setup(**kwargs)

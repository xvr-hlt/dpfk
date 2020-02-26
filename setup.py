from io import open

from setuptools import find_packages, setup

with open('dpfk/__init__.py', 'r') as f:
    for line in f:
        if line.startswith('__version__'):
            version = line.strip().split('=')[1].strip(' \'"')
            break
    else:
        version = '0.0.1'

with open('README.rst', 'r', encoding='utf-8') as f:
    readme = f.read()

REQUIRES = []

kwargs = {
    'name': 'dpfk',
    'version': version,
    'description': '',
    'long_description': readme,
    'author': 'xvr.hlt',
    'author_email': 'xvr.hlt@gmail.com',
    'maintainer': 'xvr.hlt',
    'maintainer_email': 'xvr.hlt@gmail.com',
    'url': 'https://github.com/_/dpfk',
    'license': 'MIT/Apache-2.0',
    'classifiers': [
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'License :: OSI Approved :: Apache Software License',
        'Natural Language :: English',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: Implementation :: CPython',
        'Programming Language :: Python :: Implementation :: PyPy',
    ],
    'install_requires': REQUIRES,
    'tests_require': ['coverage', 'pytest'],
    'packages': find_packages(exclude=('tests', 'tests.*')),
}

setup(**kwargs)

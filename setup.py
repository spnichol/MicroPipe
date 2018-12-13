

from setuptools import setup, find_packages
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

with open(path.join(here, 'README.rst'), encoding='utf-8') as f:
    long_description = f.read()

with open(path.join(here, 'requirements.txt')) as f:
    reqs = []
    deps = []
    for line in filter(bool, map(str.strip, f)):
        if line.startswith('git+'):
            deps.append(line)
        else:
            reqs.append(line)

setup(
    name='MicroPipe',
    version='0.0.1',

    description='Importable Python tasks',
    packages=find_packages(),
    long_description=long_description,
    url='https://github.com/prattle-analytics/PrattleNER',
    author='Prattle Analytics ',
    classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 3 - Alpha',

        # Indicate who your project is intended for
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Image Recognition',

        # Pick your license as you wish (should match "license" above)
        'License :: OSI Approved :: MIT License',

        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        'Programming Language :: Python :: 2.7',
    ],

    author_email='support@prattle.co',
    keywords='uhm no',
    license='private',
    install_requires=reqs,
    dependency_links=deps
)

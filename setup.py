from setuptools import setup, find_packages

with open('requirements.txt') as r:
    requirements = r.read().splitlines()

setup(
    name='mandelbrot',
    version='0.0.1',
    packages=find_packages(),
    include_package_data=True,
    install_requires=requirements,
    entry_points={
        'console_scripts': [
            # TODO: Define entry points
        ],
    },
)

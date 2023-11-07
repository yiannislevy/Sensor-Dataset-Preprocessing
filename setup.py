from setuptools import setup, find_packages

# Read requirements.txt and use its contents for the install_requires parameter
with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name='SensorDatasetPreprocessing',
    version='1.0.0',
    author='Ioannis Levi',
    author_email='ioanlevi@ece.auth.gr',
    description='A package for preprocessing sensor datasets (accelerometer, gyroscope, weight)',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=requirements,
)

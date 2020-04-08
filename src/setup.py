from setuptools import setup, find_packages

setup(
    name="TensorQQ",
    version="0.1.2",
    description="A NumPy-Based Deep Learning Framework",
    author="qqaatw",
    license="MIT",

    packages=find_packages(exclude=['tests', 'tests.*', '.vscode', '__pycache__']),
    install_requires=['numpy>1.16.0']
)

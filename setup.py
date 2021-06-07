from setuptools import setup, find_packages

def requirements():
    """Read requirements from requirements.txt"""
    with open('requirements.txt', 'r') as f:
        return f.readlines()

def readme():
    with open('README.md') as f:
        return f.read()

setup(
    name="place_informative",
    version="0.1.0",
    description="place informative using deep learning",
    url="git@github.com:ahmadrezaomidvar/place_informative.git",
    author="Ahmadreza Omidvar",
    author_email="ahmadreza.omidvar@gmail.com",
    license="None",
    install_requires=requirements(),
    zip_safe=False,
)
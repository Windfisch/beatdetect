from setuptools import setup

with open('requirements.txt') as f:
    dependencies = [ l for l in f.readlines() if (len(l)>1 and not l.startswith("#"))]

setup(
    name='beatdetect',
    version='0.0.1',
    description='Performs beat detection on music',
    author='windfisch',
    author_email='',
    install_requires=dependencies,
)

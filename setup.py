from setuptools import setup, find_packages

base_packages = ['altair','polars','narwhals']

setup(
    name='sootopolis',
    version='0.1.0',
    author='Sam Worley',
    description='helper functions for turning linear regression predictions into waterfall charts for explainability',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='http://github.com/sworley1/sootopolis',
    packages=find_packages(),
    install_requires=base_packages,
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
    license_files = ['LICENSE'],
)
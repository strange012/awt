from setuptools import setup, find_packages

setup(
    name='awt',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'sqlalchemy',
        'pandas',
        'sklearn',
        'numpy',
        'scipy',
        'matplotlib',
        'statsmodels'
    ],
)

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()


install_requires=[
    "fanalysis",
    "numpy",
    "pandas",
    "statsmodels"
]

setup(
    name='RFPC',
    version='0.0.2',
    url='https://github.com/titigmr/RFPC.git',
    license='mit',
    author='Thierry Gameiro',
    description='Approche RFPC (modèle à équations structurelles)',
    packages=find_packages(),
    include_package_data=True,
    platforms='any',
    long_description=long_description,
    long_description_content_type="text/markdown",
    install_requires=install_requires,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6"
)
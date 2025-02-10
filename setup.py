from setuptools import setup, find_packages

setup(
    name="cpDistiller",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
      
    ],
    entry_points={
        'console_scripts': [
            # "cpdistiller = cpDistiller.cli:main",
        ],
    },
    include_package_data=True,  
    zip_safe=False
)

from setuptools import setup, find_packages

setup(
    name='mousecraft',
    version='0.1.0',
    description='GUI for validating motion classifications',
    author='Sofia Zangila & Maxime Reygner',
    author_email='szaggila@hotmail.com'
    packages=find_packages(),
    install_requires=[
        'PyQt5',
        'numpy',
        'pandas',
        'matplotlib',
        'opencv-python',
    ],
    entry_points={
        'console_scripts': [
            'mousecraft = mousecraft.gui:main',
        ],
    },
    include_package_data=True,
)
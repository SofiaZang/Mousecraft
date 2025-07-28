from setuptools import setup, find_packages

setup(
    name='mousecraft',
    version='0.1.0',
    description='Fun GUI for validating motion classifications',
    author='Sofia Zangila & Maxime Reygnier',
    author_email ="szaggila@hotmail.com",
    url="https://github.com/SofiaZang/Mousecraft",
    license="MIT",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
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
    package_data= {
        "mousecraft": ["*.png", "*.webp", "*.gif"],
    },
)

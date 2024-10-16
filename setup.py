from setuptools import setup, find_packages

setup(
    name='SurfTrack',
    version='1.0.0',
    packages=find_packages(),
    author='Musanna Galib, Matteo Ferraresso',
    author_email='galibubc@student.ubc.ca, matfe@mail.ubc.ca',
    description='Package for tracking and analyzing in-situ optical microscopy images',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/MusannaGalib/SurfTrack.git',
    license='MIT',
    install_requires=[
        # Add any Python dependencies required by your package
    ],
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
)

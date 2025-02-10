<!-- [![PyPI downloads](https://img.shields.io/pypi/dm/MOOSEanalyze)](https://pypi.org/project/MOOSEanalyze/) -->
<!--[![Paper](https://img.shields.io/badge/ACS_Energy_Lett-blue)](https://doi.org/your-paper-doi) -->
[![Python 3.6+](https://img.shields.io/badge/python-3.6+-red.svg)](https://www.python.org/downloads/)
[![Release](https://img.shields.io/badge/release-v0.0.1-brightgreen)](https://github.com/MusannaGalib/SurfTrack)
[![License: MIT](https://img.shields.io/badge/license-MIT_2.0-yellow)](https://opensource.org/licenses/MIT)


# SurfTrack

This repository contains scripts for tracking and analyzing versatile images from in-situ optical microscopy (OM) experiment or Blood Clotting experiments.

## Overview

The project aims to develop a package for tracking and analyzing any images from a movie. It involves two main MATLAB scripts: `trracking_master_code.m` and `Tracked_surface_compare.m`.

## Requirements

- MATLAB R2022a or later
- Python 3.x

## Setup

### Installation
1. **Download the Package:**
   - Download the zip file containing the SurfTrack package.
   - Extract the zip file to a directory of your choice.

2. **Install the Package:**
   - Open a command prompt or terminal.
   - Navigate to the directory where you extracted the package.
   - Install the package by running the command:
```bash
git clone https://github.com/MusannaGalib/SurfTrack.git
cd SurfTrack
pip install .
```
   This command installs the package along with its dependencies.

   SurfTrack can also be installed from PyPI:
```bash
pip install SurfTrack
```

### Using the Package

To use this package, give your matlab executable path in ```run.py```. Then just copy your movie.mp4 file in the scirpts folder and run the following command 

```python
process = subprocess.Popen(['C:/Program Files/MATLAB/R2022a/bin/matlab', '-nosplash', '-nodesktop', '-r', f"run('{script_path}');exit;"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
```


```python

python3 run.py

```

If you want to change the number of images that needed to be generated from the movie change the following file in the ```run.py``` file:
```python

# Define how many images you want from the video
npics = 3
```

### Example Usage

An example movie is given in 'scripts' folder. You can use the following commands to play with that.

```python
# execute the python wrapper
python3 run.py
```
## Authors
This Software is developed by Matteo Ferraresso & Musanna Galib


## Citing This Work
If you use this software in your research, please cite the following paper:

```python
BibTeX entry:
@misc{galib2025dendritesuppressionznbatteries,
      title={Dendrite Suppression in Zn Batteries Through Hetero-Epitaxial Residual Stresses Shield}, 
      author={Musanna Galib and Amardeep Amardeep and Jian Liu and Mauricio Ponga},
      year={2025},
      eprint={2502.03841},
      archivePrefix={arXiv},
      primaryClass={cond-mat.mtrl-sci},
      url={https://arxiv.org/abs/2502.03841}, 
}
```

### Contact, questions, and contributing
If you have questions, please don't hesitate to reach out to galibubc[at]student[dot]ubc[dot]ca and matfe[at]mail[dot]ubc[dot]ca

If you find a bug or have a proposal for a feature, please post it in the Issues. If you have a question, topic, or issue that isn't obviously one of those, try our GitHub Disucssions.

If your post is related to the framework/package, please post in the issues/discussion on that repository. 

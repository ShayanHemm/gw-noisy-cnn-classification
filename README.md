## GR vs BGR gravitational wave classification (1D CNN)

This repository contains a first working pipeline to classify simulated gravitational wave signals in the frequency domain as GR vs beyond GR (BGR), using a 1D convolutional neural network.

### What is inside
- A Jupyter notebook that generates or loads waveforms, adds noise, and builds training and test sets
- A 1D CNN baseline model and evaluation (metrics and plots)
- Utilities for preprocessing and dataset handling (to be cleaned up and modularized)

### Status
Initial structure and baseline results. I am cleaning the code and expanding the documentation.

### How to run
Recommended: create the conda environment and run the notebook.

```bash
conda env create -f environment.yml
conda activate gw-ml
jupyter lab

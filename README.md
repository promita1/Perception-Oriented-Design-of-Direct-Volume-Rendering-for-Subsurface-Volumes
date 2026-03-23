# Perception-Oriented Design of Direct Volume Rendering for Subsurface Volumes

This project explores how direct volume rendering (DVR) design choices affect the interpretability of subsurface 3D scalar data. The implementation focuses on controlled comparisons of transfer functions, sampling density, and shading, along with an interactive prototype for exploratory visualization.

## Team Members
- Promita Panja
- Shuping Tan
- Azmal Awasaf

## Project Overview
The project uses a public 3D seismic velocity dataset and builds a Python + VTK pipeline for:
- baseline direct volume rendering
- controlled generation of 27 render conditions
- atlas creation for side-by-side comparison
- interactive switching between visualization presets

## Main Files
- `main_baseline.py` — baseline preprocessing and first DVR render
- `dvr_experiment.py` — batch generation of 27 rendering conditions and atlas
- `dvr_interactive.py` — interactive prototype for switching transfer function, sampling, and shading

## Dataset
- This project uses the USGS Cascadia v1.7 seismic velocity model. 
- Dataset name: Data for A 3-D Seismic Velocity Model for Cascadia with Shallow Soils & Topography, Version 1.7
- Dataset link - https://www.sciencebase.gov/catalog/item/65b40c6ad34e36a390458d76
- Prototype is built using CVM17_L3.nc
- The raw dataset is not included in this repository because of file size.

## Outputs
- baseline render image
- 27-condition comparative rendering set
- atlas image
- interactive screenshots

## Installation
Install dependencies with:

```bash
pip install -r requirements.txt

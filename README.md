# NanoBeamMap

nanobeammap is a Python package for processing, segmenting, and analyzing 4D Scanning Transmission Electron Microscopy (4D-STEM) data, with a focus on nanoparticle analysis in megalibrary contexts. It leverages reciprocal space information to create detailed real-space maps of nanoparticle structures.

## Features

- Efficient preprocessing of large 4D-STEM datasets
- Dimensionality reduction of diffraction patterns using Incremental PCA and UMAP
- Multiple segmentation methods:
  - Gaussian Mixture Model (GMM)
  - K-means clustering
  - DBSCAN
  - Non-negative Matrix Factorization (NMF)
- Reconstruction and visualization of real-space nanoparticle maps
- Flexible visualization tools for both reciprocal and real space results, including linear, sqrt, and log scale options

## Installation

1. Clone this repository:

```bash
git clone https://github.com/yourusername/nanobeammap.git
cd nanobeammap
```

2. Install the required packages:

```bash
pip install -r requirements.txt
```

## Usage

See `example_usage.ipynb` for a detailed example of how to use NanoBeamMap.

Basic usage:

```python
from nanobeammap import NanoBeamMap

# Initialize the mapping tool
mapper = NanoBeamMap('path/to/your/4DSTEM/file.dm4')

# Preprocess the data
mapper.preprocess(chunk_size=1000)

# Reduce dimensionality of diffraction patterns
mapper.reduce_dimensionality(n_components_pca=50, chunk_size=1000)

# Segment diffraction patterns
mapper.segment_data(method='gmm', n_clusters=5)

# Visualize results
mapper.visualize_spatial_distribution()
mapper.visualize_average_patterns(scale='log')
```


## Requirements
See `requirements.txt` for a list of required Python packages.

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments
This work is partially funded by the International Institute for Nanotechnology at Northwestern University in the Project MINDSHIFT: Multimodal Integration For Nanoparticle Data Screening Using Highthroughput Frameworks In Transmission Electron Microscopy.
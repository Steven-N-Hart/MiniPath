# MiniPath README

## Overview

**MiniPath** is a Python-based tool designed for processing and analyzing digital pathology images stored in DICOM format, particularly from whole slide images (WSIs). The primary focus of MiniPath is to extract and rank diverse image patches based on entropy and cluster analysis. The tool leverages several machine learning techniques, including Principal Component Analysis (PCA) and KMeans clustering, to identify representative patches that can be used for further analysis at higher magnification levels.

MiniPath includes various utilities for reading DICOM files from local storage or Google Cloud Storage (GCS), calculating image entropy, and selecting representative patches for downstream processing.

## Key Features

- **DICOM Image Handling**: Supports reading DICOM images from local paths, GCS, and DICOMweb.
- **Entropy Calculation**: Uses entropy as a feature for image patch diversity ranking.
- **PCA and Clustering**: Applies PCA to reduce feature dimensionality and KMeans clustering to group similar patches.
- **Patch Ranking**: Ranks patches for diversity and selects representative patches.
- **High-Resolution Image Extraction**: Extracts relevant high-magnification frames corresponding to selected low-magnification patches.


## Installation

To install the necessary dependencies, you can use the provided `requirements.txt` file:

```bash
pip install -r requirements.txt
```

## Environment Setup
This tool relies on environment variables to connect to Google Cloud services. Ensure that you have a .env file in the root directory with the following contents:
```env
GOOGLE_APPLICATION_CREDENTIALS=path/to/your/credentials.json
```
Replace `path/to/your/credentials.json` with the actual path to your Google Cloud credentials file.

## Usage
### Initialization
```python
from minipath import MiniPath
minipath = MiniPath(csv='path/to/csv_file.csv', subset=True)
```
- **`csv`**: Path to a CSV file containing metadata and GCS URLs for high-magnification DICOM images.
- **`subset`**: Boolean flag to decide if only a subset of diverse patches should be used. Defaults to True. If you 
  set it to false, all patches will be extracted.

### Get Representative Patches
```python
minipath.get_representatives(full_url='https://path.to.dicom.web/resource')
```
- **`full_url`**: The URL pointing to the low-magnification DICOM image.
This method extracts image patches from the provided DICOM file, computes entropy for each patch, applies PCA, and 
  clusters the patches to select the most representative ones.


### Extract High-Resolution Frames
```python
high_res_frames = minipath.get_high_res()
```
- `high_res_patches` will be an array of dictionaries with the following keys:
  * 'row_min': Pixel coordinate of first row
  * 'row_max': Pixel coordinate of last row
  * 'col_min': Pixel coordinate of first col
  * 'col_max': Pixel coordinate of last col
  * 'frame':   The dicom frame that represents this coordinate set
  * 'img_array': a numpy array of the image values

This method extracts high-resolution frames corresponding to the representative patches identified at low magnification.
You can loop through this array for running a model.
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
#### Required Parameter
- **`csv`**: Path to a CSV file containing metadata and GCS URLs for high-magnification DICOM images. Requires the 
  following columns:
  - **gcs_url**: path to local ('path/to/file') or remote ('gs://') DICOM file or DICOMweb address ('https://')
  - **SeriesInstanceUID**: Necessary to link together different resolutions of DICOM images
  - **row_num_asc**: should have a 1 in this column if referring to the low magnification DICOM
  - **row_num_desc**: should have a 1 in this column if referring to the high magnification DICOM
  
#### Optional Parameters
- **`subset: bool = True`**: 
  - Boolean flag to decide if only a subset of diverse patches should be used. Defaults to True. If you 
  set it to false, all patches will be extracted.

- **`explained_variance: float = 0.8`**
  - Threshold for cumulative explained variance in Principal Component Analysis (PCA).
  - Determines how many principal components are kept based on the amount of variance they explain. A value of 0.8 means the components should explain at least 80% of the variance in the data.

- **`img_size: int = 256`**
  - The size to which the low-resolution image is resized for patch extraction.
  - This option affects the resolution of the image before breaking it down into patches.

- **`patch_size: int = 8`**
  - The size of each patch to be extracted from the resized image.
  - This option controls the granularity of the patches, where smaller sizes yield more patches per image.

- **`min_k: int = 8`**
  - The minimum number of clusters to be used in KMeans clustering.
  - Ensures that at least a certain number of clusters are considered when determining the diversity of patches.
    
- **`max_k: int = 50`**
  - The maximum number of clusters to be tested during KMeans clustering.
  - This option controls how many potential clusters will be used when identifying diverse patches.

- **`km_init: str = 'k-means++'`**
  - The initialization method used for KMeans clustering.
  - The default value `'k-means++'` ensures better initialization for faster convergence and more reliable clustering results.

- **`km_max_iter: int = 300`**
  - Maximum number of iterations allowed for the KMeans algorithm.
  - This setting ensures that KMeans stops after 300 iterations even if it hasn't converged to an optimal solution, preventing excessive computation.

- **`km_n_init: int = 10`**
  - The number of times the KMeans algorithm will be run with different centroid seeds.
  - KMeans clustering is run multiple times with different initializations, and the best result (in terms of inertia) is kept. A higher value increases reliability at the cost of more computation.

- **`num_high_res_frames = 500`**
  - Maximum number of frames per slide you want to extract
  - There is an image content filter, so blanks will be filtered out and you may get fewer than this.
  - Setting this value to `None` will give you a full complement of the high resolution images in the coordinates of 
    the low resolution set. 


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

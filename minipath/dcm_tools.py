import pydicom
from google.cloud import storage
import numpy as np
import pandas as pd
from google.auth.transport.requests import AuthorizedSession
import google.auth
import io

def get_single_dcm_img(dcm_input) -> np.ndarray:
    """
    Generate a grid image from a multi-frame DICOM object.

    :param dcm_input: DICOM object containing multiple frames.
    :return: Numpy array representing the concatenated grid image.
    """
    dcm, total_pixel_matrix_columns, total_pixel_matrix_rows, columns, rows, grid_rows, grid_cols = parse_dcm_info(
        dcm_input)

    frames = dcm.pixel_array
    #img = Image.fromarray(frames)
    #img.save('test/dcm.png')
    return frames


def parse_dcm_info(dcm_input):
    dcm = read_dicom(dcm_input)
    # Extract necessary metadata
    total_pixel_matrix_columns = dcm.TotalPixelMatrixColumns
    total_pixel_matrix_rows = dcm.TotalPixelMatrixRows
    columns = dcm.Columns
    rows = dcm.Rows

    # Calculate grid size
    grid_cols = int(np.ceil(total_pixel_matrix_columns / columns))
    grid_rows = int(np.ceil(total_pixel_matrix_rows / rows))
    return dcm, total_pixel_matrix_columns, total_pixel_matrix_rows, columns, rows, grid_rows, grid_cols


def read_dicom(dcm_input):
    """
    Load a DICOM file from a local path or Google Cloud Storage.

    :param dcm_input: Local file path or GCS path.
    :return: pydicom FileDataset object.
    """
    if isinstance(dcm_input, pydicom.dataset.FileDataset):
        return dcm_input
    if isinstance(dcm_input, pd.Series):
        dcm_input = dcm_input.values[0]
    if isinstance(dcm_input, str):
        if dcm_input.startswith('gs://'):
            # Read DICOM from GCS
            return read_dicom_from_gcs(dcm_input)
        elif dcm_input.startswith('https://'):
            return read_dicomweb(dcm_input)
        else:
            # Read local DICOM file
            return pydicom.dcmread(dcm_input)
    raise f"Could not complete with {dcm_input}"


def read_dicomweb(dcm_input):
    credentials, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
    # Creates a requests Session object with the credentials.
    session = AuthorizedSession(credentials)

    headers = {"Accept": "application/dicom; transfer-syntax=*"}
    response = session.get(dcm_input, headers=headers)
    response.raise_for_status()
    return pydicom.dcmread(io.BytesIO(response.content))


def read_dicom_from_gcs(gcs_path):
    """
    Read a DICOM file from Google Cloud Storage.

    :param gcs_path: GCS path to the DICOM file.
    :return: pydicom FileDataset object.
    """
    # Split the GCS path to get bucket and file name
    path_parts = gcs_path.replace('gs://', '').split('/')
    bucket_name = path_parts[0]
    file_name = '/'.join(path_parts[1:])

    # Initialize a client and get the bucket
    try:
        client = storage.Client()
        bucket = client.get_bucket(bucket_name)
        blob = bucket.blob(file_name)
    except:
        blob = download_public_file(bucket_name, file_name, gcs_path, local=False)
    # Download the file as a bytes object
    dicom_bytes = blob.download_as_bytes()

    # Use pydicom to read the DICOM file from bytes
    dicom_file = pydicom.dcmread(io.BytesIO(dicom_bytes))

    return dicom_file

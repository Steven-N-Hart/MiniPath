from pydicom.encaps import generate_pixel_data_frame
from dotenv import load_dotenv
from tqdm import tqdm
from multiprocessing import Pool, shared_memory
import random

load_dotenv()
from .dcm_tools import *
from sklearn.cluster import KMeans
from skimage.measure import shannon_entropy
from skimage.util import view_as_blocks
import logging
import io
import numpy as np
import pandas as pd
from PIL import Image
from typing import List, Optional
import cv2
#from .debug_tools import *

from multiprocessing import Pool, shared_memory
from functools import partial

# Define this helper function at the module level.
def process_high_mag_mapping(mapping, scaling_factor, frames_to_keep, clean_high_mag_frames):
    """
    Process a single high-mag mapping to generate high-resolution patches.

    Args:
        mapping (dict): A single mapping from mag_pairs.high_mag_mappings.
        scaling_factor (float): Scaling factor to convert low-mag to high-mag coordinates.
        frames_to_keep (list): List of frame ids that are valid.
        clean_high_mag_frames (list): List of dictionaries containing high-res frame data.

    Returns:
        list: A list of dictionaries for high-res patches.
    """
    pixel_range = mapping['high_pixel_range']
    patches = []
    for frame, (row, col) in zip(mapping['frame_numbers'], mapping['row_col']):
        if frame in frames_to_keep:
            # Find the corresponding high-res image array
            img_array = next((f['img_arr'] for f in clean_high_mag_frames if f['frame_id'] == frame), None)
            patches.append({
                'row_min': int(pixel_range['y_min'] * scaling_factor),
                'row_max': int(pixel_range['y_max'] * scaling_factor),
                'col_min': int(pixel_range['x_min'] * scaling_factor),
                'col_max': int(pixel_range['x_max'] * scaling_factor),
                'frame': frame,  # DICOM frame number
                'row_col': (row, col),  # High-mag grid position
                'img_array': img_array  # Corresponding high-res image array
            })
    return patches

class MiniPath:
    def __init__(self, csv: Optional[str] = None, subset: bool = True, patch_per_cluster: int = 1, max_k: int = 50,
                 img_size: int = 256, patch_size: int = 8, min_k: int = 8,
                 km_init: str = 'k-means++', km_max_iter: int = 300, km_n_init: int = 10,
                 num_high_res_frames=500,
                 processors: int = 4):
        """
        Initializes the MiniPath class with parameters for processing images and clustering.

        Args:
            csv (Optional[str]): Path to a CSV file containing metadata for the images.
            subset (bool): Whether to use a subset of the dataset.
            patch_per_cluster (int): Number of patches to sample per cluster.
            max_k (int): Maximum number of clusters.
            img_size (int): The size of the image.
            patch_size (int): The size of each patch.
            min_k (int): Minimum number of clusters.
            km_init (str): Initialization method for KMeans clustering.
            km_max_iter (int): Maximum number of iterations for KMeans.
            km_n_init (int): Number of KMeans initializations to run.
            num_high_res_frames (int): Maximum number of frames to extract. If None, extract all frames.
            processors (int): Maximum number of processors to use
        """
        self.csv = pd.read_csv(csv) if csv else None
        self.subset = subset
        self.max_k = max_k
        self.min_k = min_k
        self.img_size = img_size
        self.patch_size = patch_size
        self.patch_per_cluster = patch_per_cluster

        # KMeans parameters
        self.km_init = km_init
        self.km_max_iter = km_max_iter
        self.km_n_init = km_n_init

        # Variables to store image data
        self.img_to_use_at_low_mag: Optional[List[Image.Image]] = None
        self.low_res_dcm = None
        self.high_mag_dcm = None
        self.num_high_res_frames = num_high_res_frames

        self.processors = processors

    def get_representatives(self, full_url: str) -> None:
        """
        Fetch the DICOM image from the provided URL, process it to get representative patches,
        and store them for further use.

        Args:
            full_url (str): URL to fetch the DICOM image.
        """
        # Read DICOM image from the web
        dcm = read_dicomweb(full_url)

        # Get image array from the DICOM data
        image = get_single_dcm_img(dcm)

        # Use the ImageEntropySampler to process the image and get representative patches
        sampler = ImageEntropySampler(image, patch_size=(self.patch_size, self.patch_size),
                                      top_n=self.max_k, patch_per_cluster=self.patch_per_cluster)
        selected_patches = sampler.process()

        # Store the results
        self.img_to_use_at_low_mag = selected_patches
        self.low_res_dcm = dcm
        logging.info(f"Processed low-resolution DICOM and selected {len(selected_patches)} representative patches.")

    def get_high_res(self):
        """
        Retrieve high-resolution images for each representative patch, including both low- and high-mag coordinates.

        Args:
            high_res_frames (list): Pre-sampled high-resolution frames.

        Returns:
            List[dict]: A list of dictionaries, each containing:
                        - 'row_min', 'row_max', 'col_min', 'col_max': Low-mag pixel coordinates of the patch.
                        - 'high_row_min', 'high_row_max', 'high_col_min', 'high_col_max': High-mag pixel coordinates.
                        - 'frame': Frame index from the DICOM file.
                        - 'row_col': Grid position of the frame in the high-mag image.
                        - 'img_array': Extracted image as a NumPy array.
        """
        if self.low_res_dcm is None or self.img_to_use_at_low_mag is None:
            raise ValueError(
                "Low-resolution DICOM or image patches not initialized. Call get_representatives() first.")

            # Create a MagPairs object to find high-resolution frames corresponding to low-resolution patches
        mag_pairs = MagPairs(
            self.low_res_dcm,
            img_to_use_at_low_mag=self.img_to_use_at_low_mag,
            bq_results_df=self.csv,
            num_high_res_frames=self.num_high_res_frames,
            patch_size=(self.patch_size, self.patch_size)
        )
        self.high_mag_dcm = mag_pairs.high_mag_dcm

        # Determine which frames to keep
        frames_to_keep = [x['frame_id'] for x in mag_pairs.high_mag_frames if
                          x is not None]
        scaling_factor = mag_pairs.scaling_factor

        # Use multiprocessing to process each mapping in parallel.
        with Pool(processes=self.processors) as pool:
            # Create a partial function with the additional parameters.
            func = partial(
                process_high_mag_mapping,
                scaling_factor=scaling_factor,
                frames_to_keep=frames_to_keep,
                clean_high_mag_frames=mag_pairs.clean_high_mag_frames
            )
            # Map the helper function over all mappings.
            results = pool.map(func, mag_pairs.high_mag_mappings)

        # Flatten the list of lists into a single list of high-resolution patches.
        high_res_patches = [patch for sublist in results for patch in sublist]

        logging.debug(
            f"Generated {len(high_res_patches)} high-resolution patches.")
        return high_res_patches

class ImageEntropySampler:
    def __init__(self, image: np.ndarray, patch_size: tuple[int, int] = (8, 8), top_n: int = 10,
                 patch_per_cluster: int = 1,
                 km_init: str = 'k-means++', km_max_iter: int = 300, km_n_init: int = 10):
        """
        Initialize the ImageEntropySampler class.

        Args:
            image (np.ndarray): The input image in RGB format.
            patch_size (tuple[int, int]): The size of the patches to extract.
            top_n (int): The number of top entropy patches to sample.
            patch_per_cluster (int): Number of patches to sample per cluster.
            km_init (str): Initialization method for KMeans clustering.
            km_max_iter (int): Maximum number of iterations for KMeans.
            km_n_init (int): Number of initializations for KMeans.
        """
        self.image = image
        self.patch_size = patch_size
        self.top_n = top_n
        self.patch_per_cluster = patch_per_cluster
        self.km_init = km_init
        self.km_max_iter = km_max_iter
        self.km_n_init = km_n_init

        self.image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        self.foreground = None
        self.entropy_values = []
        self.patch_coords = []
        self.patches = None

    def eliminate_background(self) -> np.ndarray:
        """
        Remove the background from the image using Otsu's thresholding.

        Returns:
            np.ndarray: Foreground image with background removed.
        """
        _, mask = cv2.threshold(self.image_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        foreground = cv2.bitwise_and(self.image, self.image, mask=mask.astype(np.uint8))
        #plot_fg(foreground)
        logging.debug("Background eliminated from the image.")
        return foreground

    @staticmethod
    def pad_image(image: np.ndarray, patch_size: tuple[int, int]) -> np.ndarray:
        """
        Pad the input image to ensure compatibility with the patch size.

        Args:
            image (np.ndarray): The input image to pad.
            patch_size (tuple[int, int]): The size of the patches to ensure divisibility.

        Returns:
            np.ndarray: The padded image.
        """
        pad_h = (patch_size[0] - (image.shape[0] % patch_size[0])) % patch_size[0]
        pad_w = (patch_size[1] - (image.shape[1] % patch_size[1])) % patch_size[1]
        padded_image = np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), mode='constant', constant_values=255)
        logging.debug(f"Image padded from {image.shape} to {padded_image.shape} to fit patch size: {patch_size}.")
        return padded_image

    def pad_to_size(self, pixel_array, target_shape=(256, 256, 3)) -> np.ndarray:
        """
        Pad the input array to the target shape, padding only to the right and bottom.

        Args:
            pixel_array (np.ndarray): Input array to pad.
            target_shape (tuple): Target shape for the array (height, width, channels).

        Returns:
            np.ndarray: Padded array.
        """
        # Calculate padding needed for height and width
        height_diff = target_shape[0] - pixel_array.shape[0]
        width_diff = target_shape[1] - pixel_array.shape[1]

        # Apply padding only to the right and bottom
        padded_array = np.pad(
            pixel_array,
            ((0, height_diff), (0, width_diff), (0, 0)),  # Pad bottom and right only
            mode='constant',  # Use constant padding
            constant_values=255  # Padding value
        )

        # Debug: Check dimensions of padded array
        logging.debug(f"Original shape: {pixel_array.shape}")
        logging.debug(f"Padded shape: {padded_array.shape}")

        # Save the result as an image with labeled patches
        #save_padded_array_with_labels(padded_array, patch_size=self.patch_size, output_path="test/output_image.png")

        return padded_array

    def process_patches(self):
        """Divide the image into patches and calculate entropy."""
        # Debug: Log dimensions before patching
        logging.debug(f"Foreground shape before patching: {self.foreground.shape}")
        logging.debug(f"Patch size: {self.patch_size}")

        self.patches = view_as_blocks(self.foreground, block_shape=(self.patch_size[0], self.patch_size[1], 3))

        # Debug: Check the grid dimensions
        num_rows, num_cols = self.patches.shape[:2]
        logging.debug(f"Patch grid dimensions: {num_rows} rows x {num_cols} cols")

        entropy_results = []
        coords_results = []

        # Flatten the loop for better efficiency
        for i, j in np.ndindex(self.patches.shape[:2]):
            patch = self.patches[i, j]  # Extract the patch directly
            # Skip if background
            if np.mean(patch) > 220:
                continue
            entropy = self.calculate_entropy(patch)
            entropy_results.append(entropy)
            coords_results.append((i, j))

        self.entropy_values = np.array(entropy_results)
        self.patch_coords = coords_results

        # Debug: Check calculated entropy values and patch coordinates
        logging.debug(f"Calculated entropy for {len(self.entropy_values)} patches.")

    def cluster_and_sample(self):
        """Cluster patches by entropy and sample representative patches."""
        # Debug: Log number of entropy values before clustering
        logging.debug(f"Number of patches before clustering: {len(self.entropy_values)}")

        if self.top_n >= len(self.entropy_values):
            return [(self.patches[i, j], (i, j)) for i, j in self.patch_coords]

        # Perform clustering
        kmeans = KMeans(n_clusters=self.top_n, init=self.km_init, max_iter=self.km_max_iter, n_init=self.km_n_init)
        cluster_labels = kmeans.fit_predict(self.entropy_values.reshape(-1, 1))

        logging.debug(f'Found {len(set(cluster_labels))} cluster labels')

        sampled_patches = []
        for cluster in range(self.top_n):
            cluster_indices = np.where(cluster_labels == cluster)[0]
            selected_indices = np.random.choice(cluster_indices, size=min(self.patch_per_cluster, len(cluster_indices)),
                                                replace=False)
            for idx in selected_indices:
                col, row = self.patch_coords[idx]
                patch = self.patches[col, row]  # Extract the patch directly
                # Ensure the patch has the correct shape
                if len(patch.shape) > 3:
                    patch = patch.squeeze()
                sampled_patches.append((Image.fromarray(patch), (row, col), idx))

        # Debug: Log sampled patches and their coordinates
        logging.debug(f"Sampled patches: {[(pos, idx) for _, pos, idx in sampled_patches]}")
        # save_padded_array_with_coords(self.foreground, patch_size=self.patch_size,
        # output_path="test/highlighted_image.png", patch_coords=self.patch_coords)

        """for img, pos, idx in sampled_patches:
            img.save(f'test/{pos[0]}_{pos[1]}.png')"""
        return sampled_patches

    def calculate_entropy(self, patch: np.ndarray) -> float:
        """
        Calculate the entropy of a grayscale patch.

        Args:
            patch (np.ndarray): The patch for which entropy is calculated.

        Returns:
            float: Entropy value of the patch.
        """
        if len(patch.shape) == 4 and patch.shape[0] == 1:  # Handle (1, H, W, C) case
            patch = patch[0]
        patch_gray = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY)
        return shannon_entropy(patch_gray)

    def process(self):
        """Run the entire processing pipeline."""
        # Eliminate background
        self.foreground = self.eliminate_background()

        # Debug: Log shape after background elimination
        logging.debug(f"Foreground shape after background elimination: {self.foreground.shape}")

        # Pad the image
        #self.foreground = self.pad_to_size(self.foreground, target_shape=(256, 256, 3))
        self.foreground = self.pad_image(self.foreground, patch_size=self.patch_size)

        # Process patches
        self.process_patches()

        # Cluster and sample patches
        return self.cluster_and_sample()


class MagPairs:
    def __init__(self, low_mag_dcm, img_to_use_at_low_mag=None, bq_results_df=None, num_high_res_frames=None,
                 patch_size=(256, 256)):
        """
        Initialize the MagPairs object to process DICOM images and extract patches at different magnifications.

        Args:
            low_mag_dcm: Low-magnification DICOM object or path.
            img_to_use_at_low_mag: List of image patches from low-magnification DICOM to map to high-magnification.
            bq_results_df: DataFrame containing metadata to pair DICOMs.
            num_high_res_frames: Maximum number of frames to extract. If None, extract all frames.
        """
        self.grid_cols = None
        self.low_mag_dcm = read_dicom(low_mag_dcm)
        self.high_mag_dcm = read_dicom(self.get_local_dcm_pair(low_mag_dcm, bq_results_df))
        self.pixel_spacing_at_low_mag = self.get_pixel_spacing(self.low_mag_dcm)
        self.pixel_spacing_at_high_mag = self.get_pixel_spacing(self.high_mag_dcm)
        self.scaling_factor = self.pixel_spacing_at_low_mag / self.pixel_spacing_at_high_mag
        self.fd = self.get_frame_dict(self.high_mag_dcm)

        self.minmax_list = self.get_minmax(img_to_use_at_low_mag)
        self.high_mag_mappings = self.find_high_mag_mappings()

        self.high_mag_frames = list(
            self.frame_extraction(self.high_mag_dcm, self.high_mag_mappings, num_high_res_frames))


        self.clean_high_mag_frames = [x for x in self.high_mag_frames if x is not None]
        logging.info(f'From {len(self.high_mag_frames)} frames, {len(self.clean_high_mag_frames)} had tissue')
        logging.debug("Completed initialization of MagPairs.")




    def find_high_mag_mappings(self):
        """
        Map low-magnification patches to high-magnification frames and pixel ranges.

        Returns:
            list: List of mappings for each low-mag patch.
        """
        high_res_mappings = []
        for idx, patch in enumerate(self.minmax_list):
            mapping = self.map_to_high_mag_with_frames(
                patch,  # Low-res patch coordinates
                self.fd,  # High-res frame dictionary
                self.scaling_factor
            )
            high_res_mappings.append(mapping)
            logging.debug(
                f"Mapped patch {idx} to frames {mapping['frame_numbers']} with pixels {mapping['high_pixel_range']}")
        return high_res_mappings

    @staticmethod
    def map_to_high_mag_with_frames(low_res_coords, high_res_frame_dict, scaling_factor):
        """
        Map low-resolution patch pixel coordinates to corresponding high-resolution frames and pixel ranges.

        Args:
            low_res_coords (dict):
                Dictionary with 'x_min', 'y_min', 'x_max', 'y_max' in low-resolution pixel coordinates.
            high_res_frame_dict (list of dict):
                High-resolution frame metadata containing pixel ranges.
            scaling_factor (float):
                Scaling factor between low-res and high-res.

        Returns:
            dict: Mapping with high-res pixel ranges and intersecting frames.
        """
        # Scale low-res pixel ranges to high-res
        high_x_min = int(low_res_coords['x_min'] * scaling_factor)
        high_y_min = int(low_res_coords['y_min'] * scaling_factor)
        high_x_max = int(low_res_coords['x_max'] * scaling_factor)
        high_y_max = int(low_res_coords['y_max'] * scaling_factor)

        # Debug: Validate the scaled ranges
        logging.debug(f"Low-res pixels: {low_res_coords}")

        # Initialize mapping result
        mapping_result = {
            'high_pixel_range': {
                'x_min': high_x_min,
                'y_min': high_y_min,
                'x_max': high_x_max,
                'y_max': high_y_max
            },
            'frame_numbers': [],
            'row_col': []
        }

        # Find intersecting frames in the high-res frame dictionary
        for frame in high_res_frame_dict:
            if (high_x_min <= frame['row_max'] and high_x_max >= frame['row_min'] and
                    high_y_min <= frame['col_max'] and high_y_max >= frame['col_min']):
                mapping_result['frame_numbers'].append(frame['frame'])
                mapping_result['row_col'].append(frame['row_col'])

        logging.debug(
            f"High-res pixels: x_min={high_x_min}, y_min={high_y_min}, x_max={high_x_max}, y_max={high_y_max}, row_col={mapping_result['row_col']}")
        # Warn if no frames were found
        if not mapping_result['frame_numbers']:
            logging.warning(f"No intersecting frames found for low-res coords {low_res_coords}")

        return mapping_result

    @staticmethod
    def frame_extraction(dcm, high_mag_mappings, num_high_res_frames, batch_size=10):
        """
        Extract frames from DICOM data using shared memory and batch processing.

        Args:
            dcm: DICOM object containing PixelData and frame information.
            high_mag_mappings: List of mappings with high-resolution frames and pixel ranges.
            num_high_res_frames: Maximum number of frames to extract. If None, extract all frames.
            batch_size: Number of frames to process in a single batch.

        Returns:
            List of dictionaries with extracted frames as NumPy arrays.
        """
        # Create shared memory for PixelData
        pixel_data = np.frombuffer(dcm.PixelData, dtype=np.uint8)
        shared_mem = shared_memory.SharedMemory(create=True, size=pixel_data.nbytes)
        shared_pixel_data = np.ndarray(pixel_data.shape, dtype=pixel_data.dtype, buffer=shared_mem.buf)
        np.copyto(shared_pixel_data, pixel_data)

        # Flatten the frame list for batching
        frame_tasks = [
            {
                'shared_mem_name': shared_mem.name,
                'total_frames': dcm.NumberOfFrames,
                'frame_id': frame,
                'task_index': idx
            }
            for idx, mapping in enumerate(high_mag_mappings)
            for frame in mapping['frame_numbers']
        ]

        # Limit the number of high mag frames
        # Randomly sample up to max_frames
        if num_high_res_frames is not None and len(frame_tasks) > num_high_res_frames:
            sampled_indices = random.sample(range(len(frame_tasks)), num_high_res_frames)
            frame_tasks = [frame_tasks[i] for i in sampled_indices]

        # Initialize tqdm for progress tracking
        results = []
        with tqdm(total=len(frame_tasks), desc="Extracting Frames", unit="frame") as pbar:
            with Pool() as pool:
                for i in range(0, len(frame_tasks), batch_size):
                    batch = frame_tasks[i:i + batch_size]
                    batch_results = pool.map(MagPairs.process_frame_batch, batch)
                    results.extend(batch_results)
                    pbar.update(len(batch))

        # Clean up shared memory
        shared_mem.close()
        shared_mem.unlink()

        logging.debug(f"Extracted {len(results)} high-res frames.")
        return results

    @staticmethod
    def process_frame_batch(task):
        """
        Process a single frame using shared memory.

        Args:
            task: Task dictionary containing shared memory name, frame_id, and metadata.

        Returns:
            Extracted frame as a NumPy array.
        """
        shared_mem = shared_memory.SharedMemory(name=task['shared_mem_name'])
        shared_pixel_data = np.ndarray((shared_mem.size,), dtype=np.uint8, buffer=shared_mem.buf)

        frame_id = task['frame_id']
        total_frames = task['total_frames']

        frame_generator = generate_pixel_data_frame(shared_pixel_data.tobytes(), total_frames)
        for i, frame_data in enumerate(frame_generator):
            if i == frame_id:
                try:
                    img = Image.open(io.BytesIO(frame_data))
                    img_arr = np.array(img)
                    if np.mean(img_arr) < 220:
                        return {'img_arr':img_arr,'frame_id':frame_id}
                    break
                except Exception as e:
                    logging.error(f"Failed to decode frame {frame_id}: {e}")

    @staticmethod
    def get_local_dcm_pair(dcm, bq_results_df):
        gcs_url_pair = bq_results_df['gcs_url'][
            (bq_results_df['SeriesInstanceUID'] == dcm.SeriesInstanceUID) &
            (bq_results_df['row_num_desc'] == 1)
            ]
        logging.debug(f'gcs_url_pair: {gcs_url_pair}')
        return read_dicom(gcs_url_pair)

    @staticmethod
    def get_pixel_spacing(dcm):
        return float(dcm.SharedFunctionalGroupsSequence[0].PixelMeasuresSequence[0].PixelSpacing[0])

    def get_frame_dict(self, dcm_input):
        dcm, total_pixel_matrix_columns, total_pixel_matrix_rows, columns, rows, grid_rows, grid_cols = parse_dcm_info(
            dcm_input
        )
        frame_list = []
        self.grid_cols = grid_cols  # Store grid_cols for later use if needed

        for row in range(grid_rows):
            for col in range(grid_cols):
                frame_index = row * grid_cols + col + 1  # Correct frame numbering
                # Adjust for edge frames
                row_min = row * rows
                row_max = min((row + 1) * rows, total_pixel_matrix_rows)  # Ensure row_max doesn't exceed matrix height
                col_min = col * columns
                col_max = min((col + 1) * columns, total_pixel_matrix_columns)

                frame_list.append({
                    'row_min': row_min,
                    'row_max': row_max,
                    'col_min': col_min,
                    'col_max': col_max,
                    'frame': frame_index,
                    'row_col': (row, col)
                })

        return frame_list

    def get_minmax(self, img_to_use_at_low_mag):
        """
        Calculate pixel ranges for patches in low-resolution image coordinates.

        Args:
            img_to_use_at_low_mag (list of tuple):
                List of tuples where each tuple contains:
                - `patch` (PIL.Image.Image): The image patch.
                - `raw_range` (tuple): The (row, col) position of the patch in the low-resolution image grid.
                - `idx` (int): The index of the patch.

            patch_size (tuple):
                The size of each patch (height, width) in pixels.

        Returns:
            list of dict:
                A list of dictionaries containing the pixel ranges for each patch:
                - `x_min`, `x_max`: Pixel range along the x-axis (columns).
                - `y_min`, `y_max`: Pixel range along the y-axis (rows).
                - `idx`: The index of the patch.
                - `patch_size`: The size of the patch.
        """
        return [
            {
                'x_min': raw_range[1] * patch.size[1],  # Convert column to x_min
                'x_max': (raw_range[1] + 1) * patch.size[1],  # Add patch width
                'y_min': raw_range[0] * patch.size[0],  # Convert row to y_min
                'y_max': (raw_range[0] + 1) * patch.size[0],  # Add patch height
                'idx': idx,  # Preserve the patch index
                'row_col': raw_range,
                'patch_size': patch.size  # Include the patch size for reference
            }
            for patch, raw_range, idx in img_to_use_at_low_mag
        ]

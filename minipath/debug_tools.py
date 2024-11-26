import matplotlib.pyplot as plt
import cv2
import numpy as np

def save_padded_array_with_labels(padded_array, patch_size=(8, 8), output_path="padded_array_with_labels.png"):
    """
    Display the padded array with (8, 8) patches outlined and their (x, y) positions labeled.

    Args:
        padded_array (np.ndarray): The input array to display (shape must be divisible by patch_size).
        patch_size (tuple): The size of each patch (height, width).
        output_path (str): File path to save the resulting image.

    """
    # Validate array shape
    height, width, _ = padded_array.shape
    patch_h, patch_w = patch_size
    if height % patch_h != 0 or width % patch_w != 0:
        raise ValueError("The padded_array dimensions must be divisible by the patch size.")

    # Create a copy to draw on
    image_with_grid = padded_array.copy()

    # Scale image for higher resolution labels
    scale_factor = 6  # Scale up the image for better text rendering
    scaled_h, scaled_w = height * scale_factor, width * scale_factor
    image_with_grid = cv2.resize(image_with_grid, (scaled_w, scaled_h), interpolation=cv2.INTER_NEAREST)

    # Update patch size to scaled dimensions
    patch_h, patch_w = patch_h * scale_factor, patch_w * scale_factor

    # Define font and scaling for text
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.3
    thickness = 1
    color = (0, 0, 255)  # Red text

    # Draw grid and label patches
    for y in range(0, scaled_h, patch_h):
        for x in range(0, scaled_w, patch_w):
            # Draw rectangle for the patch
            cv2.rectangle(image_with_grid, (x, y), (x + patch_w, y + patch_h), (255, 255, 255), 1)

            # Calculate center of the patch for text placement
            text_x = x + patch_w // 4
            text_y = y + patch_h // 2

            # Label the patch with its (x, y) coordinates
            label = f"({x // patch_w}, {y // patch_h})"
            cv2.putText(image_with_grid, label, (text_x, text_y), font, font_scale, color, thickness)

    # Convert to RGB for Matplotlib display
    image_with_grid = cv2.cvtColor(image_with_grid, cv2.COLOR_BGR2RGB)

    # Save the image using Matplotlib with high DPI
    plt.figure(figsize=(20, 20), dpi=300)
    plt.imshow(image_with_grid)
    plt.axis("off")
    plt.title("Padded Array with Patch Labels")
    plt.savefig(output_path, bbox_inches="tight", pad_inches=0)
    plt.close()

    print(f"High-quality image saved to {output_path}")


def plot_fg(foreground):
    plt.figure(figsize=(8, 8))
    plt.imshow(foreground)
    plt.title("Foreground (Tissue Regions)")
    plt.axis("off")
    plt.savefig('test/fg.png', bbox_inches="tight", pad_inches=0)
    plt.close()

def save_padded_array_with_coords(padded_array, patch_size=(8, 8), output_path="test/padded_array_with_labels.png", patch_coords=None, scale_factor=5):
    """
    Save the padded array as an image with patches outlined and their (x, y) positions labeled.
    The image is scaled to improve text readability.

    Args:
        padded_array (np.ndarray): The input array to save (shape must be divisible by patch_size).
        patch_size (tuple): The size of each patch (height, width).
        output_path (str): File path to save the resulting image.
        patch_coords (list of tuple): Coordinates of patches with calculated entropy.
        scale_factor (int): Factor to scale the image for better readability.
    """
    height, width, _ = padded_array.shape
    patch_h, patch_w = patch_size

    # Scale the image
    scaled_h, scaled_w = height * scale_factor, width * scale_factor
    scaled_array = cv2.resize(padded_array, (scaled_w, scaled_h), interpolation=cv2.INTER_NEAREST)

    # Adjust patch size to scaled dimensions
    patch_h, patch_w = patch_h * scale_factor, patch_w * scale_factor

    # Prepare for drawing
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.3  # Scale the font size
    thickness = 1   # Scale the thickness
    color = (0, 0, 255)  # Red text for entropy patches
    grid_color = (255, 255, 255)  # White for grid

    # Draw grid and labels
    for y in range(0, scaled_h, patch_h):
        for x in range(0, scaled_w, patch_w):
            grid_color = (0, 255, 0) if (y // patch_h, x // patch_w) in patch_coords else (255, 255, 255)
            cv2.rectangle(scaled_array, (x, y), (x + patch_w, y + patch_h), grid_color, 1)
            if patch_coords and (y // patch_h, x // patch_w) in patch_coords:
                cv2.putText(scaled_array, f"({x // patch_w}, {y // patch_h})",
                            (x + 5, y + patch_h // 2), font, font_scale, color, thickness)

    # Convert to RGB for Matplotlib
    scaled_array = cv2.cvtColor(scaled_array, cv2.COLOR_BGR2RGB)

    # Save the image using Matplotlib
    plt.figure(figsize=(12, 12), dpi=300)
    plt.imshow(scaled_array)
    plt.axis("off")
    plt.title("Padded Array with Patch Labels")
    plt.savefig(output_path, bbox_inches="tight", pad_inches=0)
    plt.close()



def debug_and_save_low_res_with_high_res_coords(
    low_res_array, high_res_mappings, scaling_factor, patch_size=(8, 8), output_path="test/debug_low_res_with_high_coords.png"
):
    height, width, _ = low_res_array.shape
    patch_h, patch_w = patch_size

    # Copy the image to draw on
    image_with_coords = low_res_array.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.4
    thickness = 1
    color_high_res = (255, 0, 0)  # Blue for high-res patch overlays
    color_low_res = (0, 255, 0)  # Green for low-res patch overlays

    # Draw reference grid
    for y in range(0, height, patch_h):
        for x in range(0, width, patch_w):
            cv2.rectangle(image_with_coords, (x, y), (x + patch_w, y + patch_h), color_low_res, 1)

    # Overlay high-res regions
    for idx, mapping in enumerate(high_res_mappings):
        if not isinstance(mapping, dict) or "high_pixel_range" not in mapping:
            logging.error(f"Invalid mapping at index {idx}: {mapping}")
            continue

        high_x_min = mapping["high_pixel_range"]["x_min"]
        high_y_min = mapping["high_pixel_range"]["y_min"]
        high_x_max = mapping["high_pixel_range"]["x_max"]
        high_y_max = mapping["high_pixel_range"]["y_max"]

        # Scale down to low-resolution coordinates
        low_x_min = int(high_x_min / scaling_factor)
        low_y_min = int(high_y_min / scaling_factor)
        low_x_max = int(high_x_max / scaling_factor)
        low_y_max = int(high_y_max / scaling_factor)

        print(f"Patch {idx}: High-res ({high_x_min}, {high_y_min}) -> Low-res ({low_x_min}, {low_y_min})")

        # Draw rectangle and label
        cv2.rectangle(image_with_coords, (low_x_min, low_y_min), (low_x_max, low_y_max), color_high_res, 2)
        cv2.putText(
            image_with_coords,
            f"({low_x_min},{low_y_min})",
            (low_x_min, low_y_min - 5),
            font,
            font_scale,
            color_high_res,
            thickness,
        )

    # Save the visualized image
    image_with_coords = cv2.cvtColor(image_with_coords, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(12, 12), dpi=300)
    plt.imshow(image_with_coords)
    plt.axis("off")
    plt.title("Low-Resolution Image with High-Resolution Coordinates")
    plt.savefig(output_path, bbox_inches="tight", pad_inches=0)
    plt.close()

    print(f"Image saved to {output_path}")

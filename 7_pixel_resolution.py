# 7. Estimate pixel resolution using known object size (DOF)

import cv2
import numpy as np

def estimate_pixel_resolution(image_path, known_length_mm, ref_points):
    """
    Estimate pixel resolution (mm/pixel).
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError("Image not found.")

    # Euclidean distance in pixels between ref points
    (x1, y1), (x2, y2) = ref_points
    pixel_length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

    # Resolution: mm per pixel
    resolution = known_length_mm / pixel_length
    return resolution


def compute_dof(f_mm, N, coc_mm, s_mm):
    """
    Compute Depth of Field (DOF) using lens equations.
    """
    H = (f_mm**2) / (N * coc_mm)  # Hyperfocal distance
    dof_near = (H * s_mm) / (H + (s_mm - f_mm))
    dof_far = (H * s_mm) / (H - (s_mm - f_mm)) if s_mm < H else float('inf')
    dof_total = dof_far - dof_near if dof_far != float('inf') else float('inf')
    return dof_near, dof_far, dof_total


# ------------------- Example Usage -------------------

# Path to your calibration image
image_path = "7_calibration_object.jpg"

# Known object length in mm (black rectangle = 10 mm)
known_length_mm = 10.0

# Pixel coordinates of the object edges (left and right edges of rectangle)
ref_points = ((200, 300), (300, 300))  # matches the synthetic image

# Step 1: Estimate resolution
resolution_mm_per_pixel = estimate_pixel_resolution(image_path, known_length_mm, ref_points)
print(f"Pixel resolution: {resolution_mm_per_pixel:.4f} mm/pixel")

# Step 2: Camera parameters (example values)
focal_length_mm = 50      # lens focal length
f_number = 8              # aperture f/8
coc_mm = 0.03             # circle of confusion (sensor dependent)
subject_distance_mm = 1000  # 1 m subject distance

# Step 3: Compute DOF
dof_near, dof_far, dof_total = compute_dof(focal_length_mm, f_number, coc_mm, subject_distance_mm)
print(f"DOF Near Limit: {dof_near:.2f} mm")
print(f"DOF Far Limit:  {dof_far:.2f} mm")
print(f"Total DOF:      {dof_total:.2f} mm")

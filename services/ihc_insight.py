# image_processing_utils.py

import cv2
import numpy as np
import base64
from io import BytesIO

def count_brown_pixels(filepath):
    """
    Counts the number of pixels within a specified brown color range in an image.
    """
    image = cv2.imread(filepath)
    if image is None:
        raise ValueError(f"Failed to load image: {filepath}")

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Two brown hue ranges to capture different shades
    # Hue (H): 0-179, Saturation (S): 0-255, Value (V): 0-255
    # Brown typically falls into low orange/yellow hues
    mask1 = cv2.inRange(hsv, np.array([5, 50, 20]), np.array([15, 255, 200]))  # Lighter browns
    mask2 = cv2.inRange(hsv, np.array([15, 80, 40]), np.array([25, 255, 220])) # Darker/more saturated browns

    combined_mask = cv2.bitwise_or(mask1, mask2)
    count = cv2.countNonZero(combined_mask)
    return count

def brown_shade_distribution(filepath):
    """
    Calculates the distribution of brown pixel intensities in an image.
    """
    image = cv2.imread(filepath)
    if image is None:
        raise ValueError(f"Failed to load image: {filepath}")

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Detect brown pixels using HSV thresholds
    mask1 = cv2.inRange(hsv, np.array([5, 50, 20]), np.array([15, 255, 200]))
    mask2 = cv2.inRange(hsv, np.array([15, 80, 40]), np.array([25, 255, 220]))
    combined_mask = cv2.bitwise_or(mask1, mask2)

    brown_pixels_hsv = cv2.bitwise_and(hsv, hsv, mask=combined_mask)
    h, s, v = cv2.split(brown_pixels_hsv)

    # Calculate intensity from saturation and value for brown pixels only
    # Convert to float to avoid overflow in division
    intensity = ((s.astype(np.float32) + v.astype(np.float32)) / 2)
    # Filter to only include pixels that are part of the brown mask
    intensity = intensity[combined_mask.astype(bool)]

    if intensity.size == 0: # Handle case with no brown pixels found
        return {"0–25%": 0, "25–50%": 0, "50–75%": 0, "75–100%": 0}

    # Normalize intensity to 0-100%
    intensity = intensity / 255 * 100

    bins = [0, 25, 50, 75, 100] # Define bins for intensity ranges
    # REMOVED: include_lowest=True as it's not supported in older NumPy versions
    hist, _ = np.histogram(intensity, bins=bins)

    return {
        "0–25%": int(hist[0]),
        "25–50%": int(hist[1]),
        "50–75%": int(hist[2]),
        "75–100%": int(hist[3]),
    }

def generate_brown_mask_image_base64(filepath):
    """
    Generates a base64 encoded PNG image of the detected brown pixel mask.
    """
    image = cv2.imread(filepath)
    if image is None:
        return None

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask1 = cv2.inRange(hsv, np.array([5, 50, 20]), np.array([15, 255, 200]))
    mask2 = cv2.inRange(hsv, np.array([15, 80, 40]), np.array([25, 255, 220]))
    combined_mask = cv2.bitwise_or(mask1, mask2)

    # Create a 3-channel image from the mask: white where brown is detected, black elsewhere
    # Convert to BGR for web display (PNGs are typically BGR/RGB)
    mask_display = cv2.cvtColor(combined_mask, cv2.COLOR_GRAY2BGR)

    # Encode the image to PNG format into a memory buffer
    is_success, buffer = cv2.imencode(".png", mask_display)
    if not is_success:
        return None

    # Convert the buffer to a base64 string
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    return img_base64
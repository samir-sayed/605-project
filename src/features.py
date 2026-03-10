"""
Feature extraction for face images.

Two modes:
  hog   : Histogram of Oriented Gradients (768-d for 62x47 images)
  pixel : Resized flattened normalized pixels (default 1024-d for 32x32)
"""
import numpy as np
from skimage.feature import hog
from skimage.transform import resize as sk_resize


def extract_hog_features(images: np.ndarray, config: dict) -> np.ndarray:
    """
    Extract HOG features from grayscale face images.

    Parameters
    ----------
    images : (N, H, W) float32 in [0, 1]
    config : loaded m2.yaml config dict

    Returns
    -------
    (N, D) float32 HOG feature matrix
    """
    hog_cfg = config["features"]["hog"]
    orientations = hog_cfg["orientations"]
    pixels_per_cell = tuple(hog_cfg["pixels_per_cell"])
    cells_per_block = tuple(hog_cfg["cells_per_block"])

    feats = []
    for img in images:
        fd = hog(
            img,
            orientations=orientations,
            pixels_per_cell=pixels_per_cell,
            cells_per_block=cells_per_block,
            feature_vector=True,
        )
        feats.append(fd)
    return np.array(feats, dtype=np.float32)


def extract_pixel_features(images: np.ndarray, config: dict) -> np.ndarray:
    """
    Resize each image to flatten_size and flatten.

    Parameters
    ----------
    images : (N, H, W) float32 in [0, 1]
    config : loaded m2.yaml config dict

    Returns
    -------
    (N, flatten_h * flatten_w) float32 feature matrix
    """
    h, w = config["features"]["pixel"]["flatten_size"]
    feats = []
    for img in images:
        resized = sk_resize(img, (h, w), anti_aliasing=True)
        feats.append(resized.ravel())
    return np.array(feats, dtype=np.float32)


def extract_features(images: np.ndarray, config: dict, mode: str = "hog") -> np.ndarray:
    """Dispatch to the requested feature extractor."""
    if mode == "hog":
        return extract_hog_features(images, config)
    elif mode == "pixel":
        return extract_pixel_features(images, config)
    else:
        raise ValueError(f"Unknown feature mode: {mode!r}. Choose 'hog' or 'pixel'.")

"""
Milestone 4 — CLI entrypoint for face verification.

Given two face images, produce a similarity score and a same/different decision.

Usage:
    python scripts/verify_pair.py --img1 path/to/face1.jpg --img2 path/to/face2.jpg
    python scripts/verify_pair.py --img1 face1.jpg --img2 face2.jpg --config configs/m4.yaml

Docker usage:
    docker run --rm -v $(pwd)/samples:/data lfw-verify \
        --img1 /data/face1.jpg --img2 /data/face2.jpg
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import json

import numpy as np
from PIL import Image
from skimage.transform import resize as sk_resize

from src.ingestion import load_config
from src.features import extract_features
from src.similarity import cosine_similarity_vectorized


def load_and_preprocess(image_path: str, target_shape=(62, 47)) -> np.ndarray:
    """Load an image file and preprocess to match LFW format."""
    img = Image.open(image_path).convert("L")  # grayscale
    img_arr = np.array(img, dtype=np.float32) / 255.0
    img_resized = sk_resize(img_arr, target_shape, anti_aliasing=True)
    return img_resized.astype(np.float32)


def main():
    parser = argparse.ArgumentParser(
        description="Face verification CLI — compare two face images"
    )
    parser.add_argument("--img1", required=True, help="Path to first face image")
    parser.add_argument("--img2", required=True, help="Path to second face image")
    parser.add_argument("--config", default="configs/m4.yaml", help="Config file path")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Override threshold (default: from config)")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    args = parser.parse_args()

    config = load_config(args.config)
    threshold = args.threshold or config["final_system"]["threshold"]
    mode = config["features"]["mode"]

    # Load and preprocess images
    img1 = load_and_preprocess(args.img1)
    img2 = load_and_preprocess(args.img2)

    # Extract features
    images = np.stack([img1, img2])
    features = extract_features(images, config, mode)

    # Compute similarity
    score = float(cosine_similarity_vectorized(
        features[0:1], features[1:2]
    )[0])

    # Decision
    decision = "SAME" if score >= threshold else "DIFFERENT"

    result = {
        "image_1": os.path.basename(args.img1),
        "image_2": os.path.basename(args.img2),
        "similarity_score": round(score, 4),
        "threshold": threshold,
        "decision": decision,
        "feature_type": mode,
        "metric": "cosine",
    }

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(f"Image 1:    {result['image_1']}")
        print(f"Image 2:    {result['image_2']}")
        print(f"Score:      {result['similarity_score']:.4f}")
        print(f"Threshold:  {result['threshold']:.4f}")
        print(f"Decision:   {result['decision']}")


if __name__ == "__main__":
    main()

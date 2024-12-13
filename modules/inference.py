import os
import numpy as np
import pandas as pd
from openvino.runtime import Core
from skimage.io import imread

def preprocess_image(image_path, Kp, Kf):
    img = imread(image_path)
    img = ((img - Kp) / Kf).astype(np.float32)
    img = np.expand_dims(img, axis=(0, 1))  # Add batch and channel dimensions
    return img

def run_inference(test_folder, model_path, output_file, log_callback=None):
    # Load OpenVINO model
    ie = Core()
    compiled_model = ie.compile_model(model=model_path, device_name="CPU")
    input_layer = compiled_model.input(0)
    output_layer = compiled_model.output(0)

    # Prepare test data
    image_pairs = {}
    if any(os.path.isdir(os.path.join(test_folder, d)) for d in os.listdir(test_folder)):
        # If there are subfolders, process them
        for subfolder in os.listdir(test_folder):
            subfolder_path = os.path.join(test_folder, subfolder)
            if not os.path.isdir(subfolder_path):
                continue
            for filename in os.listdir(subfolder_path):
                if "rec0" in filename:
                    pair_name = filename.replace("rec0", "rec1")
                    rec0_path = os.path.join(subfolder_path, filename)
                    rec1_path = os.path.join(subfolder_path, pair_name)
                    if os.path.exists(rec1_path):
                        image_pairs[rec0_path] = rec1_path
    else:
        # If no subfolders, process files in the main folder
        for filename in os.listdir(test_folder):
            if "rec0" in filename:
                pair_name = filename.replace("rec0", "rec1")
                rec0_path = os.path.join(test_folder, filename)
                rec1_path = os.path.join(test_folder, pair_name)
                if os.path.exists(rec1_path):
                    image_pairs[rec0_path] = rec1_path

    # Process each pair and log results
    results = []
    for idx, (rec0_path, rec1_path) in enumerate(image_pairs.items(), start=1):
        if log_callback:
            log_callback(f"Processing pair {idx}: {rec0_path} and {rec1_path}")

        img0 = preprocess_image(rec0_path, 2**15, float(2**15))
        img1 = preprocess_image(rec1_path, 2**15, float(2**15))

        outputs0 = compiled_model([img0])[output_layer]
        outputs1 = compiled_model([img1])[output_layer]

        softmax_scores0 = np.exp(outputs0) / np.sum(np.exp(outputs0), axis=1, keepdims=True)
        softmax_scores1 = np.exp(outputs1) / np.sum(np.exp(outputs1), axis=1, keepdims=True)

        results.append({
            "rec0_path": rec0_path,
            "rec1_path": rec1_path,
            "class_rec0": np.argmax(softmax_scores0),
            "confidence_rec0": np.max(softmax_scores0),
            "class_rec1": np.argmax(softmax_scores1),
            "confidence_rec1": np.max(softmax_scores1)
        })

    pd.DataFrame(results).to_csv(output_file, index=False)
    if log_callback:
        log_callback(f"Results saved to {output_file}")

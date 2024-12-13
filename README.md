# Pollen Recognition
Pollen Recognition from a Pair of Holographic Images__
![image](https://github.com/user-attachments/assets/9d4b0fd1-fc30-4d9c-8e4f-f47a417cfd72)

This project is a web application for recognizing pollen from pairs of holographic images using a machine learning model. The application uses OpenVINO for model inference and Flask to serve a web-based interface for user interaction.

# Features

-Upload a folder containing test images and an ONNX model file.
-Run inference on pairs of images (e.g., rec0 and rec1) using OpenVINO.
-Monitor logs in real-time during the inference process.
-View the results in a tabular format on the web interface.
-Save inference results to a CSV file.

# Requirements

## Python Version
Python >= 3.7.1

##Libraries
All required libraries are listed in the requirements.txt file. Install them with:
```
pip install -r requirements.txt
```

## Key Dependencies:
-flask: For the web interface.
-numpy: For numerical operations.
-pandas: For handling results.
-openvino: For running the ONNX model inference.
-scikit-image: For image processing.




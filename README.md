# Pollen Recognition
Pollen Recognition from a Pair of Holographic Images   

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

# How to Run Application  

### Clone the Repository  

To begin, clone the repository to your local machine:  

```
git clone <repository-url>
cd <repository-folder>
```

### Install Dependencies

Ensure you have Python installed (version >= 3.7.1). Then, install the required dependencies listed in the requirements.txt file:  

```
pip install -r requirements.txt
```

### Update the Log Path

Before running the application, you need to configure the path where the inference results will be saved. Open web_app.py and set the CONFIDENCE_LOG_PATH variable to a valid path where the .csv file can be created. For example:

```
CONFIDENCE_LOG_PATH = r"D:/log.csv"
```
Ensure that the directory exists or create it before proceeding.  

### Prepare Your Files

- Test Folder: Organize your test images in the specified format, ensuring they contain paired files (e.g., rec0 and rec1).
  
- ONNX Model File: Ensure you have a valid ONNX model file to use for inference.


### Run the Flask Application

Start the Flask server by running the following command in the project directory:  

```
python web_app.py
```

### Access the Web Application

Once the Flask server is running, open your web browser and navigate to:  

```
http://127.0.0.1:5000
```

### Upload Files and Run Inference

- Enter the path to the test folder in the input field. (For example: D:\test_folder)
  
- Upload your ONNX model file.
  
- Click on Run Inference to start the process.



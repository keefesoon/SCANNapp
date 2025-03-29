# SCANNapp

SCANNapp is a web interface built using Flask, PythonOCC, and TensorFlow that allows users to upload CAD files, process them, and perform Machining cost estimation.

## Features
- Upload STEP files for 3D model visualization.
- Process and extract geometry using PythonOCC.
- Estimate costs based on labour, raw materials, and additional processes.
- Predict machine hours using an ANN model.

## Requirements
- Python 3.12 (or the version specified)
- Flask 3.1.0
- PythonOCC 7.8.1.1
- TensorFlow 2.18.0 
- [Other dependencies listed in `requirements.txt` or `environment.yml`]

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/SCANNapp.git
   cd SCANNapp

2. Create a virtual environment and install dependencies:
   Using Conda:
   conda env create -f environment.yml
   conda activate yourenv

   Or using pip and a virtual environment:
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt

3. Run the application
   python app.py


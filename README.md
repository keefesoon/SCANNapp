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

## Folder Structure

A recommended folder structure is:
SCANNapp/
├── data/
│   ├── LabourRate.csv
│   ├── MaterialGrade.csv
│   ├── MaterialShape.csv
│   ├── options.csv
│   ├── RM Cost.csv
│   ├── SecondaryProcesses.csv
│   ├── SurfaceFinish.csv
│   └── Threading Cost.csv
├── models/
│   ├── ann_model1_new.keras
│   ├── ann_model1.h5
│   └── ann_model1.keras
├── python/
│   ├── __pycache__/
│   ├── templates/
│   │   ├── index.html
│   │   └── threading_select.html
│   ├── app.py
│   ├── cost_est.py
│   ├── pyocc_helpers.py
│   ├── test_viewer.py
│   └── test.py
├── static/
│   └── tmp/
├── uploads/
├── environment.yml
├── .gitignore
└── README.md

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

## Usage Instructions
### Single-Part Mode:
1. Upload a STEP File:
Click the "Upload STEP file" button and select a STEP file from your computer.

2. Select Material Shape:
Choose the material shape (e.g., ROD or TUBE) from the dropdown before clicking "Scan Surfaces."

3. Scan & Fill Inputs:
Click "Scan Surfaces" to process the file and extract geometry, then fill in the remaining input fields (Region, Material Specification, Surface Finish, Secondary Processes, etc.).

4. Predict Cost:
Click "Predict Cost" to calculate machining cost.

5. Generate Report (Optional):
Click "Generate Report" to download a CSV report with the detailed cost breakdown.

### Batch Mode:
1. Prepare a CSV File:
Create a CSV file with one row per part and include all required columns.

2. Upload CSV:
In the Batch Upload section, select your CSV file and click "Batch Upload" to process all rows and generate a results CSV.

## Troubleshooting
Empty Dropdowns:
If some dropdowns (e.g., for Region, Surface Finish, Material Specification) aren’t loading options, verify that the corresponding CSV files in the data/ folder contain non-empty data in the expected columns. Also, check that the column names in the CSV files match exactly what is expected (including case and spacing).

Template Not Found:
If you get a TemplateNotFound: index.html error, ensure that your index.html file is located in the templates/ folder (or configure your Flask app with the correct template_folder).


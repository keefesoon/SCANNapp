import sys
import os
import io
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import math
import random
import subprocess
import pandas as pd
import numpy as np
import scipy
import tensorflow as tf
from tensorflow.python.keras.layers import Input, Dense
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.models import load_model
from scipy.stats import boxcox, yeojohnson

from flask import Flask, jsonify, render_template, request, redirect, url_for, flash, session, send_file

from cost_est import (
    cost_est,
    get_threading_cost,
    get_labour_rate,
    get_rm_cost,
    get_secondary_processes_cost,
    reapply_transformation,
    load_options
)

#from pyocc_helpers import run_occ_viewer, fake_pyocc, stepreader, FindLength, CalcRawLength, RodVol, TubeVol
from pyocc_helpers import pyocc, run_occ_viewer, stepreader, FindLength,FindDiam, CalcRawLength, CalcRawDiam, getprops, RodVol, RodSA, TubeVol, TubeSA, VolumeFaced, CalcSurfaceArea, CalcVolOD, FindIR, FindOR, FindOuterSurfaces, FindMainHole, CircularEdgeDetector, findpocket, edgecompare, getnormal, similardirection, find_adjacent_faces, CollectSimFaces, CollSurfaceArea, NewFindGunDrillHole, calculate_main_hole_volume, pyocc



# PythonOCC
from OCC.Extend.DataExchange import read_step_file, write_stl_file
from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.Bnd import Bnd_Box
from OCC.Core.BRepBndLib import brepbndlib
from OCC.Display.SimpleGui import init_display

################################################################################
# Flask App
################################################################################
# Set global paths relative to this file’s location.
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")

app = Flask(__name__)
app.secret_key = "some_secret_key_here"
app.config['UPLOAD_FOLDER'] = os.path.join(BASE_DIR, "uploads")
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# We store the "Threading Cost.csv" in memory for the Approach A route
threading_csv_path = os.path.join(DATA_DIR, "Threading Cost.csv")
df_thread = pd.read_csv(threading_csv_path, encoding="utf-8", dtype=str)  # all as string for easy matching

@app.route('/', methods=['GET','POST'])
def index():
    """The main route that shows the big form (upload, region, etc.)."""


    # 1) Retrieve any cost/breakdown from session (they are popped so they won't persist after one GET)
    cost = session.get('machining_cost', None)
    breakdown = session.get('breakdown', None)
    stl_model = session.get('stl_model', None)
    surf_detect = session.get('surf_detect', None)

    # 2) Retrieve previously saved form inputs (if any)
    form_data = session.get('last_form_data', {})

    # 3) Load the dropdown options, etc.
    opts = load_options()

    if request.method == 'POST':
        if 'batch_upload' in request.form:
            return process_batch_upload()
        
        # Save the entire POSTed form in session so we can re-populate fields after redirect
        session['last_form_data'] = request.form.to_dict(flat=False)

        # Handle which button was clicked
        if 'scan_surfaces' in request.form:
            return upload_and_scan()
        elif 'generate_report' in request.form:
            return generate_report() # our csv generator
        elif 'predict_cost' in request.form:
            return predict_cost()
        elif 'batch_upload' in request.form:
            return process_batch_upload()  # <-- new function for batch CSV
        else:
            flash("No valid action specified.")
            return redirect(url_for('index'))

    # If it's a GET, just render the page with any cost/breakdown and the form_data
    return render_template(
        'index.html',
        cost=cost,
        breakdown=breakdown,
        stl_model=stl_model,
        surf_detect=surf_detect,
        form_data=form_data,
        **opts
    )

# ----------------------------------------------------------------------
# 6) The AJAX endpoint for dynamic thread fields
# ----------------------------------------------------------------------
@app.route('/api/thread_options')
def thread_options():
    """
    Example: /api/thread_options?type=X&material=Y&size=Z&connection=C
    We filter df_thread accordingly and return distinct values for
    each field (type/material/size/connection).
    """
    sel_type = request.args.get('type','').strip()
    sel_mat  = request.args.get('material','').strip()
    sel_size = request.args.get('size','').strip()
    sel_conn = request.args.get('connection','').strip()

    filtered = df_thread

    # Filter step by step
    if sel_type:
        filtered = filtered.loc[filtered.iloc[:,0] == sel_type]
    if sel_mat:
        filtered = filtered.loc[filtered.iloc[:,1] == sel_mat]
    if sel_size:
        filtered = filtered.loc[filtered.iloc[:,2] == sel_size]
    if sel_conn:
        filtered = filtered.loc[filtered.iloc[:,3] == sel_conn]

    # Distinct sets
    types       = sorted(filtered.iloc[:,0].unique().tolist())
    materials   = sorted(filtered.iloc[:,1].unique().tolist())
    sizes       = sorted(filtered.iloc[:,2].unique().tolist())
    connections = sorted(filtered.iloc[:,3].unique().tolist())

    return jsonify({
        "types": types,
        "materials": materials,
        "sizes": sizes,
        "connections": connections
    })

# ----------------------------------------------------------------------
# 7) upload_and_scan logic
# ----------------------------------------------------------------------
def upload_and_scan():
    step_file = request.files.get('upload_step')
    if not step_file:
        flash("No STEP file provided.")
        return redirect(url_for('index'))

    if not step_file.filename.lower().endswith('.step'):
        flash("Please select a valid .step file.")
        return redirect(url_for('index'))

    step_path = os.path.join(app.config['UPLOAD_FOLDER'], step_file.filename)
    step_file.save(step_path)
    flash(f"Uploaded {step_file.filename} successfully.")

    # Convert to STL for a web preview if desired:
    try:
        shape = read_step_file(step_path)
        stl_dir = os.path.join('static','tmp')
        os.makedirs(stl_dir, exist_ok=True)
        base_name = os.path.splitext(step_file.filename)[0]
        stl_filename = base_name + '.stl'
        stl_path = os.path.join(stl_dir, stl_filename)
        write_stl_file(shape, stl_path)
        session['stl_model'] = os.path.join('tmp', stl_filename)
        flash("Surfaces scanned and STL generated successfully.")
    except Exception as e:
        flash(f"Error generating STL: {e}")


    # Get the material shape from the form
    material_shape = request.form.get('material_shape', '').strip().upper()

    # call the pyocc function to compute geometry
    try:
        results_dict, faces = pyocc(step_path, material_shape)
        print("DEBUG: Geometry results from pyocc:", results_dict)
        sys.stdout.flush()
        session['pyocc_data'] = results_dict
        # Also store the detection percentage separately for your interface:
        session['surf_detect'] = results_dict.get("Det", "N/A")
    except Exception as geo_err:
        flash(f"Error running pyocc on the uploaded STEP: {geo_err}")


    python_exe = sys.executable
    viewer_script = os.path.join(os.path.dirname(__file__), "viewer.py")
    print("DEBUG: viewer_script =", viewer_script)
    subprocess.Popen([python_exe, viewer_script, step_path])

    return redirect(url_for('index'))

def generate_report():
    """
    Creates a CSV file reflecting the breakdown (raw_material, labour_rate, etc.)
    plus surfaces detected, etc., and returns it as a download.
    """
    # Retrieve data from session
    breakdown = session.get('breakdown', {})
    machining_cost = session.get('machining_cost', 0)
    surf_detect = session.get('surf_detect', "N/A")
    
    # Build rows for the CSV. You can adjust the labels as needed.
    rows = [
        ["Raw material ($)", breakdown.get("raw_material", 0)],
        ["Labour Rate ($)", breakdown.get("labour_rate", 0)],
        ["Sec. Process ($)", breakdown.get("sec_process", 0)],
        ["Threading Cost ($)", breakdown.get("threading", 0)],
        ["Misc ($)", breakdown.get("misc", 0)],
        ["Machining Hours", round(breakdown.get("machine_hours", 0),2)],
        ["Mass (Kg)", round(breakdown.get("mass", 0),2)],
        ["Volume", round(breakdown.get("volume", 0),2)],
        ["Surfaces Detected (%)", round(surf_detect,2)],
        ["Machining cost ($)", round(machining_cost,2)]
    ]
    
    # Create a DataFrame
    df = pd.DataFrame(rows, columns=["Item", "Value"])
    
    # Write the DataFrame to an in-memory CSV file
    output = io.StringIO()
    df.to_csv(output, index=False)
    output.seek(0)
    
    # Return the CSV file as a download
    return send_file(
        io.BytesIO(output.getvalue().encode("utf-8")),
        as_attachment=True,
        download_name="Report.csv",
        mimetype="text/csv"
    )

@app.route("/batch_upload", methods=["POST"])
def process_batch_upload():
    """
    Handles the 'Batch CSV' upload. For each row in the CSV, we parse the inputs,
    call pyocc(...) for geometry, then cost_est(...) for cost calculations,
    and finally produce an output CSV with the results.
    """
    # 1) Get the uploaded CSV file from the form
    file_obj = request.files.get('batch_csv')
    if not file_obj:
        flash("No batch CSV provided.")
        return redirect(url_for('index'))

    # 2) Read the CSV into a DataFrame
    import pandas as pd
    try:
        df = pd.read_csv(file_obj)
    except Exception as e:
        flash(f"Error reading the uploaded CSV: {e}")
        return redirect(url_for('index'))

    # 3) Prepare a list to hold the results for each row
    results_list = []

    # 4) Iterate over each row in the input CSV
    for idx, row in df.iterrows():
        # ------------------------------------------------------------------
        # (A) Extract all input columns from the row
        #     Make sure to adapt column names to your actual CSV headings
        # ------------------------------------------------------------------
        part_number     = str(row.get('Part Number', '')).strip()
        region          = str(row.get('Region', '')).strip()
        mat_spec        = str(row.get('Material Specs', '')).strip()
        mat_shape       = str(row.get('Material Shape', '')).strip().upper()
        surf_finish     = str(row.get('Surface Finish', '')).strip()
        offset_val      = str(row.get('Offset (on/off)', 'off')).strip()
        misc_str        = str(row.get('Misc Cost', '0')).strip()
        # If it's "nan" or empty, treat as "0"
        if misc_str.lower() in ("nan", ""):
            misc_str = "0"
        try:
            misc_cost_val = float(misc_str)
        except ValueError:
            misc_cost_val = 0.0

        # Threading #1 columns
        thr_type_1 = str(row.get('Threading', '')).strip()
        thr_conn_1 = str(row.get('Threading Connection', '')).strip()
        thr_mat_1  = str(row.get('Material', '')).strip()
        thr_qty_1_raw = row.get('Qty', 0)
        print("DEBUG Qty raw value:", thr_qty_1_raw)  # Debug print
        try:
            thr_qty_1 = int(float(str(thr_qty_1_raw).strip()))
        except ValueError:
            thr_qty_1 = 0
        thr_size_1 = str(row.get('Size', '')).strip()
        print("DEBUG threading #1:", thr_type_1, thr_conn_1, thr_mat_1, thr_qty_1, thr_size_1)
        
        

        # Threading #2 columns
        thr_type_2 = str(row.get('Threading 2', '')).strip()
        thr_conn_2 = str(row.get('Threading Connection 2', '')).strip()
        thr_mat_2  = str(row.get('Material.1', '')).strip()
        thr_qty_2  = row.get('Qty.1', 0)
        thr_size_2 = str(row.get('Size.1', '')).strip()

        # Threading #3 columns
        thr_type_3 = str(row.get('Threading 3', '')).strip()
        thr_conn_3 = str(row.get('Threading Connection 3', '')).strip()
        thr_mat_3  = str(row.get('Material.2', '')).strip()
        thr_qty_3  = row.get('Qty.2', 0)
        thr_size_3 = str(row.get('Size.2', '')).strip()

        # Threading #4 columns
        thr_type_4 = str(row.get('Threading 4', '')).strip()
        thr_conn_4 = str(row.get('Threading Connection 4', '')).strip()
        thr_mat_4  = str(row.get('Material.3', '')).strip()
        thr_qty_4  = row.get('Qty.3', 0)
        thr_size_4 = str(row.get('Size.3', '')).strip()

        # Secondary processes
        sec_proc_str = str(row.get('Secondary Processes', '')).strip()
        # Could be something like "Process1, Process2" or "Process1;Process2"
        # We'll split by comma here:
        selected_procs = [p.strip() for p in sec_proc_str.split(',') if p.strip()]

        # STEP file path
        step_path = str(row.get('File Path (Step file)', '')).strip()
    
        # ------------------------------------------------------------------
        # (B) Build the 'thread_inputs' list, up to 4 boxes
        # ------------------------------------------------------------------
        thread_inputs = []
        # Box 1
        if thr_type_1 and thr_conn_1 and thr_mat_1 and thr_qty_1:
            try:
                thr_qty_1 = int(thr_qty_1)
            except ValueError:
                thr_qty_1 = 0
            if thr_qty_1 > 0:
                thread_inputs.append({
                    "thr_type": thr_type_1,
                    "thr_mat": thr_mat_1,
                    "thr_size": thr_size_1,
                    "thr_conn": thr_conn_1,
                    "thr_qty": thr_qty_1
                })
        # Box 2
        if thr_type_2 and thr_conn_2 and thr_mat_2 and thr_qty_2:
            try:
                thr_qty_2 = int(thr_qty_2)
            except ValueError:
                thr_qty_2 = 0
            if thr_qty_2 > 0:
                thread_inputs.append({
                    "thr_type": thr_type_2,
                    "thr_mat": thr_mat_2,
                    "thr_size": thr_size_2,
                    "thr_conn": thr_conn_2,
                    "thr_qty": thr_qty_2
                })
        # Box 3
        if thr_type_3 and thr_conn_3 and thr_mat_3 and thr_qty_3:
            try:
                thr_qty_3 = int(thr_qty_3)
            except ValueError:
                thr_qty_3 = 0
            if thr_qty_3 > 0:
                thread_inputs.append({
                    "thr_type": thr_type_3,
                    "thr_mat": thr_mat_3,
                    "thr_size": thr_size_3,
                    "thr_conn": thr_conn_3,
                    "thr_qty": thr_qty_3
                })
        # Box 4
        if thr_type_4 and thr_conn_4 and thr_mat_4 and thr_qty_4:
            try:
                thr_qty_4 = int(thr_qty_4)
            except ValueError:
                thr_qty_4 = 0
            if thr_qty_4 > 0:
                thread_inputs.append({
                    "thr_type": thr_type_4,
                    "thr_mat": thr_mat_4,
                    "thr_size": thr_size_4,
                    "thr_conn": thr_conn_4,
                    "thr_qty": thr_qty_4
                })
        print("DEBUG row keys:", row.keys())
        

        # ------------------------------------------------------------------
        # (C) Optionally run pyocc(...) to get geometry data
        # ------------------------------------------------------------------
        pyocc_data = {
            # fallback defaults if we can't read the step file
            "Length": 130.0,
            "VOL": 250.0,
            "Act_Vol": 250.0,
            "VF": 0.0,
            "VID": 0.0,
            "VOD": 0.0,
            "SP": 0.0,
            "SGDH": 0.0,
            "HRC": 0
        }
        # We'll store the surface detection % in e.g. 'Det'
        surface_detect = 0.0

        # Convert offset string "on"/"off" properly
        offset_str = "on" if offset_val.lower() == "on" else "off"

        if step_path and os.path.isfile(step_path):
            try:
                geo_results, _faces = pyocc(step_path, mat_shape)
                print("DEBUG geo_results:", geo_results)
                # e.g. geo_results might have: "Det", "Length", "VOL", "VF", etc.
                surface_detect = geo_results.get("Det", 0.0)

                # Copy them into pyocc_data
                pyocc_data["Length"]   = geo_results.get("Length", 130.0)
                pyocc_data["VOL"]      = geo_results.get("VOL", 250.0)
                pyocc_data["Act_Vol"]  = geo_results.get("Act_Vol", 250.0)
                pyocc_data["VF"]       = geo_results.get("VF", 0.0)
                pyocc_data["VID"]      = geo_results.get("VID", 0.0)
                pyocc_data["VOD"]      = geo_results.get("VOD", 0.0)
                pyocc_data["SP"]       = geo_results.get("SP", 0.0)
                pyocc_data["SGDH"]     = geo_results.get("SGDH", 0.0)
                # If you have "HRC" from somewhere else, add that too if needed
                # pyocc_data["HRC"] = ?
                

            except Exception as geo_ex:
                print(f"Error in pyocc for row {idx}: {geo_ex}")
                # fallback to defaults above
        else:
            print(f"Row {idx}: STEP file path is invalid or empty -> {step_path}")
 
         # (C) Now call cost_est(...) if you want cost for each row
        #Debug statement
        print("DEBUG: region =", region)
        print("DEBUG: mat_spec =", mat_spec)
        print("DEBUG: surf_finish =", surf_finish)
        print("DEBUG: offset_val =", offset_val)
        print("DEBUG: pyocc_data =", pyocc_data)
        print("DEBUG: thread_inputs =", thread_inputs)
        print("DEBUG: selected_procs =", selected_procs)

        total_cost, lb_cost, rm_cost_val, sp_cost_val, thr_cost_val, machine_hrs, volume_val, mass_val = cost_est(
            region=region,
            lb_file="LabourRate.csv",
            rm_file="RM Cost.csv",
            thr_file="Threading Cost.csv",
            sp_file="SecondaryProcesses.csv",
            mat_spec=mat_spec,
            sf=surf_finish,
            offset=offset_val,
            sec_proc=selected_procs,
            pyocc=pyocc_data,
            thread_inputs=thread_inputs
        )
        # Add the user-specified misc cost
        total_cost += misc_cost_val
        print("DEBUG after adding misc:", total_cost)

        # (D) Build a dictionary of output columns for this row
        row_result = {
            "Part Number": part_number,
            "Region": region,
            "Material Specs": mat_spec,
            "Material Shape": mat_shape,  # if you want the original "Tube"/"Rod"
            "Surface Finish": surf_finish,
            "Offset": offset_val,
            "Surfaces Detected (%)": round(surface_detect,2),
            "Raw Material": rm_cost_val,
            "Labour Rate": lb_cost,
            "Sec. Process": sp_cost_val,
            "Threading": thr_cost_val,
            "Misc": misc_cost_val,
            "Machine Hours": round(machine_hrs,2),
            "Mass (kg)": round(mass_val,2),
            "Volume (in^3)": volume_val,
            "Total Cost ($)": round(total_cost,2)
        }

        results_list.append(row_result)

    # -----------------------------------------------------------
    # 5) Convert results_list -> DataFrame
    # -----------------------------------------------------------
    results_df = pd.DataFrame(results_list)
    print("DEBUG: results_df after building:\n", results_df)

    # 6) Write results_df to a new CSV in-memory
    import io
    output_stream = io.StringIO()
    results_df.to_csv(output_stream, index=False)
    output_stream.seek(0)

    # 7) Return the CSV as a file download
    from flask import send_file
    return send_file(
        io.BytesIO(output_stream.getvalue().encode("utf-8")),
        as_attachment=True,
        download_name="Batch_Results.csv",
        mimetype="text/csv"
    )
    print("hi")


def load_and_convert_model():
    """
    Loads the ANN model from a Keras H5 file.
    """
    #keras_path = os.path.join(os.path.dirname(__file__), "ann_model1.keras")
    keras_path = os.path.join("models", "ann_model1.keras")

    if os.path.exists(keras_path):
        print("Loading model from Keras H5 file:", keras_path)
        return tf.keras.models.load_model(keras_path)
    else:
        raise FileNotFoundError(f"Keras model file not found at: {keras_path}")
    
@app.route('/reset_all', methods=['POST'])
def reset_all():
    # Clear out any session data
    session.clear()
    print("DEBUG after session.clear(), keys are:", list(session.keys()))

    # If you also want to remove the STL/STEP file from disk, do so here.
    # e.g. if 'stl_model' is in session, remove that file:
    # if 'stl_model' in session:
    #     stl_path = os.path.join('static', session['stl_model'])
    #     if os.path.exists(stl_path):
    #         os.remove(stl_path)
    # You could do something similar for the uploaded STEP file.

    flash("All data has been reset.")
    return redirect(url_for('index'))



def predict_cost():
    # Build a path to "
    # _model1.keras" in the same folder as app.py
    model = load_and_convert_model()
    # Gather and strip basic inputs from the form
    
    material_spec = request.form.get('material_spec', '').strip()
    region = request.form.get('region', '').strip()
    offset = request.form.get('offset')  # "on" if checked, else None
    misc_cost_str = request.form.get('misc_cost', '0').strip()
    try:
        misc_cost_val = float(misc_cost_str)
    except ValueError:
        misc_cost_val = 0.0

    # Get PythonOCC geometry data from session (set during upload_and_scan)
    pyocc_data = session.get("pyocc_data", {
    "Length": 130.0,
    "Act_Vol": 250.0,  # Actual volume from the STEP properties
    "VOL": 250.0,      # Raw (stock) volume computed from dimensions
    "VF": 0,
    "VID": 0,
    "SID": 0,
    "VOD": 0,
    "SOD": 0,
    "SP": 0,
    "NGDH": 0,
    "SGDH": 0,
    "HRC": 0
    })
    print("DEBUG: pyocc_data retrieved in predict_cost:", pyocc_data)

    # Build a list of threading inputs for boxes 1 to 4
    thread_inputs = []
    for i in range(1, 5):
        t_type = request.form.get(f'threading{i}_type', '').strip()
        t_mat  = request.form.get(f'threading{i}_material', '').strip()
        t_conn = request.form.get(f'threading{i}_shape', '').strip()
        t_size = request.form.get(f'threading{i}_size', '').strip()
        qty_str = request.form.get(f'threading{i}_qty', '0').strip()
        try:
            t_qty = int(qty_str)
        except ValueError:
            t_qty = 0

        # Only add if all required fields are provided and quantity is positive
        if t_type and t_mat and t_conn and t_size and t_qty > 0:
            thread_inputs.append({
                "thr_type": t_type,
                "thr_mat": t_mat,
                "thr_size": t_size,
                "thr_conn": t_conn,
                "thr_qty": t_qty
            })

    # Surface finish selection (if applicable)
    surface_finish = request.form.get('surface_finish', '').strip()

    print("DEBUG surface_finish raw:", surface_finish)

    # Get secondary processes (as a stripped list)
    secondary_procs = [proc.strip() for proc in request.form.getlist('secondary_processes')]

    # Define CSV file paths (ensure these files are in the correct locations)
    lb_file = "LabourRate.csv"
    rm_file = "RM Cost.csv"
    thr_file = "Threading Cost.csv"
    sp_file = "SecondaryProcesses.csv"

    print("DEBUG in predict_cost: region =", region, " material_spec =", material_spec)

    # Call the consolidated cost estimation function (which now accepts thread_inputs)
    total_cost, lb_cost, rm_cost, sp_cost, thr_cost, machine_hours, vol, mass = cost_est(
        region=region,
        lb_file=lb_file,
        rm_file=rm_file,
        thr_file=thr_file,
        sp_file=sp_file,
        mat_spec=material_spec,
        sf=surface_finish,
        offset=offset,
        sec_proc=secondary_procs,
        pyocc=pyocc_data,
        thread_inputs=thread_inputs
    )

    # Optionally, add miscellaneous cost
    total_cost += misc_cost_val

   
    breakdown_dict = {
        "raw_material": rm_cost,
        "labour_rate": lb_cost,
        "sec_process": sp_cost,
        "threading": thr_cost,
        "misc": misc_cost_val,
        "machine_hours": machine_hours,
        "volume": vol,
        "mass": mass
    }

    session['machining_cost'] = round(total_cost,2)
    session['breakdown'] = breakdown_dict
    flash(f"Total cost for {material_spec} in {region} => ${total_cost:.2f}")
    return redirect(url_for('index'))

if __name__ == "__main__":
    # If user runs "python app.py viewer path/to/file", we do run_occ_viewer
    if len(sys.argv) >= 3 and sys.argv[1] == "viewer":
        run_occ_viewer(sys.argv[2])
        sys.exit(0)
    else:
        app.run(debug=True, host="127.0.0.1", port=5001)
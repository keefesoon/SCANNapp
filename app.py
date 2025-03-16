import sys
import os

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

from flask import Flask, jsonify, render_template, request, redirect, url_for, flash, session

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
app = Flask(__name__)
app.secret_key = "some_secret_key_here"
app.config['UPLOAD_FOLDER'] = "uploads"
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# We store the "Threading Cost.csv" in memory for the Approach A route
df_thread = pd.read_csv("Threading Cost.csv", encoding="utf-8", dtype=str)  # all as string for easy matching

@app.route('/', methods=['GET','POST'])
def index():
    """The main route that shows the big form (upload, region, etc.)."""

    # 1) Retrieve any cost/breakdown from session (they are popped so they won't persist after one GET)
    cost = session.pop('machining_cost', None)
    breakdown = session.pop('breakdown', None)
    stl_model = session.pop('stl_model', None)
    surf_detect = session.pop('surf_detect', None)

    # 2) Retrieve previously saved form inputs (if any)
    form_data = session.pop('last_form_data', {})

    # 3) Load the dropdown options, etc.
    opts = load_options()

    if request.method == 'POST':
        # Save the entire POSTed form in session so we can re-populate fields after redirect
        session['last_form_data'] = request.form.to_dict(flat=False)

        # Handle which button was clicked
        if 'scan_surfaces' in request.form:
            return upload_and_scan()
        elif 'generate_report' in request.form:
            return generate_report()
        elif 'predict_cost' in request.form:
            return predict_cost()
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

    #part_number = 123
    #rmtype = "Rod"
    #rleng, rvol, perc_surf_area = fake_pyocc(
    #    pn=part_number,
    #    filename=step_file.filename,
    #    filepath=step_path,
    #    rmtype=rmtype
    #)
    #session['surf_detect'] = perc_surf_area

    # Get the material shape from the form
    material_shape = request.form.get('material_shape', '').strip()

    # call the pyocc function to compute geometry
    try:
        results_dict, faces = pyocc(step_path,material_shape)
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
    flash("Generate report: (placeholder)")
    return redirect(url_for('index'))

def load_and_convert_model():
    """
    Loads the ANN model from a Keras H5 file.
    """
    keras_path = os.path.join(os.path.dirname(__file__), "ann_model1.keras")
    
    if os.path.exists(keras_path):
        print("Loading model from Keras H5 file:", keras_path)
        return tf.keras.models.load_model(keras_path)
    else:
        raise FileNotFoundError(f"Keras model file not found at: {keras_path}")
    
@app.route('/reset_all', methods=['POST'])
def reset_all():
    # Clear out any session data
    session.clear()

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
        "Length": 130.0, "Vol": 250.0, "Act_Vol": 250.0,
        "VOL": 250.0, "VF": 0, "VID": 0, "VOD": 0, "SP": 0, "SGDH": 0, "HRC": 0
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

    # Get secondary processes (as a stripped list)
    secondary_procs = [proc.strip() for proc in request.form.getlist('secondary_processes')]

    # Define CSV file paths (ensure these files are in the correct locations)
    lb_file = "LabourRate.csv"
    rm_file = "RM Cost.csv"
    thr_file = "Threading Cost.csv"
    sp_file = "SecondaryProcesses.csv"

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

    session['machining_cost'] = total_cost
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


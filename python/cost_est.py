import sys
import os
import math
import random
import subprocess
import pandas as pd
import tensorflow as tf
import numpy as np
from tensorflow.python.keras.layers import Input, Dense
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.models import load_model
from flask import Flask, jsonify, render_template, request, redirect, url_for, flash, session
from scipy.stats import boxcox, yeojohnson

# PythonOCC
from OCC.Extend.DataExchange import read_step_file, write_stl_file
from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.Bnd import Bnd_Box
from OCC.Core.BRepBndLib import brepbndlib
from OCC.Display.SimpleGui import init_display

################################################################################
# Threading cost function
################################################################################
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")

def get_threading_cost(file_path, thread_type, material, size, connection, qty, offset, length):
    """
    Expects a CSV with columns (by index):
      0: Thread Type
      1: Material
      2: Size
      3: Connection
      4: QTY
      5: Base Price (float)
      6: Factor for one scenario (e.g. OFFSET or short)
      7: Factor for 'LONG PART >180'
      8: Factor for 'LONG PART >120'
    """
    thr_path = os.path.join(DATA_DIR, file_path)
    df = pd.read_csv(thr_path)
    
    # Create a mask for a case-insensitive comparison on the text fields
    # and for an integer comparison on the QTY column.
    mask = (
        df.iloc[:,0].astype(str).str.strip().str.lower() == thread_type.strip().lower()
    ) & (
        df.iloc[:,1].astype(str).str.strip().str.lower() == material.strip().lower()
    ) & (
        df.iloc[:,2].astype(str).str.strip().str.lower() == size.strip().lower()
    ) & (
        df.iloc[:,3].astype(str).str.strip().str.lower() == connection.strip().lower()
    ) & (
        df.iloc[:,4].astype(int) == int(qty)
    )
    
    row = df.loc[mask]
    
    if row.empty:
        print(f"No matching record for type={thread_type}, mat={material}, size={size}, conn={connection}, qty={qty}")
        return 0.0
    else:
        print("DEBUG: Matching row found:")
        print(row)
    
    #base_price_str = str(row.iloc[0,5]).replace("$", "").strip()
    #base_price_str = base_price_str.replace(",", "")
    #try:
    #    base_price = float(base_price_str)
    #except ValueError:
    #    base_price = 0.0

        # Directly convert the numeric value
    try:
        base_price = float(row.iloc[0,5])
    except (ValueError, TypeError):
        base_price = 0.0

    if not offset or offset == "off":
        return base_price
    else:
        # Determine the correct factor column based on length
        if length < 120:
            factor_col = 6
        elif 120 < length <= 180:
            factor_col = 8 # LONG PART >120" Factor column
        else:
            factor_col = 7 # LONG PART >180" Factor column

        factor_str = str(row.iloc[0, factor_col]).strip()
        try:
            factor_val = float(factor_str)
        except ValueError:
            factor_val = 1.0
        
        return base_price * factor_val


################################################################################
# Cost-related helper functions
################################################################################

def get_labour_rate(lb_file, region):
    """
    Reads the Labour Rate CSV and calculates total labour cost.
    Expects columns "Region", "Wage (per hour)", and "Number of Workers".
    """
    lb_path = os.path.join(DATA_DIR, lb_file)
    lb_df = pd.read_csv(lb_path)
    wage = lb_df.loc[lb_df["Region"] == region, "Wage (per hour)"].values[0]
    num_workers = lb_df.loc[lb_df["Region"] == region, "Number of Workers"].values[0]
    labor_cost = num_workers * wage
    return labor_cost

def get_rm_cost(rm_file, region, mat_spec):
    rm_path = os.path.join(DATA_DIR, rm_file)
    df = pd.read_csv(rm_path)
    row = df.loc[df["Mat_Spec"].astype(str).str.strip() == mat_spec.strip()]
    if row.empty:
        print("Mat Spec data not found.")
        return None
    region_column_map = {
        "China": 7,
        "India": 8,
        "Singapore": 9,
        "Malaysia": 10,
        "USA": 11,
        "UK": 12
    }
    if region in region_column_map:
        col_idx = region_column_map[region]
        try:
            cost = float(str(row.iloc[0, col_idx]).replace("$", "").strip())
        except Exception as e:
            print("Error converting cost:", e)
            cost = 0.0
        try:
            density = float(str(row.iloc[0, 3]).replace("$", "").strip())
        except Exception as e:
            print("Error converting density:", e)
            density = 0.0
        try:
            hrc = float(str(row.iloc[0, 5]).replace("$", "").strip())
        except Exception as e:
            print("Error converting HRC:", e)
            hrc = 0.0
        return cost, density, hrc
    else:
        print(f"Region '{region}' not found in predefined column indexes.")
        return None

def get_secondary_processes_cost(file_path, region, selected_processes):
    sp_path = os.path.join(DATA_DIR, file_path)
    df = pd.read_csv(sp_path)
    print("DEBUG columns:", df.columns)
    print("DEBUG first few rows:\n", df.head())
    region_map = {
        "China": 1,
        "India": 2,
        "Singapore": 3,
        "Malaysia": 4,
        "USA": 5,
        "UK": 6
    }
    if region not in region_map:
        return 0.0
    
    col_idx = region_map[region]
    total_proc_cost = 0.0
    for proc in selected_processes:
        print("DEBUG Checking process =", proc)
        prow = df.loc[df["Processes"] == proc]
        # print("DEBUG prow:\n", prow)
        if prow.empty:
            continue
        cost_str = prow.iloc[0, col_idx]
        #cost_str = cost_str.replace("$","").strip()
        try:
            val = float(cost_str)
        except ValueError:
            val = 0.0
        total_proc_cost += val
    return total_proc_cost

################################################################################
# load_options for form dropdowns
################################################################################
def load_options():
    def read_csv_column(csv_name, col_name):
        csv_path = os.path.join(DATA_DIR, csv_name)
        print("DEBUG read_csv_column ->", csv_path, "looking for column:", col_name)
        if not os.path.exists(csv_path):
            print("DEBUG file does NOT exist:", csv_path)
            return []
        df = pd.read_csv(csv_path, encoding="utf-8")
        print("DEBUG df columns for", csv_name, ":", df.columns.tolist())
        if col_name not in df.columns:
            print(f"DEBUG column '{col_name}' not in df.columns for {csv_name}")
            return []
        values = df[col_name].dropna().unique().tolist()
        print("DEBUG: Values in", col_name, ":", values)
        return sorted(values)

    material_specs   = read_csv_column("RM Cost.csv", "Mat_Spec")
    material_shapes  = read_csv_column("MaterialShape.csv", "Material Shape")
    surface_finishes = read_csv_column("SurfaceFinish.csv", "Surface Finish")
    regions          = read_csv_column("LabourRate.csv", "Region")
    secondary_procs  = read_csv_column("SecondaryProcesses.csv", "Processes")

    # For the big 40k row file (Threading Cost):
    threading_types  = read_csv_column("Threading Cost.csv", "Thread Type")
    threading_connection = read_csv_column("Threading Cost.csv", "Connection")
    threading_materials  = read_csv_column("Threading Cost.csv", "Material")
    threading_sizes      = read_csv_column("Threading Cost.csv", "Size")

    return {
        "material_specs": material_specs,
        "material_shapes": material_shapes,
        "surface_finishes": surface_finishes,
        "regions": regions,
        "secondary_processes": secondary_procs,
        "threading_types": threading_types,
        "threading_shapes": threading_connection,
        "threading_materials": threading_materials,
        "threading_sizes": threading_sizes
    }

##############################################
# Transformation Function for ANN Input
##############################################
def reapply_transformation(df):
    """
    Reapplies previously stored transformations (log, Box-Cox, Yeo-Johnson)
    to the input DataFrame.
    """
    df = df.apply(pd.to_numeric, errors='coerce')
    transformations = {
        'HRC': 'log',
        'VOL': 'boxcox',
        'SA': 'log',
        'VF': 'log',
        'VID': 'yeojohn',
        'SID': 'yeojohn',
        'VOD': 'log',
        'SOD': 'yeojohn',
        'NP': 'log',
        'SP': 'log',
        'NGDH': 'log',
        'VGDH': 'yeojohn',
        'SGDH': 'log',
        'SF': 'log'
    }
    lambdas = {'VOL': 0.16307387842022278, 'VID': 0.14303145565579908, 
               'SID': 0.24583201045350297, 'SOD': 0.21719326317942136, 
               'VGDH': -2.379650420244089, 'MH': -0.24585189810410324}
    df_transformed = df.copy()
    for col, transform in transformations.items():
        if col in df.columns:
            if transform == 'log':
                df_transformed[col] = np.log1p(df[col])
            elif transform == 'boxcox':
                if col in lambdas:
                    df_transformed[col] = boxcox(df[col], lmbda=lambdas[col])
                else:
                    print(f"Warning: Lambda missing for column {col}")
            elif transform == 'yeojohn':
                if col in lambdas:
                    df_transformed[col] = yeojohnson(df[col], lmbda=lambdas[col])[0]
                else:
                    print(f"Warning: Lambda missing for column {col}")
    return df_transformed

##############################################
# Consolidated Cost Estimation Function
##############################################
def cost_est(region, lb_file, rm_file, thr_file, sp_file, mat_spec, sf,
             offset, sec_proc, pyocc, thread_inputs):
    """
    Consolidates cost estimation using:
      - Labour cost from lb_file
      - Raw material details from rm_file (cost, density, HRC)
      - Threading cost from thr_file (using offset and part length)
      - Secondary process cost from sp_file
      - Geometry from pyocc (a dictionary with keys like "Length", "Act_Vol" or "Vol")
      - Surface finish (sf)
    
    Returns a tuple:
       (total_cost, lb_cost, rm_cost, sp_cost, thr_cost, machine_hours, vol, mass)
       
    Note: The ANN model ("ann_model1_new.keras") is used to predict machine hours based on
          features extracted from pyocc data.
    """
    print("DEBUG cost_est: region =", region, "mat_spec =", mat_spec, "sf =", sf)
    print("DEBUG cost_est: offset =", offset, "sec_proc =", sec_proc)
    print("DEBUG cost_est: pyocc =", pyocc)
    print("DEBUG cost_est: thread_inputs =", thread_inputs)

    # Debug: Verify the model file path
    model_path = os.path.join(MODELS_DIR, "ann_model1_new.keras")
    print(f"Model path: {os.path.abspath(model_path)}")
    print(f"Model exists: {os.path.exists(model_path)}")

    # 1) Load the model correctly
    try:
        model = tf.keras.models.load_model(model_path, compile=False)
        print("Model loaded successfully from ann_model1_new.keras")
    except Exception as e:
        print(f"Error loading model: {e}")
        raise

    # 2) Labour cost
    lb_cost = get_labour_rate(lb_file, region)
    print("DEBUG cost_est: lb_cost =", lb_cost)

    
    # 3) Raw material details
    details = get_rm_cost(rm_file, region, mat_spec)
    if details is None:
        rm_cost, density, hrc = 0, 0, 0
    else:
        rm_cost, density, hrc = details
    print("DEBUG cost_est: rm_cost =", rm_cost, "density =", density, "hrc =", hrc)
    
    # 4) Threading cost: iterate over each threading box in thread_inputs
    thr_cost = 0
    for thread in thread_inputs:
        # Each thread dict should have keys: "thr_type", "thr_mat", "thr_size", "thr_conn", "thr_qty"
        cost_val = get_threading_cost(thr_file,
                                      thread["thr_type"],
                                      thread["thr_mat"],
                                      thread["thr_size"],
                                      thread["thr_conn"],
                                      thread["thr_qty"],
                                      offset,
                                      pyocc.get("Length", 0))
        thr_cost += cost_val
    print("DEBUG cost_est: thr_cost =", thr_cost)

    # 5) Secondary process cost: sum cost for each selected process
    sp_cost = get_secondary_processes_cost(sp_file,region,sec_proc)
    #for proc in sec_proc:
    #    print("DEBUG Process = :" ,proc)
    #   sp_cost += get_secondary_processes_cost(sp_file, region, proc)
    print("DEBUG cost_est: sp_cost =", sp_cost)

    # 6) Geometry: Get part length and volume (use "Act_Vol" if available)
    leng = pyocc.get("Length", 0)
    vol = pyocc.get("Act_Vol", pyocc.get("Vol", 0))
    print("DEBUG cost_est: length =", leng, "volume =", vol)

    # 7) Prepare features for machine hours prediction using the ANN model
    x_vals = {"HRC": hrc}
    features = ["VOL", "VF", "VID", "VOD", "SP", "SGDH"]
    for f in features:
        x_vals[f] = pyocc.get(f, 0)
    x_vals["SF"] = sf
    x_df = pd.DataFrame([x_vals])
    
    
    # Normalize inputs and make predictions
    x_norm = reapply_transformation(x_df)
    x_array = x_norm.to_numpy()
    print("DEBUG: x_array to model =", x_array)
    machine_hours = model.predict(x_array)
    machine_hours = machine_hours.item()
    print("DEBUG: machine_hours from model =", machine_hours)
    
    # 8) Mass calculation (assuming density in lb/ft³ and vol in in³)
    mass = (density * (vol / 1728)) / 2.205 if density and vol else 0
    print("DEBUG cost_est: mass =", mass)
    
    # 9) Calculate total cost
    total_cost = lb_cost * machine_hours + rm_cost * mass + thr_cost + sp_cost
    print("DEBUG cost_est: total_cost (before return) =", total_cost)

    return total_cost, lb_cost, rm_cost, sp_cost, thr_cost, machine_hours, vol, mass
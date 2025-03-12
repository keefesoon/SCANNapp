# pyocc_helpers.py
import os
import math
import random
import subprocess
import pandas as pd

# Import OCC modules
from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.Bnd import Bnd_Box
from OCC.Core.BRepBndLib import brepbndlib
from OCC.Display.SimpleGui import init_display
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopoDS import topods
from OCC.Core.BRepGProp import brepgprop
from OCC.Core.GProp import GProp_GProps
from OCC.Core.BRepAdaptor import BRepAdaptor_Surface, BRepAdaptor_Curve
from OCC.Core.GeomAbs import GeomAbs_Cylinder, GeomAbs_Plane, GeomAbs_Circle
from OCC.Core.TopAbs import TopAbs_FACE, TopAbs_EDGE, TopAbs_REVERSED, TopAbs_FORWARD
from OCC.Core.BRep import BRep_Tool
from OCC.Core.gp import gp_Pnt

##############################################
# STEP File Reading & Basic Geometry Functions
##############################################
def run_occ_viewer(step_path):
    step_reader = STEPControl_Reader()
    status = step_reader.ReadFile(step_path)
    if status != 1:
        print(f"Failed to read STEP file: {step_path}")
        return
    step_reader.TransferRoots()
    shape = step_reader.Shape()

    display, start_display, _, _ = init_display()
    display.DisplayShape(shape, update=True)
    print(f"Opened PythonOCC viewer for: {step_path}")
    print("About to call start_display(), blocking until window is closed.")
    start_display()
    print("Returned from start_display() - window closed.")


def stepreader(filepath):
    """Read a STEP file and return its shape."""
    step_reader = STEPControl_Reader()
    status = step_reader.ReadFile(filepath)
    if status == 1:
        print("STEP file loaded successfully!")
    else:
        print("Failed to load STEP file.")
    step_reader.TransferRoots()
    return step_reader.Shape()

def FindLength(shape):
    """Return the length (in inches) along X-direction using the bounding box."""
    bbox = Bnd_Box()
    brepbndlib.Add(shape, bbox)
    min_x = bbox.CornerMin().X()
    max_x = bbox.CornerMax().X()
    return (max_x - min_x) / 25.4

def FindDiam(shape):
    """Return the diameter (in inches) along Z-direction using the bounding box."""
    bbox = Bnd_Box()
    brepbndlib.Add(shape, bbox)
    min_z = bbox.CornerMin().Z()
    max_z = bbox.CornerMax().Z()
    return (max_z - min_z) / 25.4

def CalcRawLength(length):
    """Round up length (in inches) with a buffer."""
    buffer = 0.25 if length < 150 else 0.375
    return math.ceil((length + buffer) / 0.125) * 0.125

def CalcRawDiam(length, diameter, dir=1):
    """
    Calculate the raw diameter.
    dir = 1 for Outer Diameter (OD), -1 for Inner Diameter (ID).
    Diameter is assumed in millimeters and converted to inches.
    """
    # Convert diameter from mm to inches
    diameter_in = diameter / 25.4
    buffer = 0
    if length >= 150:
        buffer = 0.5
    elif length >= 90:
        buffer = 0.5 if diameter_in >= 15 else 0.375
    else:
        if diameter_in > 20:
            return "ERR"
        elif diameter_in >= 15:
            buffer = 0.5
        elif diameter_in >= 10:
            buffer = 0.375
        else:
            buffer = 0.25
    if dir == 1:
        return math.ceil((diameter_in + buffer) / 0.25) * 0.25
    else:
        return math.floor((diameter_in - buffer) / 0.25) * 0.25

##############################################
# Surface Area and Volume Calculations
##############################################
def getprops(shape):
    """Return (volume in in³, surface area in in²) of the shape."""
    props = GProp_GProps()
    brepgprop.VolumeProperties(shape, props)
    volume = props.Mass() / (25.4 ** 3)
    brepgprop.SurfaceProperties(shape, props)
    surfarea = props.Mass() / (25.4 ** 2)
    return volume, surfarea

def RodVol(rdiam, rleng):
    """Volume of a rod given diameter (in inches) and length (in inches)."""
    from math import pi
    return pi * (rdiam / 2) ** 2 * rleng

def RodSA(rdiam, rleng):
    """Surface area of a rod."""
    from math import pi
    return 2 * pi * (rdiam / 2) ** 2 + pi * rdiam * rleng

def TubeVol(rod, rid, rleng):
    """Volume of a tube given outer and inner diameters (in inches) and length (in inches)."""
    from math import pi
    return (pi * (rod / 2) ** 2 - pi * (rid / 2) ** 2) * rleng

def TubeSA(rod, rid, rleng):
    """Surface area of a tube."""
    from math import pi
    return 2 * (pi * (rod / 2) ** 2 - pi * (rid / 2) ** 2) + pi * rod * rleng + pi * rid * rleng

def VolumeFaced(rod, rleng, length, rid=0):
    """Compute volume faced (difference between a cylinder of rod and an inner cylinder if provided)."""
    from math import pi
    if rid == 0:
        return pi * (rod / 2) ** 2 * (rleng - length)
    else:
        return (pi * (rod / 2) ** 2 - pi * (rid / 2) ** 2) * (rleng - length)

def CalcSurfaceArea(faces):
    """Calculate the total surface area (in in²) of a list of faces."""
    total_area = 0.0
    for face in faces:
        props = GProp_GProps()
        brepgprop.SurfaceProperties(face, props)
        total_area += props.Mass() / (25.4 ** 2)
    return total_area

def CalcVolOD(faces, rawOD):
    """Calculate volume based on outer surfaces (rawOD in inches)."""
    total_volume = 0.0
    rad_list = []
    ht_list = []
    from math import pi
    for face in faces:
        adaptor = BRepAdaptor_Surface(face)
        cylinder = adaptor.Cylinder()
        radius = cylinder.Radius() / 25.4
        bbox = Bnd_Box()
        brepbndlib.Add(face, bbox)
        min_x = bbox.CornerMin().X()
        max_x = bbox.CornerMax().X()
        height = (max_x - min_x) / 25.4
        if radius not in rad_list or height not in ht_list:
            rad_list.append(radius)
            ht_list.append(height)
    for i in range(len(rad_list)):
        total_volume += pi * ((rawOD / 2) ** 2) * ht_list[i] - pi * (rad_list[i] ** 2) * ht_list[i]
    return total_volume

##############################################
# Hole and Pocket Detection Functions
##############################################
def FindIR(shape, alignment_tolerance=0.1, gdhperc=0.3):
    """Find maximum and minimum inner radii based on reversed cylinder faces."""
    explorer = TopExp_Explorer(shape, TopAbs_FACE)
    max_hole_radius = 0
    cyl = []
    while explorer.More():
        face = topods.Face(explorer.Current())
        explorer.Next()
        adaptor = BRepAdaptor_Surface(face)
        if adaptor.GetType() == GeomAbs_Cylinder:
            if face.Orientation() == TopAbs_REVERSED:
                cylinder = adaptor.Cylinder()
                radius = cylinder.Radius()
                axis = cylinder.Axis()
                direction = axis.Direction()
                cyl.append(adaptor)
                if abs(direction.Z()) < alignment_tolerance:
                    if max_hole_radius == 0 or radius > max_hole_radius:
                        max_hole_radius = radius
    min_hole_radius = max_hole_radius
    for adaptor in cyl:
        cylinder = adaptor.Cylinder()
        radius = cylinder.Radius()
        if radius < min_hole_radius and radius > max_hole_radius * gdhperc:
            min_hole_radius = radius
    return max_hole_radius, min_hole_radius

def FindOR(shape):
    """Find the outer diameter from forward-oriented cylindrical faces."""
    explorer = TopExp_Explorer(shape, TopAbs_FACE)
    od = 0
    faces = []
    while explorer.More():
        face = topods.Face(explorer.Current())
        explorer.Next()
        adaptor = BRepAdaptor_Surface(face)
        if adaptor.GetType() == GeomAbs_Cylinder:
            if face.Orientation() == TopAbs_FORWARD:
                cylinder = adaptor.Cylinder()
                radius = cylinder.Radius()
                if radius > od:
                    od = radius
                    faces = [face]
    return od * 2, faces  # Return diameter in inches

def FindOuterSurfaces(shape):
    """Return a list of forward-oriented cylindrical faces as outer surfaces."""
    outer_surfaces = []
    explorer = TopExp_Explorer(shape, TopAbs_FACE)
    while explorer.More():
        face = topods.Face(explorer.Current())
        explorer.Next()
        adaptor = BRepAdaptor_Surface(face)
        if adaptor.GetType() == GeomAbs_Cylinder and face.Orientation() == TopAbs_FORWARD:
            outer_surfaces.append(face)
    return outer_surfaces

def FindMainHole(shape, max_ir, min_ir):
    """Return faces corresponding to the main (drill) hole based on inner radii."""
    explorer = TopExp_Explorer(shape, TopAbs_FACE)
    mainhole = []
    while explorer.More():
        face = topods.Face(explorer.Current())
        explorer.Next()
        adaptor = BRepAdaptor_Surface(face)
        if adaptor.GetType() == GeomAbs_Cylinder and face.Orientation() == TopAbs_REVERSED:
            cylinder = adaptor.Cylinder()
            radius = cylinder.Radius()
            if max_ir <= radius <= min_ir:
                mainhole.append(face)
    return mainhole

def CircularEdgeDetector(edge):
    """Determine if an edge is circular and approximates a quarter-circle (fillet)."""
    edge_adaptor = BRepAdaptor_Curve(edge)
    if edge_adaptor.GetType() == GeomAbs_Circle:
        start_param, end_param = edge_adaptor.FirstParameter(), edge_adaptor.LastParameter()
        arc_angle = abs(end_param - start_param)
        arc_angle_deg = arc_angle * 180 / math.pi
        if abs(arc_angle_deg - 90.0) < 0.1:
            return True
    return False

def findpocket(shape):
    """Return a list of faces that qualify as pockets based on fillet counts."""
    pocket_faces = []
    explorer = TopExp_Explorer(shape, TopAbs_FACE)
    while explorer.More():
        face = topods.Face(explorer.Current())
        explorer.Next()
        adaptor = BRepAdaptor_Surface(face)
        if adaptor.GetType() != GeomAbs_Plane:
            continue
        props = GProp_GProps()
        brepgprop.SurfaceProperties(face, props)
        edge_explorer = TopExp_Explorer(face, TopAbs_EDGE)
        fillet_count = 0
        while edge_explorer.More():
            edge = topods.Edge(edge_explorer.Current())
            edge_explorer.Next()
            if CircularEdgeDetector(edge):
                fillet_count += 1
        if fillet_count >= 4:
            pocket_faces.append(face)
    return pocket_faces

##############################################
# Grouping and Adjacent Faces
##############################################
def edgecompare(edge1, edge2):
    """Compare two edges for equality by length and endpoint positions."""
    props = GProp_GProps()
    brepgprop.LinearProperties(edge1, props)
    length1 = props.Mass()
    props = GProp_GProps()
    brepgprop.LinearProperties(edge2, props)
    length2 = props.Mass()
    if abs(length1 - length2) < 1e-6:
        curve1, fp1, lp1 = BRep_Tool.Curve(edge1)
        sp1 = gp_Pnt()
        ep1 = gp_Pnt()
        curve1.D0(fp1, sp1)
        curve1.D0(lp1, ep1)
        curve2, fp2, lp2 = BRep_Tool.Curve(edge2)
        sp2 = gp_Pnt()
        ep2 = gp_Pnt()
        curve2.D0(fp2, sp2)
        curve2.D0(lp2, ep2)
        if sp1.IsEqual(sp2, 1e-6) and ep1.IsEqual(ep2, 1e-6):
            return 1
        else:
            return 2
    else:
        return 0

def getnormal(face):
    """Return the normal direction of a face."""
    try:
        pln = BRepAdaptor_Surface(face).Plane()
    except:
        pln = face.Plane()
    return pln.Axis().Direction()

def similardirection(face1, face2, tolerance=0.1):
    """Return True if two faces have normals in similar directions."""
    normal1 = getnormal(face1)
    normal2 = getnormal(face2)
    dot_product = normal1.Dot(normal2)
    return abs(dot_product - 1.0) <= tolerance

def find_adjacent_faces(face, shape):
    """Return a list of faces adjacent to the given face (sharing an edge)."""
    adjacent_faces = []
    edge_explorer = TopExp_Explorer(face, TopAbs_EDGE)
    while edge_explorer.More():
        edge = topods.Edge(edge_explorer.Current())
        edge_explorer.Next()
        face_explorer = TopExp_Explorer(shape, TopAbs_FACE)
        while face_explorer.More():
            potential_face = topods.Face(face_explorer.Current())
            face_explorer.Next()
            if potential_face.IsEqual(face):
                continue
            edge_check_explorer = TopExp_Explorer(potential_face, TopAbs_EDGE)
            while edge_check_explorer.More():
                potential_edge = topods.Edge(edge_check_explorer.Current())
                edge_check_explorer.Next()
                if edgecompare(edge, potential_edge) == 1:
                    if potential_face not in adjacent_faces:
                        adjacent_faces.append(potential_face)
                    break
    return adjacent_faces

def CollectSimFaces(faces, shape):
    """Group faces that are adjacent and similarly oriented."""
    collection = []
    for face in faces:
        group = [face]
        similar_faces = find_adjacent_faces(face, shape)
        for sim_face in similar_faces:
            if similardirection(faces[0], sim_face):
                group.append(sim_face)
        collection.append(group)
    return collection

def CollSurfaceArea(collection):
    """Calculate total surface area (in in²) for each group in a collection."""
    total_area_mm2 = 0.0
    areas = []
    for group in collection:
        group_area = 0
        for face in group:
            props = GProp_GProps()
            brepgprop.SurfaceProperties(face, props)
            group_area += props.Mass() / (25.4 ** 2)
            total_area_mm2 += props.Mass()
        areas.append(group_area)
    total_area_in2 = total_area_mm2 / (25.4 ** 2)
    return total_area_in2, areas

def NewFindGunDrillHole(shape, minir, radius_tol=0.5, alignment_tol=0.001, ext_tol=1):
    """Group horizontal reversed cylinders as potential gun-drill holes."""
    gdh = []
    min_radius = 0
    max_radius = minir
    explorer = TopExp_Explorer(shape, TopAbs_FACE)
    while explorer.More():
        face = topods.Face(explorer.Current())
        explorer.Next()
        adaptor = BRepAdaptor_Surface(face)
        if adaptor.GetType() == GeomAbs_Cylinder and face.Orientation() == TopAbs_REVERSED:
            cylinder = adaptor.Cylinder()
            radius = cylinder.Radius()
            axis = cylinder.Axis()
            direction = axis.Direction()
            if min_radius < radius < max_radius and abs(direction.Z()) < alignment_tol:
                props = GProp_GProps()
                brepgprop.SurfaceProperties(face, props)
                centroid = props.CentreOfMass()
                grouped = False
                for group in gdh:
                    grp_axis = group['axis']
                    grp_radius = group['radius']
                    grp_faces = group['faces']
                    if abs(grp_axis.Direction().Dot(direction)) > 1 - alignment_tol and (abs(radius - grp_radius)/grp_radius) < radius_tol:
                        from OCC.Core.BRepExtrema import BRepExtrema_DistShapeShape
                        for face2 in grp_faces:
                            extrema = BRepExtrema_DistShapeShape(face, face2)
                            extrema.Perform()
                            if extrema.Value() < ext_tol:
                                group['faces'].append(face)
                                grouped = True
                                break
                if not grouped:
                    gdh.append({'axis': axis, 'radius': radius, 'centroid': centroid, 'faces': [face]})
    faces = []
    for group in gdh:
        if len(group['faces']) % 2 == 0:
            faces.append(group['faces'])
    return faces

def calculate_main_hole_volume(faces):
    """Calculate total volume of main holes from a list of faces."""
    total_volume = 0.0
    radii = []
    heights = []
    from math import pi
    for face in faces:
        adaptor = BRepAdaptor_Surface(face)
        cylinder = adaptor.Cylinder()
        radius = cylinder.Radius() / 25.4
        bbox = Bnd_Box()
        brepbndlib.Add(face, bbox)
        min_x = bbox.CornerMin().X()
        max_x = bbox.CornerMax().X()
        height = (max_x - min_x) / 25.4
        if radius not in radii or height not in heights:
            radii.append(radius)
            heights.append(height)
    for i in range(len(radii)):
        total_volume += pi * (radii[i] ** 2) * heights[i]
    return total_volume

##############################################
# Consolidated PythonOCC Function
##############################################
def pyocc(stepfile, rm):
    """
    Reads the STEP file and computes geometry parameters.
    rm: "ROD" or "Tube"
    Returns a tuple: (results_dict, faces)
    """
    shape = stepreader(stepfile)
    # Initialize variables (placeholders)
    det = length = vol = rmv = vf = vod = sod = vid = sid = sp = ngdh = sgdh = 0
    results = {}
    faces_collection = []
    
    od, orface = FindOR(shape)
    outfaces = FindOuterSurfaces(shape)
    maxir, minir = FindIR(shape)
    minid = minir * 2
    length = FindLength(shape)
    vol, sa = getprops(shape)
    rod = CalcRawDiam(length, od, 1)
    rleng = CalcRawLength(length)
    
    if rm == "ROD":
        rmv = RodVol(rod, rleng)
        rms = RodSA(rod, rleng)
        vf = VolumeFaced(rod, rleng, length)
    elif rm == "Tube":
        rid = CalcRawDiam(length, minid, -1)
        rmv = TubeVol(rod, rid, rleng)
        rms = TubeSA(rod, rid, rleng)
        vf = VolumeFaced(rod, rleng, length, rid)
    
    sod = CalcSurfaceArea(outfaces)
    vod = CalcVolOD(outfaces, rod)
    infaces = FindMainHole(shape, maxir, minir)
    vid = calculate_main_hole_volume(infaces)
    sid = CalcSurfaceArea(infaces)
    pkfaces = findpocket(shape)
    pockets = CollectSimFaces(pkfaces, shape)
    sp, splist = CollSurfaceArea(pockets)
    gdhfaces = NewFindGunDrillHole(shape, minir)
    ngdh = len(gdhfaces)
    sgdh, sgdhlist = CollSurfaceArea(gdhfaces)
    det = ((sod + sid + sp + sgdh) / sa) * 100 if sa != 0 else 0
    faces_collection.append(outfaces)
    faces_collection.append(infaces)
    faces_collection.extend(pkfaces)
    faces_collection.extend(gdhfaces)
    
    results = {
        "Det": det,
        "Length": length,
        "Vol": vol,
        "RMV": rmv,
        "VF": vf,
        "VOD": vod,
        "SOD": sod,
        "VID": vid,
        "SID": sid,
        "SP": sp,
        "NGDH": ngdh,
        "SGDH": sgdh
    }
    return results, faces_collection

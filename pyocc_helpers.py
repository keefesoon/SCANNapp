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

# Read STEP File

from OCC.Core.STEPControl import STEPControl_Reader

def run_occ_viewer(step_path):
    """Opens a PythonOCC viewer for the given STEP file (debug use)."""
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
    # STEP Reader object
    step_reader = STEPControl_Reader()
    # Read the file
    status = step_reader.ReadFile(filepath)
    
    # Check if the file was read successfully
    if status == 1:
        print("STEP file loaded successfully!")
    else:
        print("Failed to load STEP file.")
            
    # Transfer STEP file content to a shape
    step_reader.TransferRoots()
    shape = step_reader.Shape()
    return shape

# %%
from OCC.Core.Bnd import Bnd_Box
from OCC.Core.BRepBndLib import brepbndlib

def FindLength(shape):
    
    bbox = Bnd_Box()
    brepbndlib.Add(shape, bbox)

    min_x = bbox.CornerMin().X()
    max_x = bbox.CornerMax().X()
    height = (max_x - min_x)/25.4
    return height # inches

from OCC.Core.Bnd import Bnd_Box
from OCC.Core.BRepBndLib import brepbndlib

def FindDiam(shape):
    
    bbox = Bnd_Box()
    brepbndlib.Add(shape, bbox)

    min_z = bbox.CornerMin().Z()
    max_z = bbox.CornerMax().Z()
    diam = (max_z - min_z)/25.4
    return diam # inches

import math

def CalcRawLength(length): # inches
    buffer = 0
    if length < 150: buffer = 0.25
    else: buffer = 0.375
    return math.ceil((length + buffer)/0.125)*0.125

import math

def CalcRawDiam(length, diameter,dir=1): # dir = 1 for OD, dir = -1 for ID
    diameter = diameter/25.4
    buffer = 0
    if length >= 150: buffer = 0.5
    elif length >= 90:
        if diameter >= 15: buffer = 0.5
        else: buffer = 0.375
    else:
        if diameter > 20: return "ERR"
        elif diameter >= 15: buffer = 0.5
        elif diameter >= 10: buffer = 0.375
        else: buffer = 0.25
    if dir == 1: return math.ceil((diameter + dir*buffer)/0.25)*0.25
    else: return math.floor((diameter + dir*buffer)/0.25)*0.25

# %%
from OCC.Core.TopAbs import TopAbs_FACE, TopAbs_REVERSED
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopoDS import topods
from OCC.Core.GeomAbs import GeomAbs_Cylinder
from OCC.Core.BRepAdaptor import BRepAdaptor_Surface

def FindIR(shape, alignment_tolerance=0.1, gdhperc=0.3):
    """
    Identifies the main horizontal drill hole based on the largest horizontal cylinder.
    """
    explorer = TopExp_Explorer(shape, TopAbs_FACE)
    max_hole_radius, min_hole_radius = 0, 0
    cyl = []

    while explorer.More():
        face = topods.Face(explorer.Current())
        explorer.Next()

        # Analyze the face's surface type
        adaptor_surface = BRepAdaptor_Surface(face)
        geom_type = adaptor_surface.GetType()

        if geom_type == GeomAbs_Cylinder:
            if face.Orientation() == TopAbs_REVERSED:
                # Get cylinder properties
                cylinder = adaptor_surface.Cylinder()
                radius = cylinder.Radius()
                axis = cylinder.Axis()
                direction = axis.Direction()
                cyl.append(adaptor_surface)

                # Check if the cylinder is horizontal
                if abs(direction.Z()) < alignment_tolerance:  # Z-component close to 0
                    if max_hole_radius == 0:
                        max_hole_radius = radius
                    elif radius > max_hole_radius:  # Keep the largest horizontal cylinder
                        max_hole_radius = radius
    min_hole_radius = max_hole_radius
    for i in range(len(cyl)-1):
        cylinder = cyl[i].Cylinder()
        radius = cylinder.Radius()
        if radius < min_hole_radius and radius > max_hole_radius*gdhperc:
            min_hole_radius = radius

    return max_hole_radius, min_hole_radius

# %%
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopAbs import TopAbs_FACE, TopAbs_FORWARD
from OCC.Core.TopoDS import topods
from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
from OCC.Core.GeomAbs import GeomAbs_Cylinder

def FindOR(shape):
    explorer = TopExp_Explorer(shape, TopAbs_FACE)
    od = 0
    faces = [0]

    while explorer.More():
        face = topods.Face(explorer.Current())
        explorer.Next()

        # Analyze the face's surface type
        adaptor_surface = BRepAdaptor_Surface(face)
        geom_type = adaptor_surface.GetType()

        if geom_type == GeomAbs_Cylinder:
            if face.Orientation() == TopAbs_FORWARD:
                # Get cylinder properties
                cylinder = adaptor_surface.Cylinder()
                radius = cylinder.Radius()

                if radius > od:  # Keep the largest horizontal cylinder
                    faces.pop()
                    od = radius
                    faces.append(face)

    return od*2, faces # diameter in mm

# %%
# Detect Outer Surfaces

from OCC.Core.TopoDS import topods
from OCC.Core.BRep import BRep_Tool
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.GeomAdaptor import GeomAdaptor_Surface
from OCC.Core.GeomAbs import GeomAbs_Cylinder
from OCC.Core.TopAbs import TopAbs_FACE, TopAbs_FORWARD

def FindOuterSurfaces(shape):
    # Initialize variables
    outer_surfaces = []

    # Traverse faces in the shape
    explorer = TopExp_Explorer(shape, TopAbs_FACE)
    while explorer.More():
        face = topods.Face(explorer.Current())
        
        # Check if the face is cylindrical
        surface = BRep_Tool.Surface(face)
        adaptor = GeomAdaptor_Surface(surface)
        if surface and adaptor.GetType() == GeomAbs_Cylinder:
            # Check face orientation
            if face.Orientation() == TopAbs_FORWARD:
                outer_surfaces.append(face)
        explorer.Next()
    return outer_surfaces

# %%
# Calculate the volume faced

from math import pi

def VolumeFaced(rod,rleng,length,rid=0): # 0: rod, 1: tube
    if rid == 0: return pi*(rod/2)**2*(rleng-length)
    else: 
        res = (pi*(rod/2)**2 - pi*(rid/2)**2)*(rleng-length)
        return res

# Calculate Rod Vol

def RodVol(rdiam,rleng):
    return pi*(rdiam/2)**2*rleng

# Calculate Rod SA

def RodSA(rdiam,rleng):
    res = 2*pi*(rdiam/2)**2 + pi*rdiam*rleng
    return  res

def TubeVol(rod,rid,rleng):
    res = (pi*(rod/2)**2 - pi*(rid/2)**2)*rleng
    return res

# Calculate Rod SA

def TubeSA(rod,rid,rleng):
    res = 2*(pi*(rod/2)**2 - pi*(rid/2)**2) + pi*rod*rleng + pi*rid*rleng
    return  res

# %%
# Calculate Surface Area

from OCC.Core.GProp import GProp_GProps
from OCC.Core.BRepGProp import brepgprop

def CalcSurfaceArea(faces):
    surfArea = 0.0

    for face in faces:
        # Calculate surface area
        props = GProp_GProps()
        brepgprop.SurfaceProperties(face, props)
        area_mm2 = props.Mass()  # Surface area in mm²
        surfArea += area_mm2 / 25.4**2  # Accumulate area

    return surfArea 

# %%
from OCC.Core.Bnd import Bnd_Box
from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
from OCC.Core.BRepBndLib import brepbndlib
from math import pi

def CalcVolOD(faces,rawOD):
    total_volume = 0.0
    min_x, max_x = 0, 0
    rad, ht = [], []

    for face in faces:
        adaptor = BRepAdaptor_Surface(face)
        cylinder = adaptor.Cylinder()
        radius = cylinder.Radius()/25.4

        # Calculate the height (length of the cylinder) using bounding box
        bbox = Bnd_Box()
        brepbndlib.Add(face, bbox)

        min_x = bbox.CornerMin().X()
        max_x = bbox.CornerMax().X()
        # Compute the volume of the cylinder
        height = (max_x - min_x)/25.4
        
        if (radius in rad) and (height in ht): continue
        else:
            rad.append(radius)
            ht.append(height)

    for i in range(len(rad)):
        volume = pi*((rawOD/2)**2)*ht[i] - pi*(rad[i]**2)*ht[i]
        total_volume += volume

    return total_volume

# %%
def FindMainHole(shape, max_radius, min_radius, alignment_tolerance=0.1):

    mainhole = []
    
    # Explore faces in the shape
    explorer = TopExp_Explorer(shape, TopAbs_FACE)

    while explorer.More():
        face = topods.Face(explorer.Current())
        explorer.Next()

        # Analyze the face's surface type
        adaptor_surface = BRepAdaptor_Surface(face)
        geom_type = adaptor_surface.GetType()

        if geom_type == GeomAbs_Cylinder:
            if face.Orientation() == TopAbs_REVERSED:
                # Get cylinder properties (radius and orientation)
                cylinder = adaptor_surface.Cylinder()
                radius = cylinder.Radius()
                axis = cylinder.Axis()
                direction = axis.Direction()

                # Check if the radius is within the valid range
                if min_radius <= radius <= max_radius:
                    # Check if the cylinder is horizontal
                    if abs(direction.Z()) < alignment_tolerance:  # Z-component close to 0
                        mainhole.append(face)

    return mainhole

# %%
from OCC.Core.Bnd import Bnd_Box
from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
from OCC.Core.BRepBndLib import brepbndlib
from OCC.Core.GeomAbs import GeomAbs_Cylinder
from math import pi

def calculate_main_hole_volume(faces):
    total_volume = 0.0
    min_x, max_x = 0, 0
    rad, ht = [], []

    for face in faces:
        adaptor = BRepAdaptor_Surface(face)
        cylinder = adaptor.Cylinder()
        radius = cylinder.Radius()/25.4

        # Calculate the height (length of the cylinder) using bounding box
        bbox = Bnd_Box()
        brepbndlib.Add(face, bbox)

        min_x = bbox.CornerMin().X()
        max_x = bbox.CornerMax().X()
        # Compute the volume of the cylinder
        height = (max_x - min_x)/25.4
        
        if (radius in rad) and (height in ht): continue
        else:
            rad.append(radius)
            ht.append(height)

    for i in range(len(rad)):
        volume = pi * (rad[i]**2) * ht[i]
        total_volume += volume

    return total_volume

# %%
from OCC.Core.BRepAdaptor import BRepAdaptor_Curve
from OCC.Core.GeomAbs import GeomAbs_Circle
import math

# Detect circular edges

def CircularEdgeDetector(edge):
    edge_adaptor = BRepAdaptor_Curve(edge)
    if edge_adaptor.GetType() == GeomAbs_Circle:

        # Check if the circular edge represents a quarter-circle (90 degrees)
        # Get the start and end parameters of the circular edge
        start_param, end_param = edge_adaptor.FirstParameter(), edge_adaptor.LastParameter()

        # Calculate the angle of the arc
        arc_angle = abs(end_param - start_param)
        # Convert from parametric angle to degrees
        arc_angle_deg = arc_angle * 180 / math.pi

        # If the angle is close to 90 degrees, it's likely a fillet
        if abs(arc_angle_deg - 90.0) < 0.1:  # Tolerance for arc angle
            return True
        else: return False

# %%
# Find Pockets

from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopoDS import topods
from OCC.Core.BRepGProp import brepgprop
from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
from OCC.Core.GProp import GProp_GProps
from OCC.Core.GeomAbs import GeomAbs_Plane
from OCC.Core.TopAbs import TopAbs_FACE, TopAbs_EDGE

def findpocket(shape):
    # Initialize variables
    pocket_faces = []

    # Loop through faces and analyze geometry
    explorer = TopExp_Explorer(shape, TopAbs_FACE)
    while explorer.More():
        face = topods.Face(explorer.Current())
        explorer.Next()

        # Analyze the face's surface type
        adaptor_surface = BRepAdaptor_Surface(face)
        geom_type = adaptor_surface.GetType()

        # Skip non-planar surfaces
        if geom_type != GeomAbs_Plane:
            continue

        # Calculate surface area for debugging purposes
        props = GProp_GProps()
        brepgprop.SurfaceProperties(face, props)
        surface_area = props.Mass()

        # Check edges for fillets
        edge_explorer = TopExp_Explorer(face, TopAbs_EDGE)
        fillet_count = 0

        while edge_explorer.More():
            edge = topods.Edge(edge_explorer.Current())
            edge_explorer.Next()

            if CircularEdgeDetector(edge):  # Tolerance for arc angle
                fillet_count += 1

        # Faces with insufficient fillets
        if fillet_count < 4:
            continue

        # If all checks pass, add the face to the pocket list
        pocket_faces.append(face)
    return pocket_faces


# %%
# Compare Edges

from OCC.Core.BRep import BRep_Tool
from OCC.Core.gp import gp_Pnt

def edgecompare(edge1,edge2):
    props = GProp_GProps()
    brepgprop.LinearProperties(edge1, props)
    length1 = props.Mass()

    props = GProp_GProps()
    brepgprop.LinearProperties(edge2, props)
    length2 = props.Mass()

    if length1 == length2:
        curve1, first_param1, last_param1 = BRep_Tool.Curve(edge1)
        sp1 = gp_Pnt()
        ep1 = gp_Pnt()
        curve1.D0(first_param1, sp1)
        curve1.D0(last_param1, ep1)

        curve2, first_param2, last_param2 = BRep_Tool.Curve(edge1)
        sp2 = gp_Pnt()
        ep2 = gp_Pnt()
        curve2.D0(first_param2, sp2)
        curve2.D0(last_param2, ep2)

        if (sp1.X() == sp2.X()) and (sp1.Y() == sp2.Y()) and (sp1.Z() == sp2.Z()) and (ep1.X() == ep2.X()) and (ep1.Y() == ep2.Y()) and (ep1.Z() == ep2.Z()):
            return 1
        else:
            return 2
    else: return 0


# %%
# Check Face Directions

from OCC.Core.BRepAdaptor import BRepAdaptor_Surface

def getnormal(face):

    try:
        pln = BRepAdaptor_Surface(face).Plane()
    except:
        pln = face.Plane()
    normal = pln.Axis().Direction()

    return normal

def similardirection(face1, face2, tolerance=0.1):
    
    # Get normals for both faces
    normal1 = getnormal(face1)
    normal2 = getnormal(face2)
    
    # Compute the dot product of the normals
    dot_product = normal1.Dot(normal2)
    
    # Check if the dot product is close to 1
    return abs(dot_product - 1.0) <= tolerance


# %%
# Get Adjacent Faces

from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopoDS import topods
from OCC.Core.TopAbs import TopAbs_EDGE, TopAbs_FACE

def find_adjacent_faces(face, shape):
    """
    Find adjacent faces to a given face by analyzing shared edges.
    """
    adjacent_faces = []

    # Explore edges of the given face
    edge_explorer = TopExp_Explorer(face, TopAbs_EDGE)
    while edge_explorer.More():
        edge = topods.Edge(edge_explorer.Current())
        edge_explorer.Next()

        # Explore all faces in the shape to find shared edges
        face_explorer = TopExp_Explorer(shape, TopAbs_FACE)
        while face_explorer.More():
            potential_face = topods.Face(face_explorer.Current())
            face_explorer.Next()

            # Skip the original face
            if potential_face.IsEqual(face):
                continue

            # Check if the edge is part of this potential face
            edge_check_explorer = TopExp_Explorer(potential_face, TopAbs_EDGE)
            while edge_check_explorer.More():
                potential_edge = topods.Edge(edge_check_explorer.Current())
                edge_check_explorer.Next()

                # If the edge is shared, mark the face as adjacent
                if edgecompare(edge,potential_edge):
                    if potential_face not in adjacent_faces:
                        adjacent_faces.append(potential_face)
                    break

    return adjacent_faces


# %%
# Collect Similar Faces

def CollectSimFaces(faces, shape):
    collection = []
    for face in faces:
        pocket = []
        pocket.append(face)
        similar_faces = find_adjacent_faces(face, shape)
        for sim_face in similar_faces:
            # Analyze adjacent face geometry (optional)
            sim_adaptor_surface = BRepAdaptor_Surface(sim_face)
            if sim_adaptor_surface.GetType() == GeomAbs_Plane:
                if similardirection(faces[0],sim_adaptor_surface):
                    pocket.append(sim_face)
        collection.append(pocket)
    return collection


# %%
# Calculate Surface Area of Pockets

def CollSurfaceArea(collection):
    total_area_mm2 = 0.0 # Total area in mm2
    sa = []

    for faces in collection:
        x = 0
        for face in faces:
            # Calculate surface area
            props = GProp_GProps()
            brepgprop.SurfaceProperties(face, props)
            area_mm2 = props.Mass()  # Surface area in mm²
            x += area_mm2/25.4**2
            total_area_mm2 += area_mm2  # Accumulate area
        sa.append(x)

    total_area_in2 = total_area_mm2/25.4**2
    return total_area_in2, sa

# %%
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopoDS import topods
from OCC.Core.TopAbs import TopAbs_FACE, TopAbs_REVERSED
from OCC.Core.GProp import GProp_GProps
from OCC.Core.gp import gp_Pnt
from OCC.Core.BRepGProp import brepgprop
from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
from OCC.Core.BRepExtrema import BRepExtrema_DistShapeShape
from OCC.Core.GeomAbs import GeomAbs_Cylinder

def NewFindGunDrillHole(shape, min_hole_radius, radius_tol=0.5, alignment_tol=0.001, ext_tol = 1):

    gdh = []

    # Calculate dynamic radius range for gun-drilled holes
    min_radius = 0
    max_radius = min_hole_radius

    # Explore faces in the shape
    explorer = TopExp_Explorer(shape, TopAbs_FACE)

    while explorer.More():
        face = topods.Face(explorer.Current())
        explorer.Next()

        # Analyze the face's surface type
        adaptor_surface = BRepAdaptor_Surface(face)
        geom_type = adaptor_surface.GetType()

        if geom_type == GeomAbs_Cylinder:
            if face.Orientation() == TopAbs_REVERSED:
                # Get cylinder properties (radius and orientation)
                
                cylinder = adaptor_surface.Cylinder()
                radius = cylinder.Radius()
                axis = cylinder.Axis()
                direction = axis.Direction()

                # Check if the radius is within the valid range
                if min_radius < radius < max_radius:
                    # Check if the cylinder is horizontal
                    if abs(direction.Z()) < alignment_tol:  # Z-component close to 0                    
                        # Group cylinders with similar radii and alignment
                        # if not CircularEdgeDetection(face): no way to distinguish between fillet and cylinder
                        props = GProp_GProps()
                        brepgprop.SurfaceProperties(face, props)
                        centroid = props.CentreOfMass()
                        # print(f"X: {axis.Direction().X()}, Y: {axis.Direction().Y()}, Z: {axis.Direction().Z()}")
                        # print(f"X: {centroid.X()}, Y: {centroid.Y()}, Z: {centroid.Z()}")

                        grouped = False
                        for group in gdh:
                            grp_axis = group['axis']
                            grp_radius = group['radius']
                            grp_centroid = group['centroid']
                            grp_faces = group['faces']

                            # Check if the current cylinder aligns with the group's axis and radius
                            if abs(grp_axis.Direction().Dot(direction)) > 1 - alignment_tol and (abs(radius - grp_radius)/grp_radius) < radius_tol:
                                # group['faces'].append(face)
                                # grouped = True
                                # break
                                for face2 in grp_faces: # Compare min distance between each face in the group
                                    extrema = BRepExtrema_DistShapeShape(face, face2) # loads faces
                                    extrema.Perform() 
                                    if extrema.Value() < ext_tol: # gets min distance and make sure less than 1mm
                                        group['faces'].append(face)
                                        grouped = True
                                        break
                        if not grouped:
                            # Create a new group for this cylinder
                            gdh.append({'axis': axis, 'radius': radius, 'centroid': centroid, 'faces': [face]})

    faces = []
    for group in gdh:
        grp_faces = group['faces']
        if len(grp_faces)%2 == 0:
            faces.append(grp_faces)

    return faces

# %%
def getprops(shape):
    # Initialize the GProp_GProps object to hold the properties
    props = GProp_GProps()
    
    # Compute the geometric properties of the shape
    brepgprop.VolumeProperties(shape, props)
        # Get the volume from the properties
    volume = props.Mass()/25.4**3

    # Compute the geometric properties of the shape
    brepgprop.SurfaceProperties(shape, props)
        # Get the volume from the properties
    surfarea = props.Mass()/25.4**2
    
    return volume, surfarea

# %%
# Inputs: Need to read the STEP file, Shape of the Raw Part
# Outputs: Total 9
# Need to output the 6 predictors as a list: 
# Leng, Vol, RV, VF, VOD (SID), VID (SID), SP, SGDH (NGDH)
# Need to output the volume and leng of the object and % of surfaces scanned
# aggregate faces from cylindrical and pockets.

# Raw Volume: 
# If Rod: FindOR > OD > Length > ROD > rleng > RV > VF
# If Tube: FindOR > OD > FindIR > ID > Length > ROD > RID > rleng > TV
# SOD + VOD: FindOR > Outer Faces > SOD > VOD
# SID + VID: FindIR > FindMainHole (min,max) > Inner Faces > SID > VID
# SP: findpk > findadj > SP
# NGDH + SGDH: Find IR > (0,min) > NGDH


# Future Optimization:
# Scan cylindrical surfaces and split between outer and inner
# Split cylindrical surfaces once more between (0,min) for gdh and [min,max] for ID
# Can use same outer surfaces to determine SOD and VOD
# Can use same inner surfaces to determine SID and VID
# Can use same faces for SGDH and NGDH

def pyocc(stepfile,rm):
    shape = stepreader(stepfile) #stepfile name in the same folder
    det, length, vol, rmv, vf, vod, sod, vid, sid, sp, ngdh, sgdh = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    results = {}
    faces = []
    
    #outer diameter
    od,orface = FindOR(shape)
    outfaces = FindOuterSurfaces(shape)
    
    #inner radius
    maxir, minir = FindIR(shape)
    minid = minir*2

    #get bounding box info
    length = FindLength(shape)
    vol, sa = getprops(shape)

    # raw dimension
    rod = CalcRawDiam(length,od,1)
    rleng = CalcRawLength(length)

    if rm == "ROD":
        rmv = RodVol(rod,rleng)
        rms = RodSA(rod,rleng)
        vf = VolumeFaced(rod,rleng,length)
    elif rm == "TUBE":
        rid = CalcRawDiam(length,minid,-1)
        rmv = TubeVol(rod,rid,rleng)
        rms = TubeSA(rod,rid,rleng)
        vf = VolumeFaced(rod,rleng,length,rid)
            
    sod = CalcSurfaceArea(outfaces)
    vod = CalcVolOD(outfaces,rod)
    infaces = FindMainHole(shape,maxir,minir)
    vid = calculate_main_hole_volume(infaces)
    sid = CalcSurfaceArea(infaces)
    pkfaces = findpocket(shape)
    pockets = CollectSimFaces(pkfaces,shape)
    sp, splist = CollSurfaceArea(pockets)
    gdhfaces = NewFindGunDrillHole(shape,minir)
    ngdh = len(gdhfaces)
    sgdh, sgdhlist = CollSurfaceArea(gdhfaces)
    det = ((sod+sid+sp+sgdh)/sa)*100
    faces.append(outfaces)
    faces.append(infaces)
    faces.extend(pkfaces)
    faces.extend(gdhfaces)

    results = {"Det":det, "Length":length, "Act_Vol":vol, "VOL":rmv, "VF":vf, "VID":vid, "SID":sid, "VOD":vod, "SOD":sod, "SP":sp, "NGDH":ngdh, "SGDH":sgdh}
    return results, faces
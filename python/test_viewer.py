#!/usr/bin/env pythonw
from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Display.SimpleGui import init_display
import sys
import os
os.environ['QT_MAC_WANTS_LAYER'] = '1'
os.environ['OBJC_DISABLE_INITIALIZE_FORK_SAFETY'] = 'YES'
os.environ['LIBGL_ALWAYS_SOFTWARE'] = '1'

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
    print("Viewer is starting...")
    start_display()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: pythonw test_viewer.py /path/to/file.step")
        sys.exit(1)
    run_occ_viewer(sys.argv[1])

import sys
from pyocc_helpers import run_occ_viewer

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python viewer.py /path/to/file.step")
        sys.exit(1)
    run_occ_viewer(sys.argv[1])


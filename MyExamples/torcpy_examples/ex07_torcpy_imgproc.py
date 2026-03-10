"""
Late switch from SPMD to master-worker and parallel image processing
"""

import os
import sys
import time
from PIL import Image
import torcpy as torc

# Define the absolute path to the images folder
# Assuming the script is run from /home/george-tsavos/CEID/HPC_Lab/Thesis/MyExamples/torcpy_examples/
# The images folder is ../images/ from this script's location
IMAGES_DIR = os.path.join(os.path.dirname(__file__), "..", "images")
files = []


def get_files(path):
    all_files = []
    for dirpath, dirnames, filenames in os.walk(path):
        for f in filenames:
            if f.endswith(('.ppm', '.jpg', '.jpeg', '.JPEG', '.png')):
                all_files.append(os.path.join(dirpath, f))

    return sorted(all_files)


def work(i):
    global files
    f = files[i]
    with Image.open(f) as im:
        im = im.resize((32,32))
        # do something more, e.g., apply a filter or convert to another format
        # im.save(f"processed_{os.path.basename(f)}") # Example of saving
    return None


def main():
    global files

    # SPMD execution: torc_py and MPI initialization
    torc.init()

    # Common global initialization takes place here
    # We'll hardcode the images directory since it's known
    files = get_files(IMAGES_DIR)

    if not files:
        if torc.node_id() == 0:
            print("No image files found in the specified directory:", IMAGES_DIR)
        torc.shutdown()
        return

    # Switching to master-worker
    torc.launch(None)

    t0 = time.time()
    _ = torc.map(work, range(len(files)))
    t1 = time.time()

    if torc.node_id() == 0:
        print("=" * 60)
        print("Framework: torcpy (Image Processing)")
        print(f"Elapsed time: {t1-t0:.4f} seconds")
        print("=" * 60)

    torc.shutdown()


if __name__ == "__main__":
    main()

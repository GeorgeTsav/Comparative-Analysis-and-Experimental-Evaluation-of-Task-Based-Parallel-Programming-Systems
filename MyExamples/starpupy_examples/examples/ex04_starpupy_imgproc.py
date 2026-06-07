"""
Late switch from SPMD to master-worker and parallel image processing with StarPU Python
"""

import os
import sys
import time
from PIL import Image
import asyncio
import starpu

# Define the absolute path to the images folder
# Assuming the script is run from /home/george-tsavos/CEID/HPC_Lab/Thesis/MyExamples/startpupy_examples/
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


def work_task(f):
    # This function will be executed as a StarPU task
    with Image.open(f) as im:
        im = im.resize((32, 32))
        # do something more, e.g., apply a filter or convert to another format
        # im.save(f"processed_{os.path.basename(f)}") # Example of saving
    return None


async def main():
    global files

    starpu.init()

    files = get_files(IMAGES_DIR)

    if not files:
        print("No image files found in the specified directory:", IMAGES_DIR)
        starpu.shutdown()
        return

    t0 = time.time()

    # Submit tasks to StarPU
    tasks = []
    for f in files:
        task = starpu.task_submit()(work_task, f)
        tasks.append(task)

    # Wait for all tasks to complete
    for task in tasks:
        await task

    t1 = time.time()

    print("=" * 60)
    print("Framework: StarPU Python (Image Processing)")
    print(f"Elapsed time: {t1-t0:.4f} seconds")
    print("=" * 60)

    starpu.shutdown()


if __name__ == "__main__":
    asyncio.run(main())

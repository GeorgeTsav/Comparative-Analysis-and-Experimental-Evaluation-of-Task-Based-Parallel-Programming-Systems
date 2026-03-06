"""
(C) Copyright IBM Corporation 2019
All rights reserved. This program and the accompanying materials
are made available under the terms of the Eclipse Public License v1.0
which accompanies this distribution, and is available at
http://www.eclipse.org/legal/epl-v10.html
"""

"""
Task parallelism with map
With execution time measurement for framework comparison
"""
import time
import torcpy as torc


def work(x):
    return x * x


def main():
    data = range(10)
    
    # Record start time
    t0 = time.time()
    
    results = torc.map(work, data, chunksize=1)
    
    # Record end time
    t1 = time.time()
    
    print(results)
    
    elapsed = t1 - t0
    print("=" * 60)
    print("Framework: torcpy (Map)")
    print(f"Number of tasks: {len(data)}")
    print(f"Elapsed time: {elapsed:.4f} seconds")
    print("=" * 60)


if __name__ == '__main__':
    torc.start(main)

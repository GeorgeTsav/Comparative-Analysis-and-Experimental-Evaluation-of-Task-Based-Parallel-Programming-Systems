"""
(C) Copyright IBM Corporation 2019
All rights reserved. This program and the accompanying materials
are made available under the terms of the Eclipse Public License v1.0
which accompanies this distribution, and is available at
http://www.eclipse.org/legal/epl-v10.html
"""

"""
Master-worker demo, adapted from torc-lite
With execution time measurement for framework comparison
"""
import time
import torcpy as torc
import threading


def work(x):
    time.sleep(1)
    y = x**2
    print("work inp={:.3f}, out={:.3f} ...on node {:d} worker {} thread {}".format(x, y, torc.node_id(),
                                                                                  torc.worker_id(),
                                                                                  threading.get_ident()), flush=True)
    return y


def main():
    # tr.print_diff()

    ntasks = 4  # 100000
    sequence = range(1, ntasks + 1)

    # Record start time
    t0 = torc.gettime()
    tasks = []
    for i in sequence:
        task = torc.submit(work, i)
        tasks.append(task)
    torc.wait()
    # Record end time
    t1 = torc.gettime()

    for t in tasks:
        print("Received: {}^2={:.3f}".format(t.input(), t.result()))

    del tasks

    # tr.print_diff()

    elapsed = t1 - t0
    print("=" * 60)
    print("Framework: torcpy (Master-Worker)")
    print(f"Number of tasks: {ntasks}")
    print(f"Elapsed time: {elapsed:.4f} seconds")
    print("=" * 60)


if __name__ == '__main__':
    torc.start(main)

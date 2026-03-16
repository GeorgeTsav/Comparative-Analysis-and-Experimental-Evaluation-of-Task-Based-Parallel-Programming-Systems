"""
Monitor example for torcpy
- Submits multiple CPU-bound dummy tasks (busy loop) to torcpy
- Use `htop` while it runs to observe processes/threads and CPU usage
"""

import time
import os
import torcpy as torc


def busy_work(seconds):
    # CPU-bound busy loop to generate measurable load
    t0 = time.time()
    x = 1
    while time.time() - t0 < seconds:
        # cheap integer ops to keep CPU busy
        x = ((x << 1) ^ (x >> 3)) & 0xFFFFFFFF
    return x


def main():
    print(f"PID: {os.getpid()}, ntasks=8, sec=8.0")
    print("Start submitting tasks...")

    t0 = time.time()
    tasks = []
    for i in range(8):
        tasks.append(torc.submit(busy_work, 8.0))

    # Wait for completion
    torc.wait()
    t1 = time.time()

    # Collect results (force materialization)
    for i, t in enumerate(tasks):
        try:
            _ = t.result()
        except Exception as e:
            print(f"task {i} error: {e}")

    elapsed = t1 - t0
    print("=" * 60)
    print("Framework: torcpy (Monitor dummy workload)")
    print(f"Submitted tasks: 8")
    print(f"Work per task (s): 8.0")
    print(f"Elapsed time: {elapsed:.4f} seconds")
    print("=" * 60)


if __name__ == '__main__':
    torc.start(main)

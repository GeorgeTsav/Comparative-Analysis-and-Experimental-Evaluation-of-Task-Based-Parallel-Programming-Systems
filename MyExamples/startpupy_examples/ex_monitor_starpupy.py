"""
Monitor example for StarPU Python
- Submits multiple CPU-bound dummy tasks (busy loop) via StarPU
- Use `htop` while it runs to observe threads/processes and CPU usage
"""

import time
import asyncio
import os
from starpu import starpu


def busy_work(seconds):
    # CPU-bound busy loop to generate measurable load
    t0 = time.time()
    x = 1
    while time.time() - t0 < seconds:
        # cheap integer ops to keep CPU busy
        x = ((x << 1) ^ (x >> 3)) & 0xFFFFFFFF
    return x


async def main():
    try:
        starpu.init()
    except Exception as e:
        print(f"StarPU initialization error: {e}")
        return

    print(f"PID: {os.getpid()}, ntasks=8, sec=8.0")
    print("Start submitting tasks...")


    t0 = time.time()
    futures = []
    for i in range(8):
        fut = starpu.task_submit()(busy_work, 8.0)
        futures.append(fut)

    # Wait for completion and collect results (force materialization)
    results = []
    for fut in futures:
        res = await fut
        results.append(res)

    t1 = time.time()
    elapsed = t1 - t0

    print("=" * 60)
    print("Framework: StarPU Python (Monitor dummy workload)")
    print(f"Submitted tasks: 8")
    print(f"Work per task (s): 8.0")
    print(f"Elapsed time: {elapsed:.4f} seconds")
    print("=" * 60)

    starpu.shutdown()


if __name__ == '__main__':
    try:
        asyncio.run(main())
    except Exception as e:
        print(f"Error: {e}")

"""
StarPU Python equivalent of kwargs example
Ported from torcpy example
With execution time measurement for framework comparison
"""
import time
import asyncio
from starpu import starpu


def work(x, xx):
    time.sleep(1)
    y = x**2
    print(f"work inp={x:.3f}, extra_param={xx}, out={y:.3f}")
    return y


async def main():
    try:
        starpu.init()
    except Exception as e:
        print(f"StarPU initialization error: {e}")
        return

    ntasks = 4
    sequence = range(1, ntasks + 1)

    # Record start time
    t0 = time.time()
    
    futures = []
    for i in sequence:
        # StarPU task_submit passes positional arguments directly
        fut = starpu.task_submit()(work, i, i+1)
        futures.append((i, fut))
    
    # Wait for all tasks and collect results
    results = []
    for idx, fut in futures:
        result = await fut
        results.append((idx, result))
    
    # Record end time
    t1 = time.time()

    # Print results
    for idx, result in results:
        print(f"Received: {idx}^2={result:.3f}")

    elapsed = t1 - t0
    print("=" * 60)
    print("Framework: StarPU Python (Master-Worker with additional parameters)")
    print(f"Number of tasks: {ntasks}")
    print(f"Elapsed time: {elapsed:.4f} seconds")
    print("=" * 60)

    starpu.shutdown()


if __name__ == '__main__':
    try:
        asyncio.run(main())
    except Exception as e:
        print(f"Error: {e}")

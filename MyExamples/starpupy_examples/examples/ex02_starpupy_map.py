"""
StarPU Python equivalent of Map example
Ported from torcpy example
With execution time measurement for framework comparison
"""
import time
import asyncio
from starpu import starpu


def work(x):
    return x * x


async def main():
    try:
        starpu.init()
    except Exception as e:
        print(f"StarPU initialization error: {e}")
        return

    data = range(10)
    
    # Record start time
    t0 = time.time()
    
    # Submit all tasks
    futures = []
    for d in data:
        fut = starpu.task_submit()(work, d)
        futures.append(fut)
    
    # Wait for all tasks and collect results
    results = []
    for fut in futures:
        result = await fut
        results.append(result)
    
    # Record end time
    t1 = time.time()

    print(results)
    
    elapsed = t1 - t0
    print("=" * 60)
    print("Framework: StarPU Python (Map equivalent)")
    print(f"Number of tasks: {len(data)}")
    print(f"Elapsed time: {elapsed:.4f} seconds")
    print("=" * 60)

    starpu.shutdown()


if __name__ == '__main__':
    try:
        asyncio.run(main())
    except Exception as e:
        print(f"Error: {e}")

"""
StarPU Python equivalent of Recursive Fibonacci
Ported from torcpy example
With execution time measurement for framework comparison
"""
import time
import asyncio
from starpu import starpu


async def fib(n):
    if n == 0:
        return 0
    elif n == 1:
        return 1
    else:
        # Use direct async recursion for all cases
        # StarPU's async scheduler handles the parallelization
        result1 = await fib(n - 1)
        result2 = await fib(n - 2)
        return result1 + result2


async def main():
    try:
        starpu.init()
    except Exception as e:
        print(f"StarPU initialization error: {e}")
        return

    # Record start time
    t0 = time.time()
    n = 35
    result = await fib(n)
    # Record end time
    t1 = time.time()

    print("fib({}) = {}".format(n, result))
    
    elapsed = t1 - t0
    print("=" * 60)
    print("Framework: StarPU Python (Recursive Fibonacci)")
    print(f"Fibonacci input: {n}")
    print(f"Elapsed time: {elapsed:.4f} seconds")
    print("=" * 60)

    starpu.shutdown()


if __name__ == '__main__':
    try:
        asyncio.run(main())
    except Exception as e:
        print(f"Error: {e}")

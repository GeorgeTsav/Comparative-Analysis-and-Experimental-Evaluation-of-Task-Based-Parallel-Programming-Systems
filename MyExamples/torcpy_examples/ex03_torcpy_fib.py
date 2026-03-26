"""
(C) Copyright IBM Corporation 2019
All rights reserved. This program and the accompanying materials
are made available under the terms of the Eclipse Public License v1.0
which accompanies this distribution, and is available at
http://www.eclipse.org/legal/epl-v10.html
"""

"""
Recursive fibonacci
With execution time measurement for framework comparison
"""
import time
import torcpy as torc


def fib(n):
    if n == 0:
        result = 0
    elif n == 1:
        result = 1
    else:
        n_1 = n - 1
        n_2 = n - 2
        if n < 30:
            result1 = fib(n_1)
            result2 = fib(n_2)
            result = result1 + result2
        else:
            task1 = torc.submit(fib, n_1)
            task2 = torc.submit(fib, n_2)
            torc.wait()
            result = task1.result() + task2.result()

    return result


def main():
    # Record start time
    t0 = time.time()
    n = 35
    result = fib(n)
    # Record end time
    t1 = time.time()

    print("fib({}) = {}".format(n, result))
    
    elapsed = t1 - t0
    print("=" * 60)
    print("Framework: torcpy (Recursive Fibonacci)")
    print(f"Fibonacci input: {n}")
    print(f"Elapsed time: {elapsed:.4f} seconds")
    print("=" * 60)


if __name__ == '__main__':
    torc.start(main)

"""
StarPU Python equivalent of CMA-ES example
Ported from torcpy example
With execution time measurement for framework comparison
"""
# pip install cma
import cma
import time
import asyncio
from starpu import starpu


def rosenbrock(x):
    """Rosenbrock test objective function"""
    n = len(x)
    if n < 2:
        raise ValueError('dimension must be greater one')
    return sum(100 * (x[i] ** 2 - x[i + 1]) ** 2 + (x[i] - 1) ** 2 for i in range(n - 1))


async def main():
    try:
        starpu.init()
    except Exception as e:
        print(f"StarPU initialization error: {e}")
        return

    popsize = 8
    maxfevals = 320
    es = cma.CMAEvolutionStrategy(2 * [0], 0.5, {'popsize': popsize, 'maxfevals': maxfevals, 'verb_disp': 0, 'seed': 3})
    
    # Record start time
    t0 = time.time()

    while not es.stop():
        solutions = es.ask()
        
        # Submit all evaluations as StarPU tasks
        futures = [starpu.task_submit()(rosenbrock, s) for s in solutions]
        
        # Await results
        results = []
        for fut in futures:
            result = await fut
            results.append(result)
            
        es.tell(solutions, results)
        es.logger.add(es)  # write data to disc to be plotted
        es.disp()

    # Record end time
    t1 = time.time()

    print(f"Final solution: {es.result[0]}")
    print(f"Objective value: {es.result[1]}")
    
    elapsed = t1 - t0
    print("=" * 60)
    print("Framework: StarPU Python (CMA-ES Optimization)")
    print(f"Population size: {popsize}")
    print(f"Max function evaluations: {maxfevals}")
    print(f"Elapsed time: {elapsed:.4f} seconds")
    print("=" * 60)

    # cma.plot()

    starpu.shutdown()


if __name__ == '__main__':
    try:
        asyncio.run(main())
    except Exception as e:
        print(f"Error: {e}")

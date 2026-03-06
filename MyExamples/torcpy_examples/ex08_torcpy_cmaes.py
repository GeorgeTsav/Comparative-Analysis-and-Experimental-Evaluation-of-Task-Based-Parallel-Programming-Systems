"""
(C) Copyright IBM Corporation 2019
All rights reserved. This program and the accompanying materials
are made available under the terms of the Eclipse Public License v1.0
which accompanies this distribution, and is available at
http://www.eclipse.org/legal/epl-v10.html
"""

"""
Straightforward integration of torc_py with cmaes
With execution time measurement for framework comparison
"""
# pip install cma
import cma
import time
import torcpy as torc


def rosenbrock(x):
    """Rosenbrock test objective function"""
    n = len(x)
    if n < 2:
        raise ValueError('dimension must be greater one')
    return sum(100 * (x[i] ** 2 - x[i + 1]) ** 2 + (x[i] - 1) ** 2 for i in range(n - 1))


def main():
    popsize = 8
    maxfevals = 320
    es = cma.CMAEvolutionStrategy(2 * [0], 0.5, {'popsize': popsize, 'maxfevals': maxfevals, 'verb_disp': 0, 'seed': 3})
    
    # Record start time
    t0 = time.time()
    
    while not es.stop():
        solutions = es.ask()
        # Use torc.map for parallel evaluation
        es.tell(solutions, torc.map(rosenbrock, solutions))
        es.logger.add(es)  # write data to disc to be plotted
        es.disp()

    # Record end time
    t1 = time.time()

    print(f"Final solution: {es.result[0]}")
    print(f"Objective value: {es.result[1]}")
    
    elapsed = t1 - t0
    print("=" * 60)
    print("Framework: torcpy (CMA-ES Optimization)")
    print(f"Population size: {popsize}")
    print(f"Max function evaluations: {maxfevals}")
    print(f"Elapsed time: {elapsed:.4f} seconds")
    print("=" * 60)

    # cma.plot()

if __name__ == '__main__':
    torc.start(main)

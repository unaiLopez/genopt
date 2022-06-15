![alt text](https://github.com/unaiLopez/darwin/blob/master/doc/images/darwin.jpg?raw=true)

# Genetist: A genetic algorithm powered hyperparameter optimization framework
[![Python](https://img.shields.io/badge/python-3.6%20%7C%203.7%20%7C%203.8%20%7C%203.9%20%7C%203.10-blue)](https://www.python.org)
[![pypi](https://img.shields.io/pypi/v/genetist.svg)](https://pypi.python.org/pypi/genetist)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

Genetist is a high level framework that helps optimizing functions using the power of genetic algorithms.

## Installation
Genetist is available at [PyPI](https://pypi.org/project/genetist/)
```
$ pip install genetist
```
## Quickstart

```python
from genetist.environment import Environment

#defining a 4 variable search space of float values from -100.0 to 100.0
params = {
    'x': {'low': -100.0, 'high': 100.0},
    'y': {'low': -100.0, 'high': 100.0},
    'z': {'low': -100.0, 'high': 100.0},
    'k': {'low': -100.0, 'high': 100.0}
}

#defining an objective function
def objective(individual):
    x = individual['x']
    y = individual['y']
    z = individual['z']
    k = individual['k']
    
    return (x**2 - 4*y**3 / z**4) * k**3

if __name__ == '__main__':
    #defining our Environment instance with a population of 1000 individuals, 250 generation,
    #one-point crossover and a single gene mutation with a 25% probability of mutation
    environment = Environment(
        params=params,
        num_population=50,
        generations=25,
        crossover_type='one_point',
        mutation_type='single_gene',
        prob_mutation=0.25,
        verbose=0
    )

    #minimizing the objective function
    results = environment.optimize(objective=objective, direction='minimize')

    print()
    print(f'EXECUTION TIME={results.execution_time}')
    print(f'BEST SCORE={results.best_score}')
    print(f'BEST INDIVIDUAL={results.best_individual}')
    print('BEST PER GENERATION:')
    print(results.best_per_generation_dataframe)
```
|    |   generation |     best_score |   x |   y |   z |   k |
|---:|-------------:|---------------:|----:|----:|----:|----:|
|  0 |           13 | -inf           |  97 | -86 |   0 | -97 |
|  0 |           23 | -inf           |   7 | -17 |   0 | -48 |
|  0 |           22 | -inf           |  97 | -39 |   0 | -97 |
|  0 |           21 | -inf           |  47 | -17 |   0 | -42 |
|  0 |           20 | -inf           |   7 | -17 |   0 | -97 |
|  0 |           19 | -inf           | -93 | -32 |   0 | -97 |
|  0 |           18 | -inf           |  97 | -32 |   0 | -42 |
|  0 |           17 | -inf           |  97 | -17 |   0 | -97 |
|  0 |           16 | -inf           |  67 | -19 |   0 | -97 |
|  0 |           15 | -inf           |  26 | -41 |   0 | -97 |
|  0 |           14 | -inf           |  97 | -79 |   0 | -97 |
|  0 |           24 | -inf           |  35 | -17 |   0 | -48 |
|  0 |           25 | -inf           | -27 | -39 |   0 | -48 |
|  0 |           11 | -inf           |  97 | -32 |   0 | -97 |
|  0 |           10 | -inf           |  29 | -39 |   0 | -97 |
|  0 |            9 | -inf           |  67 | -39 |   0 | -97 |
|  0 |            8 | -inf           |  97 | -39 |   0 | -97 |
|  0 |            7 | -inf           |  67 | -41 |   0 | -97 |
|  0 |            6 | -inf           |  97 | -71 |   0 | -97 |
|  0 |            5 | -inf           |  88 | -41 |   0 | -97 |
|  0 |            4 | -inf           |  67 | -39 |   0 | -97 |
|  0 |            3 | -inf           |  67 | -39 |   0 | -97 |
|  0 |           12 | -inf           |  97 | -39 |   0 | -97 |
|  0 |            2 |   -7.06774e+09 |  88 |   2 |  42 | -97 |
|  0 |            1 |   -7.06774e+09 |  88 |   2 |  42 | -97 |
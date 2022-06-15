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
        num_population=1000,
        generations=250,
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
![alt text](https://github.com/unaiLopez/darwin/blob/master/doc/images/darwin.jpg?raw=true)

# Genetist: A genetic algorithm powered hyperparameter optimization framework
[![Python](https://img.shields.io/badge/python-3.6%20%7C%203.7%20%7C%203.8%20%7C%203.9%20%7C%203.10-blue)](https://www.python.org)
[![pypi](https://img.shields.io/pypi/v/genetist.svg)](https://pypi.python.org/pypi/genetist)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

Genetist is a high level framework that helps optimizing functions using the power of genetic algorithms.

## 1. Installation
Genetist is available at [PyPI](https://pypi.org/project/genetist/)
```
$ pip install genetist
```
## 2. Quickstart
### 2.1. Define Search Space
#### 2.1.1. Fixed Search Space
```python
#defining a fixed set of Parameters for 4 variables
params = {
    'x': [35, -51, 0, 1, 2, 3, 4, 66, 11, 50, 90],
    'y': [-100, -51, 0, 7, 32, 31, 4, 51, 121, 50, 90, 1000, 231]
    'z': [-10, -51, 0, 12, 2, 43, 43, 5, 1231, 50, 90],
    'k': [-56, -51, 0, 1, 2, 13, 4, 5, 11, 50, 90]
}
```
#### 2.1.2. Flexible Search Space
```python
from genetist.parameters import Parameters

#defining a 4 variable search space of float values from -100.0 to 100.0
params = {
    'x': Parameters.suggest_float(-100, 100),
    'y': Parameters.suggest_float(-100, 100),
    'z': Parameters.suggest_float(-100, 100),
    'k': Parameters.suggest_float(-100, 100)
}
```
### 2.2. Define Single-Objective Function
```python
#defining an objective function
def objective(individual):
    x = individual['x']
    y = individual['y']
    z = individual['z']
    k = individual['k']
    
    return (x**2 - 4*y**3 / z**4) * k**3
```
### 2.3. Define Multi-Objective Function
```python
#defining an objective function
def objective(individual):
    x = individual['x']
    y = individual['y']
    z = individual['z']
    k = individual['k']

    objective_1 = ((x**2 - 4*y**3 / z**4) * k**3)
    objective_2 = (k**3 / x)

    return objective_1, objective_2
```
### 2.4. Start Optimization for Single-Objective Function and Default Parameters
```python
from genetist.environment import Environment

if __name__ == '__main__':
    #defining our Environment instance with a population of 100 individuals,
    #roulette selection for 50% of the population, one-point crossover, 
    #a single gene mutation with 10% probability of mutation,
    #10% elite rate and some verbose and without a seed for reproducibility
    environment = Environment(
        params=params
    )
    #minimizing the objective function and adding a 60 seconds timeout
    #as an stop criteria
    results = environment.optimize(
        objective=objective,
        direction='minimize',
        timeout=60
    )
```
### 2.5. Start Optimization for Single-Objective Function and Custom Parameters
```python
from genetist.environment import Environment

if __name__ == '__main__':
    #defining our Environment instance with a population of 100 individuals,
    #tournament selection for 80% of the population, one-point crossover, 
    #a single gene mutation with 25% probability of mutation and some verbose and a seed for reproducibility
    environment = Environment(
        params=params,
        num_population=100,
        selection_type='tournament',
        selection_rate=0.8,
        crossover_type='one-point',
        mutation_type='single-gene',
        prob_mutation=0.25,
        verbose=1,
        random_state=42
    )
    #minimizing the objective function and adding 
    #3 stop criterias (num_generations, timeout, stop_score)
    results = environment.optimize(
        objective=objective,
        direction='minimize',
        num_generations=9999,
        timeout=60,
        stop_score=-np.inf
    )
```
### 2.6. Start Optimization for Multi-Objective Function
```python
from genetist.environment import Environment

if __name__ == '__main__':
    #defining our Environment instance with a population of 100 individuals,
    #ranking selection for 70% of the population, one-point crossover,
    #a single gene mutation with 25% probability of mutation and some verbose and a seed for reproducibility
    environment = Environment(
        params=params,
        num_population=100,
        selection_type='ranking',
        selection_rate=0.7,
        crossover_type='one-point',
        mutation_type='single-gene',
        prob_mutation=0.25,
        verbose=1,
        random_state=42
    )
    #minimizing the first value and maximazing the second value of the objective function,
    #adding 1 stop criterias (timeout), adding 50% weight to each objective and
    #assigning a name to each objective score for the final dataframe
    results = environment.optimize(
        objective=objective,
        direction=['minimize', 'maximize'],
        weights=[0.5, 0.5],
        timeout=60,
        score_names=['complex_equation_score', 'simple_equation_score']
    )
```
### 2.7. Show Optimization  Results
```python
print(f'EXECUTION TIME={results.execution_time}')
print(f'BEST SCORE={results.best_score}')
print(f'BEST INDIVIDUAL={results.best_individual}')
print('BEST PER GENERATION:')
print(results.best_per_generation_dataframe)
print('LAST GENERATION INDIVIDUALS:')
print(results.last_generation_individuals_dataframe)
```
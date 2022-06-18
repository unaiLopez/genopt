import numpy as np

from genetist.environment import Environment #pip install genetist
from genetist.params import Params

#defining a 4 variable search space of float values from -100.0 to 100.0
params = {
    'x': Params.suggest_float(-100, 100),
    'y': Params.suggest_float(-100, 100),
    'z': Params.suggest_float(-100, 100),
    'k': Params.suggest_float(-100, 100)
}

#defining an objective function
def objective(individual):
    x = individual['x']
    y = individual['y']
    z = individual['z']
    k = individual['k']
    
    return (x**2 - 4*y**3 / z**4) * k**3

if __name__ == '__main__':
    #defining our Environment instance with a population of 100 individuals,
    #one-point crossover and a single gene mutation with a 25% probability of mutation
    environment = Environment(
        params=params,
        num_population=100,
        crossover_type='one-point',
        mutation_type='single-gene',
        prob_mutation=0.25,
        verbose=1
    )
    #minimizing the objective function and adding 3 stop criterias (num_generations, timeout, stop_score)
    results = environment.optimize(objective=objective, direction='minimize', num_generations=9999, timeout=60, stop_score=-np.inf)

    print()
    print(f'EXECUTION TIME={results.execution_time}')
    print(f'BEST SCORE={results.best_score}')
    print(f'BEST INDIVIDUAL={results.best_individual}')
    print('BEST PER GENERATION:')
    print(results.best_per_generation_dataframe)
import sys
sys.path.append('../src/')

from genetist import Genetist

params = {
    'x': {'low': -100.0, 'high': 100.0},
    'y': {'low': -100.0, 'high': 100.0},
    'z': {'low': -100.0, 'high': 100.0},
    'k': {'low': -100.0, 'high': 100.0}
}

def objective(individual):
    x = individual['x']
    y = individual['y']
    z = individual['z']
    k = individual['k']
    
    return (x**2 - 4*y**3 / z**4) * k**3

if __name__ == '__main__':
    genetist = Genetist(
        params=params,
        num_population=1000,
        generations=250,
        crossover_type='one_point',
        mutation_type='single_gene',
        verbose=1
    )

    results = genetist.optimize(objective=objective, direction='minimize')

    print()
    print(f'EXECUTION TIME={results.execution_time}')
    print(f'BEST SCORE={results.best_score}')
    print(f'BEST INDIVIDUAL={results.best_individual}')
    print('BEST PER GENERATION:')
    print(results.best_per_generation_dataframe)
import numpy as np

from genetist import Genetist

if __name__ == '__main__':
    def fitness_funct_1(individual):
        for _, gene in individual.items():
            score = 0
            if gene > 60:
                score+= 1000
            elif gene < 10:
                score += 30000
            elif gene == 22:
                score += 9999999
        
        return score

    def fitness_funct_2(individual):
        return (individual['x']**2 - 4*individual['y']**3 / individual['z']**4) * individual['k']**3
    
    def fitness_funct_3(individual):
        return individual[0] * np.cos(individual[0]) * individual[1] * np.cos(individual[1]) * individual[2] * np.cos(individual[2])

    def fitness_categorical(individual):
        score = 0
        for gene in individual:
            if gene == 'HOLA':
                score += 1
            else:
                score -= 1
        
        return score

    params = {
        'x': {'low': -1000, 'high': 1000},
        'y': {'low': -1000, 'high': 1000},
        'z': {'low': -1000, 'high': 1000},
        'k': {'low': -1000, 'high': 1000},
    }

    params_1 = {
        'x1': {'low': -1000, 'high': 1000},
        'x2': {'low': -1000, 'high': 1000},
        'x3': {'low': -1000, 'high': 1000},
        'x4': {'low': -1000, 'high': 1000},
        'x5': {'low': -1000, 'high': 1000},
        'x6': {'low': -1000, 'high': 1000},
        'x7': {'low': -1000, 'high': 1000},
        'x8': {'low': -1000, 'high': 1000},
    }

    params_2 = {
        'n_estimators': [25, 35, 50, 75, 100],
        'max_depth': [3, 5, 7, 10, 15, 20, 25, 30, 35, 40, 45, 50],
        'min_samples_split': [2, 3, 4, 5, 6, 7, 8, 9, 10]
    }

    params_categorical = {
        'x1': {'type': 'categorical', 'choices': ['HOLA', 'HOLA2', 'HOLA3', 'HOLA4', 'HOLA5', 'HOLA6', 'HOLA7', 'HOLA8', 'HOLA9', 'HOLA10']},
        'x2': {'type': 'categorical', 'choices': ['HOLA', 'HOLA2', 'HOLA3', 'HOLA4', 'HOLA5', 'HOLA6', 'HOLA7', 'HOLA8', 'HOLA9', 'HOLA10']},
        'x3': {'type': 'categorical', 'choices': ['HOLA', 'HOLA2', 'HOLA3', 'HOLA4', 'HOLA5', 'HOLA6', 'HOLA7', 'HOLA8', 'HOLA9', 'HOLA10']},
        'x4': {'type': 'categorical', 'choices': ['HOLA', 'HOLA2', 'HOLA3', 'HOLA4', 'HOLA5', 'HOLA6', 'HOLA7', 'HOLA8', 'HOLA9', 'HOLA10']},
        'x5': {'type': 'categorical', 'choices': ['HOLA', 'HOLA2', 'HOLA3', 'HOLA4', 'HOLA5', 'HOLA6', 'HOLA7', 'HOLA8', 'HOLA9', 'HOLA10']},
        'x6': {'type': 'categorical', 'choices': ['HOLA', 'HOLA2', 'HOLA3', 'HOLA4', 'HOLA5', 'HOLA6', 'HOLA7', 'HOLA8', 'HOLA9', 'HOLA10']},
        'x7': {'type': 'categorical', 'choices': ['HOLA', 'HOLA2', 'HOLA3', 'HOLA4', 'HOLA5', 'HOLA6', 'HOLA7', 'HOLA8', 'HOLA9', 'HOLA10']},
        'x8': {'type': 'categorical', 'choices': ['HOLA', 'HOLA2', 'HOLA3', 'HOLA4', 'HOLA5', 'HOLA6', 'HOLA7', 'HOLA8', 'HOLA9', 'HOLA10']},

    }

    params_categorical_fixed = {
        'x1': ['HOLA', 'HOLA2', 'HOLA3', 'HOLA4', 'HOLA5', 'HOLA6', 'HOLA7', 'HOLA8', 'HOLA9', 'HOLA10'],
        'x2': ['HOLA', 'HOLA2', 'HOLA3', 'HOLA4', 'HOLA5', 'HOLA6', 'HOLA7', 'HOLA8', 'HOLA9', 'HOLA10'],
        'x3': ['HOLA', 'HOLA2', 'HOLA3', 'HOLA4', 'HOLA5', 'HOLA6', 'HOLA7', 'HOLA8', 'HOLA9', 'HOLA10'],
        'x4': ['HOLA', 'HOLA2', 'HOLA3', 'HOLA4', 'HOLA5', 'HOLA6', 'HOLA7', 'HOLA8', 'HOLA9', 'HOLA10'],
        'x5': ['HOLA', 'HOLA2', 'HOLA3', 'HOLA4', 'HOLA5', 'HOLA6', 'HOLA7', 'HOLA8', 'HOLA9', 'HOLA10'],
        'x6': ['HOLA', 'HOLA2', 'HOLA3', 'HOLA4', 'HOLA5', 'HOLA6', 'HOLA7', 'HOLA8', 'HOLA9', 'HOLA10'],
        'x7': ['HOLA', 'HOLA2', 'HOLA3', 'HOLA4', 'HOLA5', 'HOLA6', 'HOLA7', 'HOLA8', 'HOLA9', 'HOLA10'],
        'x8': ['HOLA', 'HOLA2', 'HOLA3', 'HOLA4', 'HOLA5', 'HOLA6', 'HOLA7', 'HOLA8', 'HOLA9', 'HOLA10'],

    }

    genetist = Genetist(
        params=params,
        num_population=500,
        elite_rate=0.1,
        mutation_type='single_gene',
        generations=100,
        prob_mutation=0.3,
        verbose=3
    )

    results = genetist.optimize(objective=fitness_funct_2, direction='maximize')

    print()
    print(f'EXECUTION TIME={results.execution_time}')
    print(f'BEST SCORE={results.best_score}')
    print(f'BEST INDIVIDUAL={results.best_individual}')
    print('BEST PER GENERATION:')
    print(results.best_per_generation_dataframe)
    
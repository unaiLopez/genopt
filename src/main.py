import numpy as np

from Genetist import Genetist

if __name__ == '__main__':
    def fitness_funct_1(individual):
        score = 0
        for gene in individual:
            if gene > 60:
                score+= 1000
            elif gene < 10:
                score += 30000
            elif gene == 22:
                score += 9999999
        
        return score

    def fitness_funct_2(individual):
        return (individual[0]**2 - 4*individual[1]**3 / individual[2]**4) * individual[3]**3
    
    def fitness_funct_3(individual):
        return individual[0] * np.cos(individual[0]) * individual[1] * np.cos(individual[1]) * individual[2] * np.cos(individual[2])

    genetist = Genetist(
        objective=fitness_funct_2,
        num_population=2000,
        generations=100,
        prob_mutation=0.1,
        genome_size=20,
        type='continuous',
        direction='minimize',
        boundaries=[-10000, 10000],
        verbose=True
    )

    results = genetist.run_evolution()

    print()
    print(f'EXECUTION TIME={results.execution_time}')
    print(f'BEST SCORE={results.best_score}')
    print(f'BEST INDIVIDUAL={results.best_individual}')
    print('BEST PER GENERATION:')
    print(results.best_per_generation_dataframe)
    
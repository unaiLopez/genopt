import numpy as np
from tqdm import tqdm
import random
import time

from Results import Results

class Genetist:
    def __init__(self, objective, genome_size, num_population=100, prob_mutation=0.1, generations=100, direction='minimize', type='discrete', boundaries=[-1000000, 1000000], verbose=True):
        self.objective = objective
        self.genome_size = genome_size
        self.num_population = num_population
        self.prob_mutation = prob_mutation
        self.generations = generations
        self.direction = direction
        self.type = type
        self.boundaries = boundaries
        self.verbose = verbose

    def initialize_population(self):
        population = list()
        for _ in range(self.num_population):
            if self.type == 'discrete':
                population.append(np.random.randint(low=self.boundaries[0], high=self.boundaries[1], size=self.genome_size, dtype=int))
            elif self.type == 'continuous':
                population.append(np.random.uniform(low=self.boundaries[0], high=self.boundaries[1], size=self.genome_size))
            elif self.type == 'binary':
                population.append(np.random.randint(low=0, high=1, size=self.genome_size, dtype=int))
            else:
                raise Exception(f'Type {self.type} not supported.')

        return population

    def compete(self, population):
        competitors = list()
        for individual in population:
            competitors.append([self.objective(individual), individual])
        if self.direction == 'maximize':
            competitors = sorted(competitors, key=lambda x: x[0], reverse=True)
        elif self.direction == 'minimize':
            competitors = sorted(competitors, key=lambda x: x[0])
        else:
            raise Exception(f'Direction {self.direction} not supported.')

        return competitors

    def choose_parents(self, competitors):
        parents = list()
        weights = np.arange(len(competitors), 0, step=-1)
        competitors = [row[1] for row in competitors]
        for _ in range(len(competitors) // 2):
            parents.append(
                random.choices(
                    population=competitors,
                    weights=weights,
                    k=2
                )
            )

        return parents

    def mutate(self, offspring):
        if np.random.rand() < self.prob_mutation:
            gene_mutation_index = np.random.random_integers(low=0,  high=len(offspring)-1)
            if self.type == 'discrete':
                offspring[gene_mutation_index] = np.random.random_integers(low=self.boundaries[0], high=self.boundaries[1])
            elif self.type == 'continuous':
                offspring[gene_mutation_index] = np.random.uniform(low=self.boundaries[0], high=self.boundaries[1], size=self.genome_size)[0]
            elif self.binary == 'binary':
                if offspring[gene_mutation_index] == 0:
                    offspring[gene_mutation_index] = 1
                else:
                    offspring[gene_mutation_index] = 0
            else:
                raise Exception(f'Type {self.type} not supported.')

        return offspring

    def crossover(self, best_parents):
        offsprings = list()
        threshold = np.random.random_integers(low=0, high=self.genome_size-1)
        for parents in best_parents:
            offspring_1 = list(parents[0][:threshold]) + list(parents[1][threshold:])
            offspring_2 = list(parents[1][:threshold]) + list(parents[0][threshold:])
            offspring_1 = self.mutate(offspring_1)
            offspring_2 = self.mutate(offspring_2)
            offsprings.append(offspring_1)
            offsprings.append(offspring_2)

        return offsprings

    def evolution_loop(self):
        results = Results()
        individuals = self.initialize_population()
        progress_bar = tqdm(range(0, self.generations))
        for generation in progress_bar:
            individuals = self.compete(individuals)
            if self.verbose == True: 
                progress_bar.set_description(f'RUNNING GENERATION {generation + 1}')
                print(f'THE BEST SOLUTION IN GENERATION {generation+1} IS: {individuals[0][1]} WITH A SCORE OF {individuals[0][0]}')
            results.add_generation_results({'GENERATION': generation+1, 'BEST_SCORE': individuals[0][0], 'BEST_INDIVIDUAL': individuals[0][1]})
            best_individuals = self.choose_parents(individuals)
            individuals = self.crossover(best_individuals)
        
        return results
    
    def run_evolution(self):
        start_time = time.time()
        results = self.evolution_loop()
        end_time = time.time()

        results.set_execution_time(end_time - start_time)
        results.sort_best_per_generation_dataframe(column='BEST_SCORE', direction=self.direction)
        results.set_best_score()
        results.set_best_individual()
            
        return results


import numpy as np
import pandas as pd
import random

class Genetist:
    def __init__(self, fitness_funct, genome_size, num_population=100, prob_mutation=0.1, generations=100, type='int', boundaries=[-1000000, 1000000]):
        self.fitness_funct = fitness_funct
        self.genome_size = genome_size
        self.num_population = num_population
        self.prob_mutation = prob_mutation
        self.generations = generations
        self.type = type
        self.boundaries = boundaries

    def initialize_population(self):
        population = list()
        for _ in range(self.num_population):
            if self.type == 'int':
                population.append(np.random.randint(low=self.boundaries[0], high=self.boundaries[1], size=self.genome_size, dtype=int))
            elif self.type == 'float':
                population.append(np.random.uniform(low=self.boundaries[0], high=self.boundaries[1], size=self.genome_size))

        return population

    def compete(self, population):
        competitors = list()
        for individual in population:
            competitors.append([self.fitness_funct(individual), individual])
        competitors.sort(key=lambda x: x[0], reverse=True)

        return competitors

    def choose_parents(self, competitors):
        parents = list()
        weights = [row[0] for row in competitors]
        competitors = [row[1] for row in competitors]
        for _ in range(len(competitors) // 2):
            parents.append(random.choices(
                           population=competitors,
                           weights=weights,
                           k=2
            ))

        return parents

    def mutate(self, offspring):
        if np.random.rand() < self.prob_mutation:
            gene_mutation_index = np.random.random_integers(low=0,  high=len(offspring)-1)
            if self.type == 'int':
                offspring[gene_mutation_index] = np.random.random_integers(low=self.boundaries[0], high=self.boundaries[1])
            elif self.type == 'float':
                offspring[gene_mutation_index] = np.random.uniform(low=self.boundaries[0], high=self.boundaries[1], size=self.genome_size)[0]

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

    def evolution(self):
        results = pd.DataFrame()
        individuals = self.initialize_population()
        for generation in range(0, self.generations):
            individuals = self.compete(individuals)
            print(f'THE BEST SOLUTION IN GENERATION {generation+1} IS: {individuals[0][1]} WITH A SCORE OF {individuals[0][0]}')
            results = results.append({'GENERATION': generation+1, 'BEST_SCORE': individuals[0][0], 'BEST_SOLUTION': individuals[0][1]}, ignore_index=True)
            best_individuals = self.choose_parents(individuals)
            individuals = self.crossover(best_individuals)
            
        return results


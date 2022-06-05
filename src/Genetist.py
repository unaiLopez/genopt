import numpy as np
from tqdm import tqdm
import random
import time

from DataTypeInference import DataTypeInference
from Results import Results

class Genetist:
    def __init__(self, objective, params, num_population=100, prob_mutation=0.1, generations=100, direction='minimize', verbose=True):
        self.objective = objective
        self.params = params
        self.num_population = num_population
        self.prob_mutation = prob_mutation
        self.generations = generations
        self.direction = direction
        self.verbose = verbose

        self.data_inference_object = DataTypeInference(params)
        self.search_space_type = self.data_inference_object.infer_search_space_type()
        self.params = self.data_inference_object.infer_param_types()

    def create_individual(self):
        new_individual = list()
        if self.search_space_type == 'flexible_search':
            for _, values in self.params.items():
                if values['type'] == 'int':
                    new_individual.append(np.random.random_integers(values['low'], values['high']))
                elif values['type'] == 'float':
                    new_individual.append(np.random.uniform(values['low'], values['high']))
                elif values['type'] == 'categorical':
                    new_individual.append(np.random.choice(values['choices']))
                else:
                    raise ValueError(f'Type {values["type"]} not supported.')
        else:
            for _, values in self.params.items():
                new_individual.append(np.random.choice(values))
        
        return new_individual
            
    def initialize_population(self):
        population = list()
        for _ in range(self.num_population):
            individual = self.create_individual()
            population.append(individual)

        return population
    
    def individual_from_list_to_dict(self, individual):
        new_individual = {}
        names = list(self.params.keys())
        for name, gene in zip(names, individual):
            new_individual[name] = gene
        
        return new_individual
        

    def compete(self, population):
        competitors = list()
        for individual in population:
            competitors.append([self.objective(self.individual_from_list_to_dict(individual)), individual])

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
    
    def mutate_in_fixed_search(self, offspring, gene_mutation_index, param):
        if len(self.params.get(param)) == 2:
            if self.params.get(param)[0] != offspring[gene_mutation_index]:
                offspring[gene_mutation_index] = self.params.get(param)[0]
            else:
                offspring[gene_mutation_index] = self.params.get(param)[1]
        else:
            offspring[gene_mutation_index] = np.random.choice(self.params.get(param))
        
        return offspring

    def mutate_in_flexible_search(self, offspring, gene_mutation_index, param):
        if self.params.get(param).get('type') == 'int':
            if self.params[param]['low'] == 0 and self.params[param]['high'] == 1:
                 if offspring[gene_mutation_index] == 0:
                    offspring[gene_mutation_index] = 1
                 else:
                    offspring[gene_mutation_index] = 0
            else:
                offspring[gene_mutation_index] = np.random.random_integers(self.params[param]['low'], (self.params[param]['high']))

        elif self.params.get(param).get('type') == 'float':
            offspring[gene_mutation_index] = np.random.uniform(self.params[param]['low'], self.params[param]['high'])

        elif self.params.get(param).get('type') == 'categorical':
            if len(self.params.get(param).get('choices')) == 2:
                if self.params.get(param).get('choices')[0] != offspring[gene_mutation_index]:
                    offspring[gene_mutation_index] = self.params.get(param)[0]
                else:
                    offspring[gene_mutation_index] = self.params.get(param)[1]
            else:
                offspring[gene_mutation_index] = np.random.choice(self.params.get(param).get('choices'))
        
        return offspring

    def mutate(self, offspring):
        if np.random.rand() < self.prob_mutation:
            gene_mutation_index = np.random.random_integers(low=0,  high=len(offspring)-1)
            param = list(self.params.keys())[gene_mutation_index]
            if self.search_space_type == 'flexible_search':
                offspring = self.mutate_in_flexible_search(offspring, gene_mutation_index, param)
            else:
               offspring = self.mutate_in_fixed_search(offspring, gene_mutation_index, param)
            
        
        return offspring

    def crossover(self, best_parents):
        offsprings = list()
        threshold = np.random.random_integers(low=0, high=len(best_parents[0])-1)
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
        results.convert_best_individuals_to_param_columns(self.params)
        results.sort_best_per_generation_dataframe(column='BEST_SCORE', direction=self.direction)
        results.set_best_score()
        results.set_best_individual()
            
        return results


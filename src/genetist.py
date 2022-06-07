import numpy as np
from tqdm import tqdm
import random
import time
import math

from crossover import Crossover
from mutation import Mutation
from data_type_inference import DataTypeInference
from results import Results

class Genetist:
    def __init__(self, objective, params, num_population=100, cross_over_type='one_point', mutation_type='single_gene', elite_rate=0.1, prob_mutation=0.1, generations=100, direction='minimize', verbose=True):
        self.objective = objective
        self.params = params
        self.num_population = num_population
        self.cross_over_type = cross_over_type
        self.mutation_type = mutation_type
        self.elite_rate = elite_rate
        self.prob_mutation = prob_mutation
        self.generations = generations
        self.direction = direction
        self.verbose = verbose
        self.search_space_type = None

        self.input_datatypes_inference()
        self.crossover = Crossover(self.cross_over_type, self.search_space_type)
        self.mutation = Mutation(self.mutation_type, self.prob_mutation, self.search_space_type, self.params)

    def input_datatypes_inference(self):
        data_inference_object = DataTypeInference(self.params)
        self.search_space_type = data_inference_object.infer_search_space_type()
        self.params = data_inference_object.infer_param_types()

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
        number_of_parents = math.ceil(len(competitors) * (1 - self.elite_rate)) // 2
        for _ in range(number_of_parents):
            parents.append(
                random.choices(
                    population=competitors,
                    weights=weights,
                    k=2
                )
            )

        return parents

    def run_crossover(self, best_parents):
        childs = list()
        for parents in best_parents:
            child_1, child_2 = self.crossover.crossover(parents[0], parents[1])
            child_1 = self.mutation.mutate(child_1)
            child_2 = self.mutation.mutate(child_2)
            childs.append(child_1)
            childs.append(child_2)

        return childs

    def get_elite(self, competitors):
        competitors = [row[1] for row in competitors]
        elite = competitors[:math.ceil(len(competitors) * self.elite_rate)]

        return elite

    def evolution_loop(self):
        results = Results()
        individuals = self.initialize_population()
        progress_bar = tqdm(range(0, self.generations))
        for generation in progress_bar:
            individuals = self.compete(individuals)
            if self.verbose == True: 
                progress_bar.set_description(f'RUNNING GENERATION {generation + 1}')
                print(f'THE BEST SOLUTION IN GENERATION {generation+1} IS: {individuals[0][1]} WITH A SCORE OF {individuals[0][0]}')
            elite_individuals = self.get_elite(individuals)
            results.add_generation_results({'GENERATION': generation+1, 'BEST_SCORE': individuals[0][0], 'BEST_INDIVIDUAL': individuals[0][1]})
            best_individuals = self.choose_parents(individuals)
            individuals = self.run_crossover(best_individuals)
            individuals.extend(elite_individuals)
            
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


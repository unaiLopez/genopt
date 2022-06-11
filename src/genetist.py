import numpy as np
from tqdm import tqdm
import random
import time
import math

import logging
from crossover import Crossover
from mutation import Mutation
from datatype_inference import DataTypeInference
from results import Results

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('GENETIST')

class Genetist:
    def __init__(self, params, num_population=100, generations=100, cross_over_type='one_point', mutation_type='single_gene', prob_mutation=0.1, elite_rate=0.1, verbose=1):
        self.params = params
        self.num_population = num_population
        self.generations = generations
        self.cross_over_type = cross_over_type
        self.mutation_type = mutation_type
        self.prob_mutation = prob_mutation
        self.elite_rate = elite_rate
        self.verbose = verbose

        data_inference_object = DataTypeInference(self.params)
        self.search_space_type = data_inference_object.infer_search_space_type()
        self.params = data_inference_object.infer_param_types()
        self.crossover = Crossover(self.cross_over_type, self.search_space_type)
        self.mutation = Mutation(self.mutation_type, self.prob_mutation, self.search_space_type, self.params)

    def _initialize_individual(self):
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
            
    def _initialize_population(self):
        if self.verbose > 1: logger.info(f'Initializing population...')
        population = list()
        for _ in range(self.num_population):
            individual = self._initialize_individual()
            population.append(individual)

        return population
    
    def _individual_from_list_to_dict(self, individual):
        new_individual = {}
        names = list(self.params.keys())
        for name, gene in zip(names, individual):
            new_individual[name] = gene
        
        return new_individual
    
    def _calculate_population_fitness(self, individuals, objective):
        if self.verbose > 1: logger.info(f'Calculating population fitness...')
        new_individuals = list()
        for individual in individuals:
            new_individuals.append([objective(self._individual_from_list_to_dict(individual)), individual])

        return new_individuals

    def _order_population_by_fitness(self, individuals, direction):
        if self.verbose > 1: logger.info(f'Ordering population by fitness...')
        if direction == 'maximize':
            individuals = sorted(individuals, key=lambda x: x[0], reverse=True)
        elif direction == 'minimize':
            individuals = sorted(individuals, key=lambda x: x[0])
        else:
            raise Exception(f'Direction {direction} not supported.')

        return individuals

    def _choose_parents(self, competitors):
        if self.verbose: logger.info(f'Selecting parents...')
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

    def _run_crossover_with_mutation(self, best_parents):
        if self.verbose > 1: logger.info(f'Running crossover with mutation...')
        childs = list()
        for parents in best_parents:
            child_1, child_2 = self.crossover.crossover(parents[0], parents[1])
            child_1 = self.mutation.mutate(child_1)
            child_2 = self.mutation.mutate(child_2)
            childs.append(child_1)
            childs.append(child_2)

        return childs

    def _get_elite(self, competitors):
        if self.verbose > 1: logger.info(f'Getting elite individuals...')
        competitors = [row[1] for row in competitors]
        elite = competitors[:math.ceil(len(competitors) * self.elite_rate)]

        return elite
    
    def optimize(self, objective, direction):
        start_time = time.time()
        
        results = Results()
        individuals = self._initialize_population()
        individuals = self._calculate_population_fitness(individuals, objective)
        individuals = self._order_population_by_fitness(individuals, direction)
        progress_bar = tqdm(range(0, self.generations))
        for generation in progress_bar:
            elite_individuals = self._get_elite(individuals)
            parents = self._choose_parents(individuals)
            individuals = self._run_crossover_with_mutation(parents)
            individuals.extend(elite_individuals)
            individuals = self._calculate_population_fitness(individuals, objective)
            individuals = self._order_population_by_fitness(individuals, direction)
            best_individual = self._individual_from_list_to_dict(individuals[0][1])
            best_score = individuals[0][0]

            results.add_generation_results(generation+1, best_score, best_individual)
            progress_bar.set_description(f'RUNNING GENERATION {generation + 1}')
            if self.verbose == 1:
                logger.info(f'THE BEST SOLUTION IN GENERATION {generation+1} IS: {best_individual}')

        end_time = time.time()

        results.set_execution_time(end_time - start_time)
        results.sort_best_per_generation_dataframe(column='BEST_SCORE', direction=direction)
        results.get_best_score()
        results.get_best_individual()
            
        return results


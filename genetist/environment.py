import numpy as np
import random
import time
import math
import logging

from typing import Callable, List, Tuple, Union
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
from genetist.crossover import Crossover
from genetist.mutation import Mutation
from genetist.individual import Individual
from genetist.datatype_inference import DataTypeInference
from genetist.results import Results

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('ENVIRONMENT')

class Environment:
    def __init__(self, params: dict, num_population: int = 100, generations: int = 100, crossover_type: str = 'one_point', mutation_type: str = 'single_gene', prob_mutation: float = 0.1, elite_rate: float = 0.1, verbose: int = 1):
        self.params = params
        self.num_population = num_population
        self.generations = generations
        self.crossover_type = crossover_type
        self.mutation_type = mutation_type
        self.prob_mutation = prob_mutation
        self.elite_rate = elite_rate
        self.verbose = verbose

        self.search_space_type = DataTypeInference.infer_search_space_type(params)
        self.crossover = Crossover(self.crossover_type)
        self.mutation = Mutation(self.mutation_type, self.prob_mutation, self.search_space_type, self.params)
            
    def _initialize_population(self, objective: Callable[[dict], Union[int,float]]) -> List[Individual]:
        if self.verbose > 1: logger.info(f'Initializing population...')
        population = list()
        for _ in range(self.num_population):
            population.append(Individual(self.params, self.search_space_type, objective))

        return population
    
    def _calculate_fitness_process(self, individual: Individual) -> Individual:
        individual.calculate_fitness()
        return individual

    def _calculate_population_fitness(self, individuals: List[Individual], n_jobs: int = 1) -> List[Individual]:
        if self.verbose > 1: logger.info(f'Calculating population fitness...')
        if n_jobs == -1: n_jobs = cpu_count()

        if n_jobs != 1:
            pool = Pool(n_jobs)
            new_individuals = pool.map(self._calculate_fitness_process, individuals)
        else:
            new_individuals = list()
            for individual in individuals:
                new_individuals.append(self._calculate_fitness_process(individual))

        return new_individuals

    def _order_population_by_fitness(self, individuals: List[Individual], direction: str) -> List[Individual]:
        if self.verbose > 1: logger.info(f'Ordering population by fitness...')
        if direction == 'maximize':
            individuals = sorted(individuals, key=lambda individual: individual.get_fitness(), reverse=True)
        elif direction == 'minimize':
            individuals = sorted(individuals, key=lambda individual: individual.get_fitness())
        else:
            raise Exception(f'Direction {direction} not supported.')

        return individuals

    def _choose_parents(self, individuals: List[Individual]) -> List[Tuple[Individual, Individual]]:
        if self.verbose > 1: logger.info(f'Selecting parents...')
        parents = list()
        weights = np.arange(len(individuals), 0, step=-1)
        number_of_parents = math.ceil(len(individuals) * (1 - self.elite_rate)) // 2
        for _ in range(number_of_parents):
            parents.append(
                random.choices(
                    population=individuals,
                    weights=weights,
                    k=2
                )
            )

        return parents

    def _run_crossover_with_mutation(self, best_parents: List[Tuple[Individual, Individual]]) -> List[Individual]:
        if self.verbose > 1: logger.info(f'Running crossover with mutation...')
        childs = list()
        for parents in best_parents:
            child_1, child_2 = self.crossover.crossover(parents[0], parents[1])
            child_1 = self.mutation.mutate(child_1)
            child_2 = self.mutation.mutate(child_2)
            childs.append(child_1)
            childs.append(child_2)

        return childs

    def _get_elite(self, individuals: List[Individual]) -> List[Individual]:
        if self.verbose > 1: logger.info(f'Getting elite individuals...')
        elite = individuals[:math.ceil(len(individuals) * self.elite_rate)]

        return elite
    
    def optimize(self, objective: Callable[[dict], Union[int,float]], direction: str, n_jobs: int = 1) -> Results:
        start_time = time.time()
        
        results = Results()
        individuals = self._initialize_population(objective)
        individuals = self._calculate_population_fitness(individuals, n_jobs)
        individuals = self._order_population_by_fitness(individuals, direction)
        progress_bar = tqdm(range(0, self.generations))
        for generation in progress_bar:
            elite_individuals = self._get_elite(individuals)
            parents = self._choose_parents(individuals)
            individuals = self._run_crossover_with_mutation(parents)
            individuals.extend(elite_individuals)
            individuals = self._calculate_population_fitness(individuals, n_jobs)
            individuals = self._order_population_by_fitness(individuals, direction)
            best_individual = individuals[0].get_name_genome_genes()
            best_score = individuals[0].get_fitness()

            results.add_generation_results(generation+1, best_score, best_individual)
            progress_bar.set_description(f'RUNNING GENERATION {generation + 1}')
            if self.verbose >= 1:
                logger.info(f'THE BEST SOLUTION IN GENERATION {generation+1} IS {best_individual} WITH A SCORE OF {best_score}')

        end_time = time.time()

        results.set_execution_time(end_time - start_time)
        results.sort_best_per_generation_dataframe(column='best_score', direction=direction)
        results.get_best_score()
        results.get_best_individual()
            
        return results


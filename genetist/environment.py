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

MAX_GENERATIONS = 999999999999
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('ENVIRONMENT')

class Environment:
    def __init__(self, params: dict, num_population: int = 100, crossover_type: str = 'one_point', mutation_type: str = 'single_gene', prob_mutation: float = 0.1, elite_rate: float = 0.1, verbose: int = 1):
        self.params = params
        self.num_population = num_population
        self.crossover_type = crossover_type
        self.mutation_type = mutation_type
        self.prob_mutation = prob_mutation
        self.elite_rate = elite_rate
        self.verbose = verbose

        self.search_space_type = DataTypeInference.infer_search_space_type(params)
            
    def _initialize_population(self, objective: Callable[[dict], Union[int,float]]) -> List[Individual]:
        if self.verbose > 1: logger.info(f'Initializing population...')
        population = list()
        for _ in range(self.num_population):
            individual = Individual(self.params, self.search_space_type, objective)
            individual.initialize_genome()
            population.append(individual)

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
            individuals = sorted(individuals, key=lambda individual: individual.fitness, reverse=True)
        elif direction == 'minimize':
            individuals = sorted(individuals, key=lambda individual: individual.fitness)
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
        crossover =  Crossover.getInstance(self.crossover_type)
        mutation = Mutation.getInstance(self.mutation_type, self.prob_mutation, self.search_space_type, self.params)
        if self.verbose > 1: logger.info(f'Running crossover with mutation...')
        childs = list()
        for parents in best_parents:
            child_1, child_2 = crossover.crossover(parents[0], parents[1])
            child_1 = mutation.mutate(child_1)
            child_2 = mutation.mutate(child_2)
            childs.append(child_1)
            childs.append(child_2)

        return childs

    def _get_elite(self, individuals: List[Individual]) -> List[Individual]:
        if self.verbose > 1: logger.info(f'Getting elite individuals...')
        elite = individuals[:math.ceil(len(individuals) * self.elite_rate)]

        return elite
    
    def _check_stop_criterias(self, num_generations, timeout, stop_score):
        if num_generations == None and (timeout != None or stop_score != None):
            num_generations = MAX_GENERATIONS
        elif num_generations == None and timeout == None and stop_score == None:
            raise Exception('Stop criteria does not exist. Define num_generations, timeout, stop_score or all of them.')
        return num_generations, timeout, stop_score
    
    def _check_stop_timeout(self, timeout: Union[float, int], start_time: float) -> bool:
        if timeout != None:
            elapsed_time = time.time() - start_time
            if elapsed_time >= timeout:
                return True
            else:
                return False
        else:
            return False
    
    def _check_stop_score(self, stop_score: Union[float, int], best_score: Union[float, int], direction: str) -> bool:
        if stop_score != None:
            if direction == 'minimize' and stop_score >= best_score:
                return True
            elif direction == 'maximize' and stop_score <= best_score:
                return True
            else:
                return False
        else:
            return False
    
    def _check_num_generations_criteria(self, num_generations: int, generation: int) -> bool:
        if num_generations != None:
            if generation >= num_generations - 1:
                return True
            else:
                return False
        else:
            return False

    def optimize(self, objective: Callable[[dict], Union[int,float]], direction: str, num_generations: int = None, timeout: int = None, stop_score: Union[float, int] = None, n_jobs: int = 1) -> Results:
        start_time = time.time()
        
        results = Results()
        individuals = self._initialize_population(objective)
        individuals = self._calculate_population_fitness(individuals, n_jobs)
        individuals = self._order_population_by_fitness(individuals, direction)

        num_generations, timeout, stop_score = self._check_stop_criterias(num_generations, timeout, stop_score)
        for generation in range(0, num_generations):
            elite_individuals = self._get_elite(individuals)
            parents = self._choose_parents(individuals)
            individuals = self._run_crossover_with_mutation(parents)
            individuals.extend(elite_individuals)
            individuals = self._calculate_population_fitness(individuals, n_jobs)
            individuals = self._order_population_by_fitness(individuals, direction)
            best_individual = individuals[0].get_name_genome_genes()
            best_score = individuals[0].fitness
            results.add_generation_results(generation+1, best_score, best_individual)
            
            stop_timeout_criteria = self._check_stop_timeout(timeout, start_time)
            stop_score_criteria = self._check_stop_score(stop_score, best_score, direction)
            stop_num_generations_criteria = self._check_num_generations_criteria(num_generations, generation)
            if self.verbose >= 1:
                logger.info(f'THE BEST SOLUTION IN GENERATION {generation+1} IS {best_individual} WITH A SCORE OF {best_score}')
                if stop_timeout_criteria:
                    logger.info('TIMEOUT STOP CRITERIA SATISFIED.')
                if stop_score_criteria:
                    logger.info('SCORE STOP CRITERIA SATISFIED.')
                if stop_num_generations_criteria:
                    logger.info('NUM GENERATIONS CRITERIA SATISFIED.')
            if stop_timeout_criteria or stop_score_criteria or stop_num_generations_criteria:
                if self.verbose >= 1: logger.info('STOPPING OPTIMIZATION...')
                break

        end_time = time.time()
        results.set_execution_time(end_time - start_time)
        results.sort_best_per_generation_dataframe(column='best_score', direction=direction)
        results.get_best_score()
        results.get_best_individual()
            
        return results


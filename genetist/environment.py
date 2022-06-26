import numpy as np
import pandas as pd
import random
import time
import math
import logging

from typing import Callable, List, Tuple, Union
from multiprocessing import Pool, cpu_count
from genetist.crossover import Crossover
from genetist.mutation import Mutation
from genetist.individual import Individual
from genetist.datatype_inference import DataTypeInference
from genetist.results import Results
from genetist.utils import define_weights_by_default_if_not_defined, normalize_best_score_by_index, calculate_weighted_sum_score_by_index


MAX_GENERATIONS = 999999999999
MAX_ATTEMPS_PER_INDIVIDUAL = 5
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('ENVIRONMENT')

class Environment:
    def __init__(self, params: dict, num_population: int = 100, crossover_type: str = 'one_point', mutation_type: str = 'single_gene', prob_mutation: float = 0.1, elite_rate: float = 0.1, verbose: int = 1, random_state: int = None):
        self.params = params
        self.num_population = num_population
        self.crossover_type = crossover_type
        self.mutation_type = mutation_type
        self.prob_mutation = prob_mutation
        self.elite_rate = elite_rate
        self.verbose = verbose
        self.results = Results()
        self.history_genomes = set()
        self.search_space_type = DataTypeInference.infer_search_space_type(params)
        random.seed(random_state)
        np.random.seed(random_state)

    def _is_in_history(self, individual: Individual) -> bool:
        if tuple(individual.genome) in self.history_genomes:
            return False
        else:
            return True
    
    def _add_individual_in_history(self, individual: Individual) -> None:
        self.history_genomes.add(tuple(individual.genome))
    
    def _create_individual_checking_duplicates(self, objective):
        attemps = 0
        keep = True
        while keep:
            attemps += 1
            individual = Individual(self.params, self.search_space_type, objective)
            individual.initialize_genome()
            if self._is_in_history(individual) == False or attemps == MAX_ATTEMPS_PER_INDIVIDUAL:
                self._add_individual_in_history(individual)
                keep = False

        return individual
            
    def _initialize_population(self, objective: Callable[[dict], Union[int,float]]) -> List[Individual]:
        if self.verbose > 1: logger.info(f'Initializing population...')
        population = list()
        for _ in range(self.num_population):
            individual = self._create_individual_checking_duplicates(objective)
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
    
    def _order_population_by_multiple_fitness(self, individuals: List[Individual], direction: List[str], weights: List[Union[int, float]]) -> List[Individual]:
        if len(direction) == len(weights):
            number_of_fitnesses = len(individuals[0].fitness)
            if number_of_fitnesses != len(direction):
                raise Exception(f'Direction and weights do not match number of fitness values.')
            weights = define_weights_by_default_if_not_defined(weights, direction)
            df_ranks = pd.DataFrame({'idx_individual': range(len(individuals))})
            for i in range(number_of_fitnesses):
                best_scores = [individual.fitness[i] for individual in individuals]
                df_ranks[f'best_score_{i}'] = best_scores
                df_ranks = normalize_best_score_by_index(df_ranks, direction, i)
                df_ranks = calculate_weighted_sum_score_by_index(df_ranks, weights, i)
            df_ranks_sorted = df_ranks.sort_values(['overall_best_score'], ascending=False)
            idx_individuals = df_ranks_sorted['idx_individual'].values
            individuals = [individuals[i] for i in idx_individuals]

        return individuals

    def _order_population_by_single_fitness(self, individuals: List[Individual], direction: str):
        if direction == 'maximize':
            individuals = sorted(individuals, key=lambda individual: individual.fitness, reverse=True)
        elif direction == 'minimize':
            individuals = sorted(individuals, key=lambda individual: individual.fitness)
        else:
            raise Exception(f'Direction {direction} not supported.')
            
        return individuals

    def _order_population_by_fitness(self, individuals: List[Individual], direction: Union[str, List[str]], weights: List[Union[int, float]]) -> List[Individual]:
        if self.verbose > 1: logger.info(f'Ordering population by fitness...')
        if isinstance(direction, str):
            individuals = self._order_population_by_single_fitness(individuals, direction)
        elif isinstance(direction, list):
            individuals = self._order_population_by_multiple_fitness(individuals, direction, weights)
        else:
            raise Exception(f'Direction {direction} not supported. Must be of type str or List[str]')

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

    def optimize(self, objective: Callable[[dict], Union[int,float, Tuple[Union[int, float]]]], direction: Union[str, List[str]], weights: List[Union[int, float]] = None, score_names: Union[str, List[str]] = None, num_generations: int = None, timeout: int = None, stop_score: Union[float, int] = None, n_jobs: int = 1) -> Results:
        start_time = time.time()
        
        individuals = self._initialize_population(objective)
        individuals = self._calculate_population_fitness(individuals, n_jobs)
        individuals = self._order_population_by_fitness(individuals, direction, weights)

        num_generations, timeout, stop_score = self._check_stop_criterias(num_generations, timeout, stop_score)
        for generation in range(1, num_generations):
            elite_individuals = self._get_elite(individuals)
            parents = self._choose_parents(individuals)
            individuals = self._run_crossover_with_mutation(parents)
            individuals.extend(elite_individuals)
            individuals = self._calculate_population_fitness(individuals, n_jobs)
            individuals = self._order_population_by_fitness(individuals, direction, weights)
            best_individual = individuals[0].get_name_genome_genes()
            best_score = individuals[0].fitness
            self.results.add_generation_results(generation, best_score, best_individual)
            
            stop_timeout_criteria = self._check_stop_timeout(timeout, start_time)
            stop_score_criteria = self._check_stop_score(stop_score, best_score, direction)
            stop_num_generations_criteria = self._check_num_generations_criteria(num_generations, generation)
            if self.verbose >= 1:
                logger.info(f'THE BEST SOLUTION IN GENERATION {generation} IS {best_individual} WITH A SCORE OF {best_score}')
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
        self.results.execution_time = end_time - start_time
        self.results.best_score = best_score
        self.results.best_individual = best_individual
        self.results.create_last_generation_individuals_dataframe(generation, individuals, score_names)
        self.results.sort_best_per_generation_dataframe(direction, weights, score_names)

        return self.results


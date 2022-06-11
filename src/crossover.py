import numpy as np

from copy import copy
from individual import Individual
from typing import List, Union, Tuple

class Crossover:
    def __init__(self, crossover_type: str, search_space_type: str):
        self.crossover_type = crossover_type
        self.search_space_type = search_space_type

    def _one_point_crossover(self, parent_1: Individual, parent_2: Individual) -> Tuple[Individual, Individual]:
        point = np.random.randint(1, len(parent_1) - 1)
        
        child_1_genome = parent_1.get_genome()[:point] + parent_2.get_genome()[point:]
        child_2_genome = parent_2.get_genome()[:point] + parent_1.get_genome()[point:]
        child_1 = copy(parent_1)
        child_2 = copy(parent_2)
        child_1.set_genome(child_1_genome)
        child_2.set_genome(child_2_genome)
        
        return child_1, child_2
        
    def _two_point_crossover(self, parent_1: List[Union[int, float, str]], parent_2: List[Union[int, float, str]]) -> Tuple[List[Union[int, float, str]], List[Union[int, float, str]]]:        
        pass

    def crossover(self, parent_1, parent_2) -> Tuple[List[Union[int, float, str]], List[Union[int, float, str]]]:
        if self.crossover_type == 'one_point':
            return self._one_point_crossover(parent_1, parent_2)
        elif self.crossover_type == 'two_point':
            return self._two_point_crossover(parent_1, parent_2)
        else:
            raise Exception(f'Crossover {self.crossover_type} not supported.')

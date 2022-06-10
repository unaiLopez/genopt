import numpy as np

from typing import List, Union, Tuple

class Crossover:
    def __init__(self, crossover_type: str, search_space_type: str):
        self.crossover_type = crossover_type
        self.search_space_type = search_space_type

    def _one_point_crossover(self, parent_1: List[Union[int, float, str]], parent_2: List[Union[int, float, str]]) -> Tuple[List[Union[int, float, str]], List[Union[int, float, str]]]:
        point = np.random.randint(0, len(parent_1) - 1)
        child_1 = parent_1[:point] + parent_2[point:]
        child_2 = parent_2[:point] + parent_1[point:]
        
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

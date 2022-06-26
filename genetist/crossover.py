from copy import copy
import numpy as np

from typing import Tuple, List, Union
from genetist.individual import Individual

class Crossover:
    __instance = None

    @staticmethod
    def getInstance(crossover_type: str):
        if Crossover.__instance == None:
            Crossover(crossover_type)
        return Crossover.__instance 

    def __init__(self, crossover_type: str):
        Crossover.__instance = self
        self.crossover_type = crossover_type

    def _one_point_crossover(self, genome_1: Individual, genome_2: Individual) -> Tuple[Individual, Individual]:
        point = np.random.randint(1, len(genome_1) - 1)
        
        child_1_genome = genome_1[:point] + genome_2[point:]
        child_2_genome = genome_2[:point] + genome_1[point:]
        
        return child_1_genome, child_2_genome
        
    def _two_point_crossover(self, genome_1: List[Union[int, float, str]], genome_2: List[Union[int, float, str]]) -> Tuple[List[Union[int, float, str]], List[Union[int, float, str]]]:
        max_points = len(genome_1) - 2
        if max_points < 2:
            raise Exception(f'Too much points tried. Try less points for the crossover.')
        else:     
            points = sorted(np.random.randint(1, len(genome_1) - 1, size=2))
            point_1 = points[0]
            point_2 = points[1]
            
            child_1_genome = genome_1[:point_1] + genome_2[point_1:point_2] + genome_1[point_2:]
            child_2_genome = genome_2[:point_1] + genome_1[point_1:point_2] + genome_2[point_2:]

        return child_1_genome, child_2_genome
    
    def _three_point_crossover(self, genome_1: List[Union[int, float, str]], genome_2: List[Union[int, float, str]]) -> Tuple[List[Union[int, float, str]], List[Union[int, float, str]]]:
        max_points = len(genome_1) - 2
        if max_points < 3:
            raise Exception(f'Too much points tried. Try less points for the crossover.')
        else:
            points = sorted(np.random.randint(1, len(genome_1) - 1, size=3))
            point_1 = points[0]
            point_2 = points[1]
            point_3 = points[2]

            child_1_genome = genome_1[:point_1] + genome_2[point_1:point_2] + genome_1[point_2:point_3] +  genome_2[point_3:]
            child_2_genome = genome_2[:point_1] + genome_1[point_1:point_2] + genome_2[point_2:point_3] +  genome_1[point_3:]

        return child_1_genome, child_2_genome
        

    def crossover(self, parent_1: Individual, parent_2: Individual) -> Tuple[Individual, Individual]:
        if self.crossover_type == 'one-point':
            child_1_genome, child_2_genome = self._one_point_crossover(parent_1.genome, parent_2.genome)
        elif self.crossover_type == 'two-point':
            child_1_genome, child_2_genome = self._two_point_crossover(parent_1.genome, parent_2.genome)
        elif self.crossover_type == 'three-point':
            child_1_genome, child_2_genome = self._three_point_crossover(parent_1.genome, parent_2.genome)
        else:
            raise Exception(f'Crossover {self.crossover_type} not supported.')
        
        child_1 = copy(parent_1)
        child_2 = copy(parent_2)
        child_1.genome = child_1_genome
        child_2.genome = child_2_genome

        return child_1, child_2

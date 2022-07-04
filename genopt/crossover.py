import numpy as np

from copy import copy
from typing import Tuple
from genopt.individual import Individual

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

    def _one_point_crossover(self, parent_1: Individual, parent_2: Individual) -> Tuple[Individual, Individual]:
        point = np.random.randint(1, len(parent_1) - 1)
        
        child_1_genome = parent_1.genome[:point] + parent_2.genome[point:]
        child_2_genome = parent_2.genome[:point] + parent_1.genome[point:]
        
        child_1 = copy(parent_1)
        child_2 = copy(parent_2)
        child_1.genome = child_1_genome
        child_2.genome = child_2_genome
        
        return child_1, child_2
        
    def _two_point_crossover(self, parent_1: Individual, parent_2: Individual) -> Tuple[Individual, Individual]:
        max_points = len(parent_1) - 2
        if max_points < 2:
            raise Exception(f'Too much points tried. Try less points for the crossover.')
        else:     
            points = sorted(np.random.randint(1, len(parent_1) - 1, size=2))
            point_1 = points[0]
            point_2 = points[1]
            
            child_1_genome = parent_1.genome[:point_1] + parent_2.genome[point_1:point_2] + parent_1.genome[point_2:]
            child_2_genome = parent_2.genome[:point_1] + parent_1.genome[point_1:point_2] + parent_2.genome[point_2:]

            child_1 = copy(parent_1)
            child_2 = copy(parent_2)
            child_1.genome = child_1_genome
            child_2.genome = child_2_genome
        
        return child_1, child_2
    
    def _three_point_crossover(self, parent_1: Individual, parent_2: Individual) -> Tuple[Individual, Individual]:
        max_points = len(parent_1) - 2
        if max_points < 3:
            raise Exception(f'Too much points tried. Try less points for the crossover.')
        else:
            points = sorted(np.random.randint(1, len(parent_1) - 1, size=3))
            point_1 = points[0]
            point_2 = points[1]
            point_3 = points[2]

            child_1_genome = parent_1.genome[:point_1] + parent_2.genome[point_1:point_2] + parent_1.genome[point_2:point_3] +  parent_2.genome[point_3:]
            child_2_genome = parent_2.genome[:point_1] + parent_1.genome[point_1:point_2] + parent_2.genome[point_2:point_3] +  parent_1.genome[point_3:]

            child_1 = copy(parent_1)
            child_2 = copy(parent_2)
            child_1.genome = child_1_genome
            child_2.genome = child_2_genome
        
        return child_1, child_2
    
    def _uniform_crossover(self, parent_1: Individual, parent_2: Individual) -> Tuple[Individual, Individual]:
        parents_genome = list(zip(*[parent_1.genome, parent_2.genome]))
        child_1_genome = list([np.random.choice(genome) for genome in parents_genome])
        child_2_genome = list([np.random.choice(genome) for genome in parents_genome])

        child_1 = copy(parent_1)
        child_2 = copy(parent_2)
        child_1.genome = child_1_genome
        child_2.genome = child_2_genome

        return child_1, child_2
        
    def crossover(self, parent_1: Individual, parent_2: Individual) -> Tuple[Individual, Individual]:
        if self.crossover_type == 'one-point':
            return self._one_point_crossover(parent_1, parent_2)
        elif self.crossover_type == 'two-point':
            return self._two_point_crossover(parent_1, parent_2)
        elif self.crossover_type == 'three-point':
            return self._three_point_crossover(parent_1, parent_2)
        elif self.crossover_type == 'uniform':
            return self._uniform_crossover(parent_1, parent_2)
        else:
            raise Exception(f'Crossover {self.crossover_type} not supported.')

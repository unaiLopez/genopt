import random
import numpy as np

from typing import List, Tuple
from genopt.individual import Individual

class Selection:
    __instance = None

    @staticmethod
    def getInstance(selection_type: str, tournament_size: int):
        if Selection.__instance == None:
            Selection(selection_type, tournament_size)
        return Selection.__instance 

    def __init__(self, selection_type: str, tournament_size: int):
        Selection.__instance = self
        self.selection_type = selection_type
        self.tournament_size = tournament_size

    def _roulette_selection(self, individuals: List[Individual], number_of_parents: int) -> List[Tuple[Individual, Individual]]:
        all_parents = list()
        for _ in range(number_of_parents):
            weights = np.arange(len(individuals), 0, step=-1)
            parents = random.choices(
                population=individuals,
                weights=weights,
                k=2
            )
            all_parents.append(parents)

        return all_parents

    def _tournament_selection(self, individuals: List[Individual], number_of_parents: int) -> List[Tuple[Individual, Individual]]:
        all_parents = list()
        for _ in range(number_of_parents):
            indexes = sorted(np.random.randint(0, len(individuals) - 1, size=self.tournament_size))
            all_parents.append([individuals[indexes[0]], individuals[indexes[1]]])

        return all_parents
    
    def _ranking_selection(self, individuals: List[Individual], number_of_parents: int) -> List[Tuple[Individual, Individual]]:
        all_parents = list()
        parent_individual = iter(range(number_of_parents))
        parents = zip(parent_individual, parent_individual)
        for parent_index_1, parent_index_2 in parents:
            all_parents.append([individuals[parent_index_1], individuals[parent_index_2]])
        
        return all_parents

    def selection(self, individuals: List[Individual], number_of_parents: int) -> List[Individual]:
        if self.selection_type == 'tournament':
            return self._tournament_selection(individuals, number_of_parents)
        elif self.selection_type == 'roulette':
            return self._roulette_selection(individuals, number_of_parents)
        elif self.selection_type == 'ranking':
            return self._ranking_selection(individuals, number_of_parents)
        else:
            raise Exception(f'Selection {self.selection_type} not supported.')
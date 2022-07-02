import random
import numpy as np

from typing import List, Tuple
from genetist.individual import Individual

class Selection:
    __instance = None

    @staticmethod
    def getInstance(selection_type: str):
        if Selection.__instance == None:
            Selection(selection_type)
        return Selection.__instance 

    def __init__(self, selection_type: str):
        Selection.__instance = self
        self.selection_type = selection_type

    def _roulette_selection(self, individuals: List[Individual]) -> Tuple[Individual, Individual]:
        weights = np.arange(len(individuals), 0, step=-1)
        parents = random.choices(
            population=individuals,
            weights=weights,
            k=2
        )

        return parents[0], parents[1]

    def _tournament_selection(self, individuals: List[Individual], k: int = 5) -> Tuple[Individual, Individual]:
        indexes = sorted(np.random.randint(0, len(individuals) - 1, size=k))

        return individuals[indexes[0]], individuals[indexes[1]]

    def selection(self, individuals: List[Individual]) -> List[Individual]:
        if self.selection_type == 'tournament':
            return self._tournament_selection(individuals)
        elif self.selection_type == 'roulette':
            return self._roulette_selection(individuals)
        else:
            raise Exception(f'Selection {self.selection_type} not supported.')
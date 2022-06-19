import numpy as np

from typing import Callable, Union

class Individual:
    def __init__(self, params: dict, search_space_type: str, objective: Callable[[dict], Union[int,float]]):
        self.params = params
        self.search_space_type = search_space_type
        self.objective = objective
        self._genome = list()
        self._fitness = None
    
    def __len__(self) -> int:
        return len(self._genome)
    
    @property
    def genome(self) -> list:
        return self._genome
    
    @property
    def fitness(self) -> Union[int, float]:
        return self._fitness
    
    @fitness.setter
    def fitness(self, fitness: Union[int, float]) -> None:
        self._fitness = fitness
    
    @genome.setter
    def genome(self, genome: list) -> None:
        self._genome = genome

    def initialize_genome(self) -> list:
        if self.search_space_type == 'flexible_search':
            for _, values in self.params.items():
                if values['type'] == 'int':
                    self._genome.append(np.random.randint(values['low'], values['high'] + 1))
                elif values['type'] == 'float':
                    self._genome.append(np.random.uniform(values['low'], values['high']))
                elif values['type'] == 'categorical':
                    self._genome.append(np.random.choice(values['choices']))
                else:
                    raise ValueError(f'Type {values["type"]} not supported.')
        else:
            for _, values in self.params.items():
                self._genome.append(np.random.choice(values))
        
        return self._genome
    
    def get_name_genome_genes(self) -> dict:
        genome_gene_names = {}
        names = list(self.params.keys())
        for name, gene in zip(names, self._genome):
            genome_gene_names[name] = gene
        
        return genome_gene_names
    
    def calculate_fitness(self) -> None:
        self.fitness = self.objective(self.get_name_genome_genes())
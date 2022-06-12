import numpy as np

from typing import Callable, Union

class Individual:
    def __init__(self, params: dict, search_space_type: str, objective: Callable[[dict], Union[int,float]]):
        self.params = params
        self.search_space_type = search_space_type
        self.objective = objective
        self.genome = self._initialize_genome()
        self.fitness = None
    
    def __len__(self) -> int:
        return len(self.genome)
    
    def _initialize_genome(self) -> list:
        genome = list()
        if self.search_space_type == 'flexible_search':
            for _, values in self.params.items():
                if values['type'] == 'int':
                    genome.append(np.random.random_integers(values['low'], values['high']))
                elif values['type'] == 'float':
                    genome.append(np.random.uniform(values['low'], values['high']))
                elif values['type'] == 'categorical':
                    genome.append(np.random.choice(values['choices']))
                else:
                    raise ValueError(f'Type {values["type"]} not supported.')
        else:
            for _, values in self.params.items():
                genome.append(np.random.choice(values))
        
        return genome
    
    def get_name_genome_genes(self) -> dict:
        genome_gene_names = {}
        names = list(self.params.keys())
        for name, gene in zip(names, self.genome):
            genome_gene_names[name] = gene
        
        return genome_gene_names
    
    def calculate_fitness(self) -> None:
        self.fitness = self.objective(self.get_name_genome_genes())
    
    def get_genome(self) -> list:
        return self.genome
    
    def get_fitness(self) -> Union[int, float]:
        return self.fitness
    
    def set_genome(self, genome: list) -> None:
        self.genome = genome
import numpy as np

from genetist.individual import Individual

class Mutation:
    def __init__(self, mutation_type: str, prob_mutation: float, search_space_type: str, params: dict):
        self.mutation_type = mutation_type
        self.prob_mutation = prob_mutation
        self.search_space_type = search_space_type
        self.params = params
    
    def _fixed_search_mutation_process(self, child: Individual, gene_index: int, param: str) -> Individual:
        child_genome = child.get_genome()
        if len(self.params.get(param)) == 2:
            if self.params.get(param)[0] != child_genome[gene_index]:
                child_genome[gene_index] = self.params.get(param)[0]
            else:
                child_genome[gene_index] = self.params.get(param)[1]
        else:
            child_genome[gene_index] = np.random.choice(self.params.get(param))
        
        child.set_genome(child_genome)
        
        return child

    def _flexible_search_mutation_process(self, child: Individual, gene_index: int, param: str) -> Individual:
        child_genome = child.get_genome()
        if self.params.get(param).get('type') == 'int':
            if self.params[param]['low'] == 0 and self.params[param]['high'] == 1:
                 if child_genome[gene_index] == 0:
                    child_genome[gene_index] = 1
                 else:
                    child_genome[gene_index] = 0
            else:
                child_genome[gene_index] = np.random.random_integers(self.params[param]['low'], (self.params[param]['high']))
        elif self.params.get(param).get('type') == 'float':
            child_genome[gene_index] = np.random.uniform(self.params[param]['low'], self.params[param]['high'])
        elif self.params.get(param).get('type') == 'categorical':
            if len(self.params.get(param).get('choices')) == 2:
                if self.params.get(param).get('choices')[0] != child_genome[gene_index]:
                    child_genome[gene_index] = self.params.get(param).get('choices')[0]
                else:
                    child_genome[gene_index] = self.params.get(param).get('choices')[1]
            else:
                child_genome[gene_index] = np.random.choice(self.params.get(param).get('choices'))

        child.set_genome(child_genome)
        
        return child

    def _single_mutation_in_fixed_search(self, child: Individual) -> Individual:
        gene_index = np.random.randint(0,  len(child))
        param = list(self.params.keys())[gene_index]
        child = self._fixed_search_mutation_process(child, gene_index, param)
            
        return child

    def _single_mutation_in_flexible_search(self, child: Individual) -> Individual:
        gene_index = np.random.randint(0,  len(child))
        param = list(self.params.keys())[gene_index]
        child = self._flexible_search_mutation_process(child, gene_index, param)
        
        return child

    def _multiple_mutation_in_fixed_search(self, child: Individual) -> Individual:
        number_of_mutations = np.random.randint(1, len(child))
        gene_indexes = np.random.randint(0, len(child), size=number_of_mutations)
        for gene_index in gene_indexes:
            param = list(self.params.keys())[gene_index]
            child = self._fixed_search_mutation_process(child, gene_index, param)
        
        return child
                

    def _multiple_mutation_in_flexible_search(self, child: Individual) -> Individual:
        number_of_mutations = np.random.randint(1, len(child))
        gene_indexes = np.random.randint(0, len(child), size=number_of_mutations)
        for gene_index in gene_indexes:
            param = list(self.params.keys())[gene_index]
            child = self._flexible_search_mutation_process(child, gene_index, param)
        
        return child

    def mutate(self, child: Individual) -> Individual:
        if np.random.rand() < self.prob_mutation:
            if self.mutation_type == 'single_gene':
                if self.search_space_type == 'fixed_search':
                    return self._single_mutation_in_fixed_search(child)
                else:
                    return self._single_mutation_in_flexible_search(child)
            elif self.mutation_type == 'multiple_genes':
                if self.search_space_type == 'fixed_search':
                    return self._multiple_mutation_in_fixed_search(child)
                else:
                    return self._multiple_mutation_in_flexible_search(child)
            else:
                raise Exception(f'Mutation type {self.mutation_type} not supported.')
        else:
            return child
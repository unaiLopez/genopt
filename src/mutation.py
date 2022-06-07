import numpy as np

from typing import List, Union

class Mutation:
    def __init__(self, mutation_type: str, prob_mutation: float, search_space_type: str, params: dict):
        self.mutation_type = mutation_type
        self.prob_mutation = prob_mutation
        self.search_space_type = search_space_type
        self.params = params
    
    def _fixed_search_mutation_process(self, child: List[Union[int, float, str]], gene_index: int, param: str) -> List[Union[int, float, str]]:
        if len(self.params.get(param)) == 2:
            if self.params.get(param)[0] != child[gene_index]:
                child[gene_index] = self.params.get(param)[0]
            else:
                child[gene_index] = self.params.get(param)[1]
        else:
            child[gene_index] = np.random.choice(self.params.get(param))
        
        return child

    def _flexible_search_mutation_process(self, child: List[Union[int, float, str]], gene_index: int, param: str) -> List[Union[int, float, str]]:
        if self.params.get(param).get('type') == 'int':
            if self.params[param]['low'] == 0 and self.params[param]['high'] == 1:
                 if child[gene_index] == 0:
                    child[gene_index] = 1
                 else:
                    child[gene_index] = 0
            else:
                child[gene_index] = np.random.random_integers(self.params[param]['low'], (self.params[param]['high']))
        elif self.params.get(param).get('type') == 'float':
            child[gene_index] = np.random.uniform(self.params[param]['low'], self.params[param]['high'])
        elif self.params.get(param).get('type') == 'categorical':
            if len(self.params.get(param).get('choices')) == 2:
                if self.params.get(param).get('choices')[0] != child[gene_index]:
                    child[gene_index] = self.params.get(param).get('choices')[0]
                else:
                    child[gene_index] = self.params.get(param).get('choices')[1]
            else:
                child[gene_index] = np.random.choice(self.params.get(param).get('choices'))
        
        return child

    def _single_mutation_in_fixed_search(self, child: List[Union[int, float, str]]) -> List[Union[int, float, str]]:
        if np.random.rand() < self.prob_mutation:
            gene_index = np.random.randint(0,  len(child)-1)
            param = list(self.params.keys())[gene_index]
            child = self._fixed_search_mutation_process(child, gene_index, param)
            
        return child

    def _single_mutation_in_flexible_search(self, child: List[Union[int, float, str]]) -> List[Union[int, float, str]]:
        if np.random.rand() < self.prob_mutation:
            gene_index = np.random.randint(0,  len(child)-1)
            param = list(self.params.keys())[gene_index]
            child = self._flexible_search_mutation_process(child, gene_index, param)
        
        return child

    def _multiple_mutation_in_fixed_search(self, child: List[Union[int, float, str]]) -> List[Union[int, float, str]]:
        if np.random.rand() < self.prob_mutation:
            for gene_index in range(len(child)):
                param = list(self.params.keys())[gene_index]
                child = self._fixed_search_mutation_process(child, gene_index, param)
        
        return child
                

    def _multiple_mutation_in_flexible_search(self, child: List[Union[int, float, str]]) -> List[Union[int, float, str]]:
        if np.random.rand() < self.prob_mutation:
            for gene_index in range(len(child)):
                param = list(self.params.keys())[gene_index]
                child = self._flexible_search_mutation_process(child, gene_index, param)
        
        return child

    def mutate(self, child: List[Union[int, float, str]]) -> List[Union[int, float, str]]:
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
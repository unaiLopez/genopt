import unittest

from genetist.mutation import Mutation
from genetist.params import Params

class TestMutation(unittest.TestCase):
    def setUp(self):
        self.fixed_params = {
            'x': [0, 1],
            'y': [0, 1, 3, 10, -10],
            'z': [-2.1, -2.9, 1.3]
        }
        self.flexible_params = {
            'x': Params.suggest_int(0, 10),
            'y': Params.suggest_categorical(['hello', 'goodbye']),
            'z': Params.suggest_float(-10, 20)
        }
        self.flexible_params_multiple_categorical = {
            'x': Params.suggest_int(0, 10),
            'y': Params.suggest_categorical(['hello', 'goodbye', 'hey', 'how are you', 'hi']),
            'z': Params.suggest_float(-10, 20)
        }
        self.single_gene_mutation_fixed_search = Mutation('single-gene', 1.0, 'fixed_search', self.fixed_params)
        self.single_gene_mutation_flexible_search = Mutation('single-gene', 1.0, 'flexible_search', self.flexible_params)
        self.single_gene_mutation_flexible_search_multiple_categoricals = Mutation('single-gene', 1.0, 'flexible_search', self.flexible_params_multiple_categorical)

    def test_mutate_fixed_binary_param(self):
        gene_index = 0
        param = 'x'

        failures = list()
        for i in range(1):
            child_genome = [0, 1, -2.9]
            mutated_genome = self.single_gene_mutation_fixed_search._mutate_fixed_binary_param(child_genome, gene_index, param)
            if mutated_genome != [1, 1, -2.9]:
                failures.append(i)

        self.assertEqual(failures, list())

    def test_mutate_flexible_int_param(self):
        gene_index = 0
        param = 'x'
        failures = list()
        for i in range(100):
            child_genome = [4, 'hello', -9.33]
            mutated_genome = self.single_gene_mutation_flexible_search._mutate_flexible_int_param(child_genome, gene_index, param)
            if mutated_genome[gene_index] == 4:
                failures.append(i)

        self.assertEqual(failures, list())

    def test_mutate_flexible_categorical_binary_param(self):
        gene_index = 1
        param = 'y'
        failures = list()
        for i in range(100):
            child_genome = [4, 'hello', -9.33]
            mutated_genome = self.single_gene_mutation_flexible_search._mutate_flexible_categorical_param(child_genome, gene_index, param)
            if mutated_genome[gene_index] == 'hello':
                failures.append(i)

        self.assertEqual(failures, list())
    
    def test_mutate_flexible_categorical_non_binary_param(self):
        gene_index = 1
        param = 'y'
        failures = list()
        for i in range(100):
            child_genome = [4, 'hello', -9.33]
            mutated_genome = self.single_gene_mutation_flexible_search_multiple_categoricals._mutate_flexible_categorical_param(child_genome, gene_index, param)
            if mutated_genome[gene_index] == 'hello':
                failures.append(i)
                
        self.assertEqual(failures, list())

    def test_mutate_fixed_non_binary_param(self):
        gene_index = 1
        param = 'y'
        failures = list()
        for i in range(100):
            child_genome = [0, 3, -2.1]
            mutated_genome = self.single_gene_mutation_fixed_search._mutate_fixed_non_binary_param(child_genome, gene_index, param)
            if mutated_genome[gene_index] == 3:
                failures.append(i)

        self.assertEqual(failures, list())

if __name__ == '__main__':
    unittest.main()
import unittest

from genetist.individual import Individual

class TestIndividual(unittest.TestCase):
    def setUp(self):
        def objective(individual):
            return 4
        self.params = {
            'x': [0, 1, 2, 3, 5],
            'y': [11, 12, 13],
        }
        self.search_space_type = 'fixed_search'
        self.individual = Individual(self.params, self.search_space_type, objective)
    
    def test_initialize_genome(self):
        self.individual.initialize_genome()

        self.assertTrue(isinstance(self.individual.genome, list))
    
    def test_get_genome_length(self):
        self.assertEqual(len(self.individual), 0)

    def test_get_genome(self):
        self.assertTrue(isinstance(self.individual.genome, list))
    
    def test_get_fitness(self):
        self.assertEqual(self.individual.fitness, None)
    
    def test_set_genome(self):
        new_genome = [1]

        self.individual.genome = new_genome

        self.assertEqual(self.individual.genome, new_genome)
    
    def test_set_fitness(self):
        new_fitness = 2

        self.individual.fitness = new_fitness

        self.assertEqual(self.individual.fitness, new_fitness)
    
    def test_get_name_genome_genes_contains_x(self):
        genome = [1, 2]
        result_gene = 'x'

        self.individual.genome = genome
        genome_gene_names = self.individual.get_name_genome_genes()
        names = list(genome_gene_names.keys())

        self.assertEqual(names[0], result_gene)
        
    def test_get_name_genome_genes_contains_y(self):
        genome = [1, 2]
        result_gene = 'y'

        self.individual.genome = genome
        genome_gene_names = self.individual.get_name_genome_genes()
        names = list(genome_gene_names.keys())

        self.assertEqual(names[1], result_gene)
    
    def test_get_name_genome_genes_contains_two_genes(self):
        genome = [1, 2]
        genome_length = 2

        self.individual.genome = genome
        genome_gene_names = self.individual.get_name_genome_genes()
        names = list(genome_gene_names.keys())

        self.assertEqual(len(names), genome_length)

    def test_get_name_genome_genes_is_dict(self):
        genome = [1, 2]
        
        self.individual.genome = genome
        genome_gene_names = self.individual.get_name_genome_genes()

        self.assertTrue(isinstance(genome_gene_names, dict))
    
    def test_get_name_genome_genes_is_dict_with_x_and_y(self):
        genome = [1, 2]
        result_genome_gene_names = {'x': 1, 'y': 2}

        self.individual.genome = genome
        genome_gene_names = self.individual.get_name_genome_genes()

        self.assertEqual(genome_gene_names, result_genome_gene_names)
    
    def test_calculate_fitness(self):
        result_fitness = 4
        
        self.individual.calculate_fitness()

        self.assertEqual(self.individual.fitness, result_fitness)

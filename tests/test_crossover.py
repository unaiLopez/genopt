import unittest
from unittest.mock import Mock

from genetist.crossover import Crossover

class TestCrossover(unittest.TestCase):
    def setUp(self):
        self.one_point_crossover = Crossover('one-point')
        self.two_point_crossover = Crossover('two-point')
        self.three_point_crossover = Crossover('three-point')
        self.not_supported_crossover = Crossover('unknown')

        self.mock_individual_1 = Mock()
        self.mock_individual_2 = Mock()
        self.mock_individual_1_short_genome = Mock()
        self.mock_individual_2_short_genome = Mock()
        self.mock_individual_1.__len__ = Mock(return_value=4)
        self.mock_individual_2.__len__ = Mock(return_value=4)
        self.mock_individual_1_short_genome.__len__ = Mock(return_value=1)
        self.mock_individual_2_short_genome.__len__ = Mock(return_value=1)
        self.mock_individual_1.get_genome = Mock(return_value=[1, 3, 10, 2])
        self.mock_individual_2.get_genome = Mock(return_value=[7, 1, 31, 1])
        self.mock_individual_1_short_genome.get_genome = Mock(return_value=[1])
        self.mock_individual_2_short_genome.get_genome = Mock(return_value=[7])


    def test_one_point_crossover_succesfull(self):
        child_1, child_2 = self.one_point_crossover._one_point_crossover(self.mock_individual_1, self.mock_individual_2)
        self.assertTrue(isinstance(child_1.get_genome(), list))
        self.assertTrue(isinstance(child_2.get_genome(), list))

    def test_two_point_crossover_succesfull(self):
        child_1, child_2 = self.one_point_crossover._one_point_crossover(self.mock_individual_1, self.mock_individual_2)
        self.assertTrue(isinstance(child_1.get_genome(), list))
        self.assertTrue(isinstance(child_2.get_genome(), list))
    
    def test_crossover_succesfull(self):
        child_1, child_2 = self.one_point_crossover.crossover(self.mock_individual_1, self.mock_individual_2)
        self.assertTrue(isinstance(child_1.get_genome(), list))
        self.assertTrue(isinstance(child_2.get_genome(), list))

    def test_two_point_crossover_fails_for_genome_shorter_than_two_genes(self):
        with self.assertRaises(Exception):
            self.one_point_crossover._one_point_crossover(self.mock_individual_1_short_genome, self.mock_individual_2_short_genome)

    def test_three_point_crossover__fails_for_genome_shorter_than_three_genes(self):
         with self.assertRaises(Exception):
            self.one_point_crossover._one_point_crossover(self.mock_individual_1_short_genome, self.mock_individual_2_short_genome)

    def test_crossover_fails_for_unknown_mutation_type(self):
        with self.assertRaises(Exception):
             self.not_supported_crossover.crossover(self.mock_individual_1, self.mock_individual_2)

import sys
sys.path.append('../')

import pytest
import numpy as np
from genetist.individual import Individual

params = {
    'x': np.arange(-10, 10),
    'y': np.arange(-10, 10),
    'z': np.arange(-10, 10),
    'k': np.arange(-10, 10),

}
search_space_type = 'fixed_search'

def objective(individual):
    return individual['x'] + individual['y'] + individual['z'] + individual['k']

def test_initialize_genome():
    pass

def test_get_name_genome_genes():
    pass

def test_calculate_fitness():
    pass
import unittest
import pandas as pd

from genetist.results import Results

class TestResults(unittest.TestCase):
    def setUp(self):
        self.results = Results()
    
    def test_get_best_score(self):
        self.assertEqual(self.results.best_score, None)
    
    def test_set_best_score(self):
        score = 100

        self.results.best_score = score

        self.assertEqual(self.results.best_score, score)
    
    def test_get_best_individual(self):
        self.assertEqual(self.results.best_individual, None)
    
    def test_set_best_individual(self):
        best_individual = {'x': 100, 'y': 3}

        self.results.best_individual = best_individual

        self.assertEqual(self.results.best_individual, best_individual)
    
    def test_get_execution_time(self):
        self.assertEqual(self.results.execution_time, None)

    def test_set_execution_time(self):
        time = 10
        execution_time_string = '0 hours 00 minutes 10 seconds'

        self.results.execution_time = time

        self.assertEqual(self.results.execution_time, execution_time_string)
    
    def test_best_per_generation_dataframe(self):
        self.assertTrue(isinstance(self.results.best_per_generation_dataframe, pd.DataFrame))

    def test_add_generation_results(self):
        generation = 1
        best_score = 10
        best_individual = {'x': 5, 'y': 5}
        length_df_after_adding_a_result = 1

        self.results.add_generation_results(generation, best_score, best_individual)

        self.assertTrue(len(self.results.best_per_generation_dataframe), length_df_after_adding_a_result)

    def test_sort_best_per_generation_dataframe_not_supported_direction_fails(self):
        #TODO
        self.assertTrue(False, True)


    def test_sort_best_per_generation_dataframe_minimize(self):
        #TODO
        self.assertTrue(False, True)

    def test_sort_best_per_generation_dataframe_maximize(self):
        #TODO
        self.assertTrue(False, True)

if __name__ == '__main__':
    unittest.main()
 
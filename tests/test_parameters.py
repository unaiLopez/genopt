import unittest
from genetist.parameters import Parameters

class TestParameters(unittest.TestCase):

    def test_suggest_int_returns_well_structured_dict(self):
        dict_int = Parameters.suggest_int(1, 10)
        dict_result = {'type': 'int', 'low': 1, 'high': 10}

        self.assertTrue(dict_int, dict_result)

    def test_suggest_float_returns_well_structured_dict(self):
        dict_float = Parameters.suggest_float(1, 10)
        dict_result = {'type': 'float', 'low': 1, 'high': 10}
        
        self.assertTrue(dict_float, dict_result)
    
    def test_suggest_categorical_returns_well_structured_dict(self):
        choices = ['Hello', 'Goodbye']
        dict_categorical = Parameters.suggest_categorical(choices)
        dict_result = {'type': 'categorical', 'choices': choices}
        
        self.assertTrue(dict_categorical == dict_result)

    def test_suggest_int_fails_with_non_float_and_non_integer_values(self):
         with self.assertRaises(Exception):
            Parameters.suggest_int('hi', dict())
         with self.assertRaises(Exception):
            Parameters.suggest_int(1, 'hi')
         with self.assertRaises(Exception):
            Parameters.suggest_int('hi', 3)
         with self.assertRaises(Exception):
            Parameters.suggest_int(list(), 'hi')
         with self.assertRaises(Exception):
            Parameters.suggest_int(set(), 2)
         with self.assertRaises(Exception):
            Parameters.suggest_int(dict(), set())

    def test_suggest_float_fails_with_non_float_and_non_integer_values(self):
        with self.assertRaises(Exception):
            Parameters.suggest_float('hi', dict())
        with self.assertRaises(Exception):
            Parameters.suggest_float(1, 'hi')
        with self.assertRaises(Exception):
            Parameters.suggest_float('hi', 3)
        with self.assertRaises(Exception):
            Parameters.suggest_float(list(), 'hi')
        with self.assertRaises(Exception):
            Parameters.suggest_float(set(), 2)
        with self.assertRaises(Exception):
             Parameters.suggest_float(dict(), set())

    def test_suggest_categorical_fails_with_non_string_values(self):
        with self.assertRaises(Exception):
            Parameters.suggest_categorical([1, 3, 'hi'])
        with self.assertRaises(Exception):
            Parameters.suggest_categorical(['hello', 3, 'hi'])
        with self.assertRaises(Exception):
            Parameters.suggest_categorical(['hello', 'hey', list()])

if __name__ == '__main__':
    unittest.main()
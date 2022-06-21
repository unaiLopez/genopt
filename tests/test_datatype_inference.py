import unittest
from genetist.datatype_inference import DataTypeInference

class TestDataTypeInference(unittest.TestCase):
    def setUp(self):
        self.flexible_search_params = {
            'x': {'type': 'int', 'low': 1, 'high': 10},
            'y': {'type': 'categorical', 'choices': ['max_depth', 'n_estimators']},
            'z': {'type': 'float', 'low': 1, 'high': 10}
        }
        self.fixed_search_params = {
            'x': [1, 2, 3, 4],
            'y': ['max_depth', 'n_estimators'],
            'z': [1.0, 1.5, 2, 4.5, 7.2]
        }
        self.mixed_search_type_structure_params = {
            'x': [1, 2, 3, 4],
            'y': {'type': 'categorical', 'choices': ['max_depth', 'n_estimators']},
            'z': [1.0, 1.5, 2, 4.5, 7.2]
        }

    def test_infer_search_space_type_flexible_search_type(self):
        self.assertEqual(DataTypeInference.infer_search_space_type(self.flexible_search_params), 'flexible_search')

    def test_infer_search_space_type_fixed_search_type(self):
        self.assertEqual(DataTypeInference.infer_search_space_type(self.fixed_search_params), 'fixed_search')
    
    def test_infer_search_space_type_mixed_search_type_structure_params_fails(self):
        with self.assertRaises(Exception):
            DataTypeInference.infer_search_space_type(self.mixed_search_type_structure_params)

if __name__ == '__main__':
    unittest.main()
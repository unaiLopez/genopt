import numpy as np

class DataTypeInference:

    @staticmethod
    def infer_search_space_type(params: dict) -> str:
        key = list(params.keys())[0]
        if isinstance(params[key], dict):
            search_space_type = 'flexible_search'
        elif isinstance(params[key], list) or isinstance(params[key], np.ndarray):
            search_space_type = 'fixed_search'
        else:
            raise Exception('Unable to infer search space type. Please check your search space format.')
    
        return search_space_type
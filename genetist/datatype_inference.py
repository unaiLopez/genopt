import numpy as np

class DataTypeInference:
    
    @staticmethod
    def infer_search_space_type(params: dict) -> str:
        keys = list(params.keys())
        for i, key in enumerate(keys):
            if isinstance(key, str) == False and isinstance(key, int) == False:
                raise Exception('Params keys must be string or int type.')
            else:
                if i == 0 and isinstance(params[key], dict):
                    search_space_type = 'flexible_search'
                elif i == 0 and isinstance(params[key], list) or isinstance(params[key], np.ndarray):
                    search_space_type = 'fixed_search'
                if search_space_type == 'flexible_search' and (isinstance(params[key], list) or isinstance(params[key], np.ndarray)):
                    raise Exception('Unable to infer search space type. Please check your search space format.')
                elif search_space_type == 'fixed_search' and isinstance(params[key], dict):
                    raise Exception('Unable to infer search space type. Please check your search space format.')
        
        return search_space_type
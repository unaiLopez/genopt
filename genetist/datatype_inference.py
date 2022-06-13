class DataTypeInference:
    def __init__(self, params: dict):
        self.params = params
        self.search_space_type = self.infer_search_space_type()

    def infer_search_space_type(self) -> str:
        key = list(self.params.keys())[0]
        if isinstance(self.params[key], dict):
            search_space_type = 'flexible_search'
        elif isinstance(self.params[key], list):
            search_space_type = 'fixed_search'
        else:
            raise Exception('Unable to infer search space type. Please check your search space format.')
    
        return search_space_type
    
    def infer_param_types(self) -> dict:
        if self.search_space_type == 'flexible_search':
            keys = list(self.params.keys())
            for key in keys:
                subdict = self.params.get(key)
                subkeys = subdict.keys()
                if 'low' in subkeys and 'high' in subkeys:
                    if isinstance(subdict.get('low'), int) and isinstance(subdict.get('high'), int):
                        self.params[key]['type'] = 'int'
                    elif isinstance(subdict.get('low'), float) and isinstance(subdict.get('high'), float):
                        self.params[key]['type'] = 'float'
                    else:
                        raise Exception('Unable to infer parameters type. Please check your parameters format.')
                elif 'choices' in subkeys:
                    if isinstance(subdict.get('choices')[0], str):
                        self.params[key]['type'] = 'categorical'
                    else:
                        raise Exception('Unable to infer parameters type. Please check your parameters format.')

        return self.params

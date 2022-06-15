from typing import List

class Params:

    @staticmethod
    def suggest_categorical(choices: List[str]) -> dict:
        for choice in choices:
            if isinstance(choice, str) == False:
                raise Exception(f'Choices must be string.')
        return {'type': 'categorical', 'choices': choices}

    @staticmethod
    def suggest_int(low: int, high: int) -> dict:
        if isinstance(low, int) == False and isinstance(low, float) == False:
            raise Exception(f'Low must be a number.')
        elif isinstance(high, int) == False and isinstance(high, float) == False:
            raise Exception(f'High must be a number.')
        else:
            return {'type': 'int', 'low': low, 'high': high}

    @staticmethod
    def suggest_float(low: float, high: float) -> dict:
        if isinstance(low, float) == False and isinstance(low, int) == False:
            raise Exception(f'Low must be float.')
        elif isinstance(high, float) == False and isinstance(high, int) == False:
            raise Exception(f'High must be a number.')
        else:
            return {'type': 'float', 'low': low, 'high': high}
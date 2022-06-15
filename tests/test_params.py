import sys
sys.path.append('../')

import pytest
from genetist.params import Params

def test_suggest_int_returns_well_structured_dict():
    dict_int = Params.suggest_int(1, 10)
    dict_result = {'type': 'int', 'low': 1, 'high': 10}

    assert dict_int == dict_result

def test_suggest_int_fails_with_non_float_and_non_integer_values():
    with pytest.raises(Exception):
        dict_int = Params.suggest_int('hi', dict())
    with pytest.raises(Exception):
        dict_int = Params.suggest_int(1, 'hi')
    with pytest.raises(Exception):
        dict_int = Params.suggest_int('hi', 3)
    with pytest.raises(Exception):
        dict_int = Params.suggest_int(list(), 'hi')
    with pytest.raises(Exception):
        dict_int = Params.suggest_int(set(), 2)
    with pytest.raises(Exception):
        dict_int = Params.suggest_int(dict(), set())

def test_suggest_float_returns_well_structured_dict():
    dict_float = Params.suggest_float(1, 10)
    dict_result = {'type': 'float', 'low': 1, 'high': 10}
    
    assert dict_float == dict_result

def test_suggest_float_fails_with_non_float_and_non_integer_values():
    with pytest.raises(Exception):
        dict_float = Params.suggest_float('hi', dict())
    with pytest.raises(Exception):
        dict_float = Params.suggest_float(1, 'hi')
    with pytest.raises(Exception):
        dict_float = Params.suggest_float('hi', 3)
    with pytest.raises(Exception):
        dict_float = Params.suggest_float(list(), 'hi')
    with pytest.raises(Exception):
        dict_float = Params.suggest_float(set(), 2)
    with pytest.raises(Exception):
        dict_float = Params.suggest_float(dict(), set())

def test_suggest_categorical_returns_well_structured_dict():
    choices = ['Hello', 'Goodbye']
    dict_categorical = Params.suggest_categorical(choices)
    dict_result = {'type': 'categorical', 'choices': choices}
    
    assert dict_categorical == dict_result

def test_suggest_categorical_fails_with_non_string_values():
    with pytest.raises(Exception):
        choices = [1, 3, 'hi']
        dict_categorical = Params.suggest_categorical(choices)
    with pytest.raises(Exception):
        choices = ['hello', 3, 'hi']
        dict_categorical = Params.suggest_categorical(choices)
    with pytest.raises(Exception):
        choices = ['hello', 'hey', list()]
        dict_categorical = Params.suggest_categorical(choices)

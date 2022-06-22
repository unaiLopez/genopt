import pandas as pd
import datetime
import logging

from typing import Union, List

SCORE_COLUMN_DEFAULT_NAME = 'best_score'
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('RESULTS')

class Results:
    def __init__(self):
        self._best_score = None
        self._best_individual = None
        self._execution_time = None
        self._best_per_generation_dataframe = pd.DataFrame()
    
    @property
    def best_score(self):
        return self._best_score
    
    @property
    def execution_time(self):
        return self._execution_time
    
    @property
    def best_individual(self):
        return self._best_individual
    
    @property
    def best_per_generation_dataframe(self):
        return self._best_per_generation_dataframe
    
    @best_score.setter
    def best_score(self, best_score):
        self._best_score = best_score
    
    @execution_time.setter
    def execution_time(self, execution_time):
        time = str(datetime.timedelta(seconds=execution_time))
        hours = time.split(':')[0]
        minutes = time.split(':')[1]
        seconds = time.split(':')[2]
        self._execution_time = f'{hours} hours {minutes} minutes {seconds} seconds'
    
    @best_individual.setter
    def best_individual(self, best_individual):
        self._best_individual = best_individual
    
    @best_per_generation_dataframe.setter
    def best_per_generation_dataframe(self, best_per_generation_dataframe):
        self._best_per_generation_dataframe = best_per_generation_dataframe
        
    def add_generation_results(self, generation: int, best_score: Union[int, float], best_individual: dict) -> None:
        generation_results = {'generation': generation, 'best_score': best_score}
        generation_results.update(best_individual)
        values = [list(generation_results.values())]
        columns = list(generation_results.keys())
        df_generation_results = pd.DataFrame(values, columns=columns)
        self.best_per_generation_dataframe = pd.concat([self.best_per_generation_dataframe, df_generation_results], axis=0)
    
    def _sort_best_per_generation_dataframe_by_single_objective(self, direction, score_names):
        if direction == 'maximize':
                self.best_per_generation_dataframe = self.best_per_generation_dataframe.sort_values(by=SCORE_COLUMN_DEFAULT_NAME, ascending=False)
        elif direction == 'minimize':
                self.best_per_generation_dataframe = self.best_per_generation_dataframe.sort_values(by=SCORE_COLUMN_DEFAULT_NAME, ascending=True)
        else:
            raise Exception(f'Direction {direction} is not supported.')
        
        if isinstance(score_names, str):
            self.best_per_generation_dataframe.rename(columns={
                SCORE_COLUMN_DEFAULT_NAME: score_names
            }, inplace=True)
    
    def _normalize_best_score_by_index(self, direction, i):
        if direction[i] == 'maximize':
            self.best_per_generation_dataframe[f'normalized_best_score_{i}'] = (self.best_per_generation_dataframe[f'best_score_{i}'] - self.best_per_generation_dataframe[f'best_score_{i}'].min()) / (self.best_per_generation_dataframe[f'best_score_{i}'].max() - self.best_per_generation_dataframe[f'best_score_{i}'].min())
        elif direction[i] == 'minimize':
            self.best_per_generation_dataframe[f'normalized_best_score_{i}'] = (self.best_per_generation_dataframe[f'best_score_{i}'] - self.best_per_generation_dataframe[f'best_score_{i}'].max()) / (self.best_per_generation_dataframe[f'best_score_{i}'].min() - self.best_per_generation_dataframe[f'best_score_{i}'].max())
        else:
            raise Exception(f'Direction {direction} is not supported.')
    
    def _rename_best_score_name_by_index(self, score_names, i):
        if score_names != None:
            self.best_per_generation_dataframe.rename(columns={
                f'best_score_{i}': score_names[i]
            }, inplace=True)
    
    def _calculate_weighted_sum_score_by_index(self, weights, i):
        if i == 0:
            self.best_per_generation_dataframe['overall_best_score'] = 0
            self.best_per_generation_dataframe['overall_best_score'] = self.best_per_generation_dataframe[f'normalized_best_score_{i}'].values * weights[i]
        else:
            self.best_per_generation_dataframe['overall_best_score'] = self.best_per_generation_dataframe['overall_best_score'].values + self.best_per_generation_dataframe[f'normalized_best_score_{i}'].values * weights[i]

    def _define_weights_by_default_if_not_defined(self, weights, direction):
        if weights == None:
            weights = [1 / len(direction)] * len(direction)
            logger.warning(f'Weights value is None. Weights will be defined as {weights}')

    def _sort_best_per_generation_dataframe_by_multiple_objectives(self, direction, weights, score_names):
        if len(direction) == len(weights):
            number_of_fitnesses = len(self.best_per_generation_dataframe['best_score'].iloc[0])
            if number_of_fitnesses != len(direction):
                raise Exception(f'Direction and weights do not match number of fitness values.')
            if score_names != None and number_of_fitnesses != len(score_names):
                raise Exception(f'Score_names does not match number of fitness values.')
            self._define_weights_by_default_if_not_defined(weights, direction)
            for i in range(number_of_fitnesses):
                self.best_per_generation_dataframe[f'best_score_{i}'] = self.best_per_generation_dataframe['best_score'].apply(lambda best_score: best_score[i])
                self._normalize_best_score_by_index(direction, i)
                self._rename_best_score_name_by_index(score_names, i)
                self._calculate_weighted_sum_score_by_index(weights, i)
                self.best_per_generation_dataframe.drop(f'normalized_best_score_{i}', axis=1, inplace=True)
            self.best_per_generation_dataframe = self.best_per_generation_dataframe.sort_values(by='overall_best_score', ascending=False)
            self.best_per_generation_dataframe.drop(['best_score', 'overall_best_score'], axis=1, inplace=True)
        else:
            raise Exception(f'Direction length does not match weights length.')
            

    def sort_best_per_generation_dataframe(self, direction: Union[str, List[str]], weights: List[Union[int, float]] = None, score_names: Union[str, List[str]] = None) -> None:
        if isinstance(direction, str):
            if isinstance(score_names, str) or score_names == None:
                self._sort_best_per_generation_dataframe_by_single_objective(direction, score_names)
            else:
                raise Exception(f'Score_names must be string or None type for single objective optimization.')
        elif isinstance(direction, list):
            if isinstance(weights, list) or weights == None or isinstance(score_names, list) or score_names == None:
                self._sort_best_per_generation_dataframe_by_multiple_objectives(direction, weights, score_names)
            else:
                raise Exception(f'Weights and score_names must be list or None type for multiple objective optimization.')
        else:
            raise Exception(f'Direction {direction} not supported. Must be of type str or List[str]')
        
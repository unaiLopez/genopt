import pandas as pd
import datetime

from typing import Union

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
        self.best_per_generation_dataframe = pd.concat([self._best_per_generation_dataframe, df_generation_results], axis=0)

    def sort_best_per_generation_dataframe(self, column: str, direction: str) -> None:
        if direction == 'maximize':
            self._best_per_generation_dataframe = self._best_per_generation_dataframe.sort_values(by=column, ascending=False)
        elif direction == 'minimize':
            self._best_per_generation_dataframe = self._best_per_generation_dataframe.sort_values(by=column, ascending=True)
        else:
            raise Exception(f'Direction {direction} not supported.')
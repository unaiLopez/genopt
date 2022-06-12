import pandas as pd
import datetime

from typing import Union

class Results:
    def __init__(self):
        self.best_score = None
        self.best_individual = None
        self.execution_time = None
        self.best_per_generation_dataframe = pd.DataFrame()
    
    def get_best_score(self) -> None:
        self.best_score = self.best_per_generation_dataframe['best_score'].max()

    def get_best_individual(self) -> None:
        best_generation = self.best_per_generation_dataframe[self.best_per_generation_dataframe['best_score'] == self.best_per_generation_dataframe['best_score'].max()]
        self.best_individual = best_generation.iloc[0,2:].to_dict()
    
    def set_execution_time(self, execution_time: float) -> None:
        self.execution_time = execution_time
        time = str(datetime.timedelta(seconds=self.execution_time))
        hours = time.split(':')[0]
        minutes = time.split(':')[1]
        seconds = time.split(':')[2]
        self.execution_time = f'{hours} hours {minutes} minutes {seconds} seconds'
    
    def add_generation_results(self, generation: int, best_score: Union[int, float], best_individual: dict) -> None:
            generation_results = {'generation': generation, 'best_score': best_score}
            generation_results.update(best_individual)
            values = [list(generation_results.values())]
            columns = list(generation_results.keys())
            df_generation_results = pd.DataFrame(values, columns=columns)
            self.best_per_generation_dataframe = pd.concat([self.best_per_generation_dataframe, df_generation_results], axis=0)

    def sort_best_per_generation_dataframe(self, column: str, direction: str) -> None:
        if direction == 'maximize':
            self.best_per_generation_dataframe = self.best_per_generation_dataframe.sort_values(by=column, ascending=False)
        elif direction == 'minimize':
            self.best_per_generation_dataframe = self.best_per_generation_dataframe.sort_values(by=column, ascending=True)
        else:
            raise Exception(f'Direction {direction} not supported.')
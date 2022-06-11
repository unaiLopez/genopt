import pandas as pd
import datetime

class Results:
    def __init__(self):
        self.best_score = None
        self.best_individual = None
        self.execution_time = None
        self.best_per_generation_dataframe = pd.DataFrame()
    
    def get_best_score(self):
        self.best_score = self.best_per_generation_dataframe['BEST_SCORE'].max()

    def get_best_individual(self):
        best_generation = self.best_per_generation_dataframe[self.best_per_generation_dataframe['BEST_SCORE'] == self.best_per_generation_dataframe['BEST_SCORE'].max()]
        self.best_individual = best_generation.iloc[0,2:].to_dict()
    
    def set_execution_time(self, execution_time):
        self.execution_time = execution_time
        time = str(datetime.timedelta(seconds=self.execution_time))
        hours = time.split(':')[0]
        minutes = time.split(':')[1]
        seconds = time.split(':')[2]
        self.execution_time = f'{hours} hours {minutes} minutes {seconds} seconds'
    
    def add_generation_results(self, generation, best_score, best_individual):
        generation_results = {'GENERATION': generation, 'BEST_SCORE': best_score}
        generation_results.update(best_individual)
        self.best_per_generation_dataframe = self.best_per_generation_dataframe.append(generation_results, ignore_index=True)

    def sort_best_per_generation_dataframe(self, column, direction):
        if direction == 'maximize':
            self.best_per_generation_dataframe = self.best_per_generation_dataframe.sort_values(by=column, ascending=False)
        elif direction == 'minimize':
            self.best_per_generation_dataframe = self.best_per_generation_dataframe.sort_values(by=column, ascending=True)
        else:
            raise Exception(f'Direction {direction} not supported.')
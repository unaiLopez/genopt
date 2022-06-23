import logging
import pandas as pd

from typing import List, Union
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('RESULTS')

def normalize_best_score_by_index(df: pd.DataFrame, direction: List[str], i: int) -> pd.DataFrame:
    if direction[i] == 'maximize':
        df[f'normalized_best_score_{i}'] = (df[f'best_score_{i}'] - df[f'best_score_{i}'].min()) / (df[f'best_score_{i}'].max() - df[f'best_score_{i}'].min())
    elif direction[i] == 'minimize':
        df[f'normalized_best_score_{i}'] = (df[f'best_score_{i}'] - df[f'best_score_{i}'].max()) / (df[f'best_score_{i}'].min() - df[f'best_score_{i}'].max())
    else:
        raise Exception(f'Direction {direction} is not supported.')
    
    return df

def rename_best_score_name_by_index(df: pd.DataFrame, score_names: Union[List[str], None], i: int) -> pd.DataFrame:
    if score_names != None:
        df.rename(columns={
            f'best_score_{i}': score_names[i]
        }, inplace=True)
    
    return df

def calculate_weighted_sum_score_by_index(df: pd.DataFrame, weights: List[float], i: int) -> pd.DataFrame:
    if i == 0:
        df['overall_best_score'] = 0
        df['overall_best_score'] = df[f'normalized_best_score_{i}'].values * weights[i]
    else:
        df['overall_best_score'] = df['overall_best_score'].values + df[f'normalized_best_score_{i}'].values * weights[i]
    
    return df

def define_weights_by_default_if_not_defined(weights: Union[List[float], None], direction: List[str]) -> pd.DataFrame:
    if weights == None:
        weights = [1 / len(direction)] * len(direction)
        logger.warning(f'Weights value is None. Weights will be defined as {weights}')
    
    return weights
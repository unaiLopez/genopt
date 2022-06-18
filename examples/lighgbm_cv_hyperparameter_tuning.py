
import numpy as np
import pandas as pd


from lightgbm import LGBMRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error

from genetist.environment import Environment #pip install genetist
from genetist.params import Params

#defining a fixed set of params for 5 variables
params = {
    'num_leaves': np.arange(2, 256),
    'max_depth': np.arange(3, 20),
    'learning_rate': [0.0005, 0.001, 0.005, 0.01, 0.05, 0.1],
    'n_estimators': np.arange(50, 300),
    'objective': ['regression', 'regression_l1']
}
#defining a 5 variable search space to fine tune the model
params = {
    'num_leaves': Params.suggest_int(2, 256),
    'max_depth': Params.suggest_int(3, 20),
    'learning_rate': Params.suggest_float(0.0005, 0.1),
    'n_estimators': Params.suggest_int(50, 300),
    'objective': Params.suggest_categorical(['regression', 'regression_l1'])
}

#defining an objective function
def objective(individual):
    df = pd.read_csv('../datasets/california_housing.csv')
    df = pd.get_dummies(df, drop_first=True, dummy_na=True)
    df.dropna(how='any', axis=0, inplace=True)
    
    X = df.drop('median_house_value', axis=1)
    y = df['median_house_value']
    
    num_leaves = individual['num_leaves']
    max_depth = individual['max_depth']
    learning_rate = individual['learning_rate']
    n_estimators = individual['n_estimators']
    objective = individual['objective']
    
    maes = list()
    kf = KFold(n_splits=3)
    for train_indexes, test_indexes in kf.split(X):
        X_train, X_test = X.iloc[train_indexes, :], X.iloc[test_indexes, :]
        y_train, y_test = y.iloc[train_indexes], y.iloc[test_indexes]
        
        model = LGBMRegressor(
            num_leaves=num_leaves,
            max_depth=max_depth,
            learning_rate=learning_rate,
            n_estimators=n_estimators,
            objective=objective,
            n_jobs=2,
            verbosity=-1,
        )

        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        
        mae = mean_absolute_error(y_test, predictions)
        maes.append(mae)
    
    return np.mean(maes)

if __name__ == '__main__':
    #defining our Environment instance with a population of 40 individuals, 15 generation, one-point crossover 
    #and a single gene mutation with a 25% probability of mutation
    environment = Environment(
        params=params,
        num_population=40,
        generations=15,
        crossover_type='one-point',
        mutation_type='single-gene',
        prob_mutation=0.25,
        verbose=1
    )

    #minimizing the objective function using all the available cores
    results = environment.optimize(objective=objective, direction='minimize', n_jobs=-1)

    print()
    print(f'EXECUTION TIME={results.execution_time}')
    print(f'BEST SCORE={results.best_score}')
    print(f'BEST INDIVIDUAL={results.best_individual}')
    print('BEST PER GENERATION:')
    print(results.best_per_generation_dataframe)
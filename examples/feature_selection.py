from tensorflow.keras.datasets import mnist
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

from genetist.environment import Environment #pip install genetist

(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape(X_train.shape[0], X_train.shape[1] * X_train.shape[2])
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1] * X_test.shape[2])
num_columns = X_train.shape[1]

params = dict()
for i in range(num_columns):
    params[i] = [0, 1]

def objective(individual):
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1] * X_train.shape[2])
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1] * X_test.shape[2])
    columns_mask = [item[1] for item in individual.items()]
    X_train = X_train[:, columns_mask]
    X_test = X_test[:, columns_mask]

    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    
    return accuracy_score(y_test, predictions)

if __name__ == '__main__':
    environment = Environment(
        params=params,
        num_population=100,
        generations=25,
        crossover_type='three-point',
        mutation_type='single-gene',
        prob_mutation=0.25,
        verbose=2
    )

    results = environment.optimize(objective=objective, direction='maximize')

    print()
    print(f'EXECUTION TIME={results.execution_time}')
    print(f'BEST SCORE={results.best_score}')
    print(f'BEST INDIVIDUAL={results.best_individual}')
    print('BEST INDIVIDUALS PER GENERATION:')
    print(results.best_per_generation_dataframe)
    print('LAST GENERATION INDIVIDUALS:')
    print(results.last_generation_individuals_dataframe)
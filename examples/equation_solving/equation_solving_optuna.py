import optuna

def objective(trial):
    x = trial.suggest_float('x', -100.0, 100.0)
    y = trial.suggest_float('y', -100.0, 100.0)
    z = trial.suggest_float('z', -100.0, 100.0)
    k = trial.suggest_float('k', -100.0, 100.0)

    return (x**2 - 4*y**3 / z**4) * k**3

if __name__ == '__main__':
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, timeout=60)

    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

import mlflow
import optuna
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split


def objective(trial):
    mlflow.set_experiment("ML-Workout-winequality")
    metrics = {}

    dataset = pd.read_csv("winequality-white.csv", sep=";").astype(float)

    selected_features = [
        "total sulfur dioxide",
        "pH",
        "residual sugar",
        "free sulfur dioxide",
        "volatile acidity",
        "alcohol",
    ]
    target_col = "quality"

    X = dataset[selected_features]
    y = dataset[target_col]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=93)

    with mlflow.start_run():
        params = {
            "n_estimators": trial.suggest_int("n_estimators", low=1, high=666),
            "max_depth": trial.suggest_int("max_depth", low=1, high=16),
            "min_samples_split": trial.suggest_int("min_samples_split", low=2, high=16),
        }
        model = RandomForestRegressor(**params, n_jobs=-1, random_state=93)
        mlflow.log_params(params)

        model.fit(X_train, y_train)

        for features, target, d_name in [(X_train, y_train, "train"), (X_test, y_test, "test")]:
            y_pred = model.predict(features)
            mae = mean_absolute_error(target, y_pred)
            print(f"{d_name}_mae: {mae}")
            mlflow.log_metric(f"{d_name}_mae", mae)
            metrics[f"{d_name}_mae"] = mae

        mlflow.sklearn.log_model(sk_model=model, input_example=X_train, artifact_path="model")

    return metrics["test_mae"]


study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=25, n_jobs=-1)

print(study.best_params)

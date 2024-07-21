import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split


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

model = RandomForestRegressor(n_estimators=2, n_jobs=-1, random_state=93)
model.fit(X_train, y_train)

for features, target, d_name in [(X_train, y_train, "train"), (X_test, y_test, "test")]:
    y_pred = model.predict(features)
    mae = mean_absolute_error(target, y_pred)
    print(f"{d_name}_mae: {mae}")

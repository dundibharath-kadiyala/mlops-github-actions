
import os
import joblib

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score


def train_model():

    # 1. Load dataset
    print("Loading diabetes dataset...")

    data = load_diabetes()

    X = data.data
    y = data.target

    print("Dataset loaded successfully!")
    print("Features:", X.shape)
    print("Target:", y.shape)

    # 2. Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    print("Training samples:", len(X_train))
    print("Testing samples:", len(X_test))

    # 3. Initialize model
    model = RandomForestRegressor(
        n_estimators=100,
        random_state=42
    )

    # 4. Train model
    print("Training model...")

    model.fit(X_train, y_train)

    print("Model training completed!")

    # 5. Evaluate model
    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    print("Model Evaluation:")
    print("MAE:", mae)
    print("R2 Score:", r2)

    # 6. Save model
    os.makedirs("models", exist_ok=True)

    joblib.dump(model, "models/model.joblib")

    print("Model saved successfully!")


if __name__ == "__main__":
    train_model()

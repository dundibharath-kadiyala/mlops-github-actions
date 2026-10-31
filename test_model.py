
import joblib
import numpy as np
import os


MODEL_PATH = "models/model.joblib"


def test_model_exists():
    """Check whether the trained model exists."""

    assert os.path.exists(MODEL_PATH)


def test_model_loading():
    """Check whether the model loads successfully."""

    model = joblib.load(MODEL_PATH)

    assert model is not None


def test_model_prediction():
    """Check whether the model generates predictions."""

    model = joblib.load(MODEL_PATH)

    sample_data = np.array([
        [0.038, 0.050, 0.062, 0.021, 0.043,
         0.015, -0.035, 0.034, 0.066, 0.025]
    ])

    prediction = model.predict(sample_data)

    assert len(prediction) == 1


def test_prediction_is_valid():
    """Check whether prediction is a valid number."""

    model = joblib.load(MODEL_PATH)

    sample_data = np.zeros((1, 10))

    prediction = model.predict(sample_data)

    assert np.isfinite(prediction[0])

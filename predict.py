import joblib
import numpy as np

# Load trained model
model = joblib.load("models/model.joblib")

print("Model loaded successfully!")

# Prepare sample input
sample_data = np.array([
    [0.038, 0.050, 0.062, 0.021, 0.043,
     0.015, -0.035, 0.034, 0.066, 0.025]
])

# Make prediction
prediction = model.predict(sample_data)

print("Predicted diabetes progression score:", prediction[0])
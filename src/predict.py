import joblib
import numpy as np

model = joblib.load("models/model.pkl")

def predict(input_data):
    input_array = np.array(input_data).reshape(1, -1)
    prediction = model.predict(input_array)
    probability = model.predict_proba(input_array)

    return {
        "prediction": int(prediction[0]),
        "probability": float(probability[0][1])
    }

# Example test
if __name__ == "__main__":
    sample = [30, 1500, 500, 200, 1, 6, 0.3, 0.4]
    print(predict(sample))
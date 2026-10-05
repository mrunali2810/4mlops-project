"""Load the trained model and make predictions."""
import sys

import joblib


def predict(features, model_path="models/model.joblib"):
    model = joblib.load(model_path)
    return model.predict([features]).tolist()


if __name__ == "__main__":
    # usage: python src/predict.py 5.1 3.5 1.4 0.2
    values = [float(x) for x in sys.argv[1:]] or [5.1, 3.5, 1.4, 0.2]
    print("Prediction:", predict(values))

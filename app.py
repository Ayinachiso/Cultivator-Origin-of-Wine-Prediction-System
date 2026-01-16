from flask import Flask, render_template, request
import numpy as np
from tensorflow.keras.models import load_model
import joblib
import os

# Initialize Flask app
app = Flask(__name__)

# Load trained Keras model
MODEL_PATH = "model.h5"
model = load_model(MODEL_PATH)

# Load scaler for input preprocessing
SCALER_PATH = "scaler.save"
if os.path.exists(SCALER_PATH):
    scaler = joblib.load(SCALER_PATH)
else:
    scaler = None

@app.route("/")
def home():
    """Render the main page with the input form."""
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    """
    Receive input from the form, preprocess, predict wine cultivator origin, and render result.
    """
    try:
        # Get form data and convert to float
        features = [
            float(request.form["alcohol"]),
            float(request.form["malic_acid"]),
            float(request.form["ash"]),
            float(request.form["alcalinity"]),
            float(request.form["magnesium"])
        ]

        # Prepare input for model
        input_features = np.array(features).reshape(1, -1)

        # Scale input if scaler exists
        if scaler:
            input_features = scaler.transform(input_features)

        # Predict class probabilities
        prediction = model.predict(input_features)
        cultivator = int(np.argmax(prediction))
        confidence = float(np.max(prediction))

        # Map class index to human-readable name (optional)
        cultivator_names = ["Cultivator A", "Cultivator B", "Cultivator C"]
        result = f"Predicted Origin: {cultivator_names[cultivator]} (Confidence: {confidence:.2%})"
    except Exception as e:
        result = f"Error: {str(e)}"

    return render_template("index.html", prediction=result)

# Run the app
if __name__ == "__main__":
    app.run(debug=True)
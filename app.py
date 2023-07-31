# app.py
from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load the trained model
clf = joblib.load('fish_species_classifier_model.joblib')
le = joblib.load('label_encoder.joblib')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    # Convert the input data into a DataFrame
    input_data = pd.DataFrame(data, index=[0])

    # Make predictions
    species_prediction = clf.predict(input_data)

    return jsonify({
        'species_prediction': le.inverse_transform(species_prediction)[0]
    })


if __name__ == '__main__':
    app.run(debug=True)

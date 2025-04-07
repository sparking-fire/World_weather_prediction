from flask import Flask, render_template, request
import joblib
import numpy as np
import os
import json


# Load city-country to lat-lon map
with open(os.path.join('model', 'city_lat_lon_map.json')) as f:
    city_lat_lon_map = json.load(f)

app = Flask(__name__)

# Load models and encoders
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'model')

temp_model = joblib.load(os.path.join(MODEL_DIR, "temp_model.pkl"))
wind_model = joblib.load(os.path.join(MODEL_DIR, "wind_model.pkl"))
weather_model = joblib.load(os.path.join(MODEL_DIR, "weather_model.pkl"))

le_city = joblib.load(os.path.join(MODEL_DIR, "city_encoder.pkl"))
le_country = joblib.load(os.path.join(MODEL_DIR, "country_encoder.pkl"))
le_weather = joblib.load(os.path.join(MODEL_DIR, "weather_encoder.pkl"))

@app.route('/')
def home():
    return render_template('home.html', result=None)

@app.route('/predict', methods=['POST'])
def predict():
    city = request.form.get('City')
    country = request.form.get('Country')

    try:
        # Get lat/lon from the dictionary
        key = f"{country}|{city}"
        if key not in city_lat_lon_map:
            raise ValueError("Coordinates not found for this city-country pair.")

        coords = city_lat_lon_map[key]
        latitude = coords['latitude']
        longitude = coords['longitude']

        # Encode city and country
        city_encoded = le_city.transform([city])[0]
        country_encoded = le_country.transform([country])[0]

        # Prepare features for the models
        features = np.array([[city_encoded, country_encoded, latitude, longitude]])

        # Predictions
        temp = temp_model.predict(features)[0]
        wind = wind_model.predict(features)[0]
        probas = weather_model.predict_proba(features)[0]

        top_3_idx = np.argsort(probas)[-3:][::-1]
        top_3_labels = le_weather.inverse_transform(top_3_idx)
        top_3_probs = probas[top_3_idx]

        weather_description = (
            f"Most likely: {top_3_labels[0]} ({top_3_probs[0] * 100:.1f}%)<br>"
            f"Next: {top_3_labels[1]} ({top_3_probs[1] * 100:.1f}%)<br>"
            f"Then: {top_3_labels[2]} ({top_3_probs[2] * 100:.1f}%)"
        )

        result = {
            'city': city,
            'country': country,
            'temperature': f"{temp:.2f} °C",
            'wind_speed': f"{wind:.2f} km/h",
            'weather_description': weather_description
        }

    except Exception as e:
        result = {'error': f"An error occurred: {str(e)}"}

    return render_template('home.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)

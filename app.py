from flask import Flask, request, jsonify, render_template, request, redirect, url_for, session, flash,send_from_directory
import threading
import webbrowser
import os
import numpy as np
import pandas as pd
import xgboost as xgb
from xgboost import Booster
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import requests
from google.oauth2 import service_account
import google.auth.transport.requests
import re
import joblib
from scipy.spatial import KDTree
import math
from datetime import datetime
import sqlite3
from werkzeug.security import generate_password_hash, check_password_hash
from flask_mail import Mail, Message
import random
from database import init_db, register_user, get_user_by_username
from datetime import timedelta
from datetime import datetime, timedelta
from keras.models import load_model
from joblib import load
import json
import traceback
import tensorflow as tf
import zipfile


def get_access_token():
    credentials = service_account.Credentials.from_service_account_file(
        "models/crowd-explanation-bot-gych-60c7ce1e50ce.json",  
        scopes=["https://www.googleapis.com/auth/cloud-platform"]
    )
    auth_request = google.auth.transport.requests.Request()
    credentials.refresh(auth_request)
    return credentials.token

app = Flask(__name__)

# ------------------ Helper Functions ------------------ #

# Generate station and day mappings dynamically
def generate_mappings(df):
    station_mapping = {}
    day_mapping = {}

    for col in df.columns:
        if col.startswith('Station_'):
            station_name = col.replace('Station_', '')
            station_mapping[station_name.lower()] = col
        elif col.startswith('Day_'):
            day_name = col.replace('Day_', '')
            day_mapping[day_name.lower()] = col

    return station_mapping, day_mapping

# Load the XGBoost model from JSON format
def load_model(filename='models/crowd_model_randomsearch_prob.json'):  
    loaded_model = Booster()
    loaded_model.load_model(filename)
    print(f"Model loaded successfully from '{filename}'.")
    return loaded_model

# Scale the hour feature using MinMaxScaler
def scale_hour(hour):
    scaler = MinMaxScaler(feature_range=(0, 0.5))
    scaler.fit(np.array([0, 23]).reshape(-1, 1))
    scaled_hour = scaler.transform(np.array([[hour]]))[0, 0]
    return scaled_hour

def convert_to_24_hour(hour_str):
    hour = int(re.sub(r'[^0-9]', '', hour_str)) 
    
    if "pm" in hour_str.lower() and hour != 12:
        hour += 12
    elif "am" in hour_str.lower() and hour == 12:
        hour = 0  
    
    return hour

# Shared function used by both standalone and predict crowd level flask route
def process_crowd_level_prediction(station, day, hour):
    try:
        station_name = station.strip().lower()
        day_name = day.strip().lower()

        if station_name not in station_mapping:
            return {"error": f"Invalid station name '{station_name}' provided."}
        if day_name not in day_mapping:
            return {"error": f"Invalid day name '{day_name}' provided."}
        if hour is None or not isinstance(hour, (int, float)) or not (0 <= hour <= 23):
            return {"error": f"Invalid hour '{hour}'. Hour must be between 0 and 23."}

        scaled_hour = scale_hour(hour)

        input_features = {col: 0 for col in feature_names}
        input_features[station_mapping[station_name]] = 1
        input_features[day_mapping[day_name]] = 1  # Keep it consistent (1 or 2)
        input_features["Hour"] = scaled_hour

        aligned_features = [input_features.get(f, 0) for f in feature_names]

        feature_array = np.array([aligned_features], dtype=np.float64)
        dmatrix_input = xgb.DMatrix(feature_array, feature_names=feature_names)

        probabilities = loaded_model.predict(dmatrix_input)
        predicted_class = np.argmax(probabilities[0])
        prediction_label = ['l', 'm', 'h'][predicted_class]

        return {
            "prediction": prediction_label,
            "probabilities": {
                "l": float(round(probabilities[0][0] * 100, 2)),
                "m": float(round(probabilities[0][1] * 100, 2)),
                "h": float(round(probabilities[0][2] * 100, 2))
            }
        }

    except Exception as e:
        return {"error": str(e)}

# ------------------ Load Model and Mappings ------------------ #


       # --------------- Weather Prediction --------------- #
app.secret_key = "weather_prediction_app"

# Initialize database
init_db()

# Flask-Mail Configuration
app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USERNAME'] = 'your-email@gmail.com'
app.config['MAIL_PASSWORD'] = 'your-app-password'
app.config['MAIL_DEFAULT_SENDER'] = 'your-email@gmail.com'

mail = Mail(app)

# Function to send email alerts
def send_email_alert(email, message):
    msg = Message("🌤️ Weather Alert!", recipients=[email])
    msg.body = message
    mail.send(msg)

# Load trained XGBoost models
models = {}
targets = ['temp', 'precip', 'humidity', 'uvindex']
model_dir = os.path.join(os.getcwd(), 'models/weather_datasets')

for target in targets:
    model_path = os.path.join(model_dir, f"xgboost_model_{target}.json")
    if os.path.exists(model_path):
        models[target] = xgb.Booster()
        models[target].load_model(model_path)

    else:
        print(f"⚠️ Warning: Model file not found for {target}: {model_path}")

       # --------------- Carpark Prediction --------------- #

# Load the dataset
zip_path = 'models/carpark_datasets/filtered_processed_carpark_availability.zip'  # Path to your ZIP file
csv_filename = 'filtered_processed_carpark_availability.csv'  # Name of the CSV inside the ZIP

# Open the ZIP file and read the CSV
with zipfile.ZipFile(zip_path, 'r') as z:
    with z.open(csv_filename) as f:
        data_df_carpark = pd.read_csv(f)

# Load the carpark information with X, Y coordinates
carpark_info_df_carpark = pd.read_csv('models/carpark_datasets/filtered_HDB_Carpark_Information.csv')
carpark_info_df_carpark['address_carpark'] = carpark_info_df_carpark['address'].astype(str)

# Merge addresses into `data_df_carpark` to ensure it has an "address_carpark" column
data_df_carpark = data_df_carpark.merge(carpark_info_df_carpark[['car_park_no', 'address_carpark']], 
                                        left_on='carpark_number', right_on='car_park_no', how='left')

# Drop unnecessary columns
data_df_carpark.drop(columns=['car_park_no'], inplace=True)

# Keep only relevant columns for location search
carpark_locations_carpark = carpark_info_df_carpark[['car_park_no', 'x_coord', 'y_coord', 'address_carpark']].dropna()

# Create a KDTree for fast nearest neighbor lookup
carpark_tree_carpark = KDTree(carpark_locations_carpark[['x_coord', 'y_coord']].values)

# Load the trained LightGBM model
model_carpark = joblib.load("models/model_carpark.joblib")

      # --------------- MRT Crowd Prediction --------------- #

# load dataset for reasoning dictionary
station_dictionary = pd.read_csv("models/MRT_crowd_datasets/all_stations_cleaned.csv")

# Load the training dataset to generate mappings
crowdlevel_file = 'models/MRT_crowd_datasets/updated_crowd_data.csv'  
data_df_mrt_crowd = pd.read_csv(crowdlevel_file)
data_df_mrt_crowd = pd.get_dummies(data_df_mrt_crowd, columns=['Station', 'Day'], drop_first=False)

# Generate station and day mappings
station_mapping, day_mapping = generate_mappings(data_df_mrt_crowd)

# Load the XGBoost model
model_filename = 'models/crowd_model_randomsearch_prob.json'  
loaded_model = load_model(model_filename)

# Retrieve feature names from the trained model
feature_names = loaded_model.feature_names

# ------------------ Flask API Endpoints ------------------ #

@app.route('/crowd')
def crowd_model():
    return render_template('Crowd_Prediction.html')  

@app.route('/carpark')
def carpark_model():
    return render_template('index_carpark.html')

@app.route('/weather')
def weather_model():
    return render_template('home_weather.html') 

@app.route('/traffic', methods=['GET'])
def serve_traffic():
    return render_template('index_traffic.html')

# Route: Home Page (Protected)
@app.route("/")
def home():
    return render_template("main_home.html")

app.permanent_session_lifetime = timedelta(minutes=15)

@app.before_request
def check_session_timeout():
    if "logged_in" in session:
        last_activity = session.get("last_activity")
        if last_activity:
            last_activity = datetime.strptime(last_activity, "%Y-%m-%d %H:%M:%S")
            if (datetime.now() - last_activity).total_seconds() > 900:  # 15 minutes
                session.clear()
                flash("⚠️ Your session has expired. Please log in again.", "warning")
                return redirect(url_for("login"))
        # Update last activity timestamp
        session["last_activity"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# ------------------ Flask Server Endpoints ------------------ #

@app.route('/suggest_carpark', methods=['GET'])
def suggest_carpark():
    query_carpark = request.args.get('query_carpark', '').strip().lower()
    
    if not query_carpark:
        return jsonify([])  # Return empty list if input is empty

    # Ensure the dataset has address information
    suggestions_carpark = carpark_info_df_carpark[carpark_info_df_carpark['address_carpark'].str.contains(query_carpark, case=False, na=False)]
    
    # Return only the address_carpark
    return jsonify(suggestions_carpark['address_carpark'].head(10).tolist())


@app.route('/predict_carpark', methods=['POST'])
def predict_carpark():
    data_carpark = request.get_json()
    input_address_carpark = data_carpark.get('address_carpark', '').strip()  # Ensure input is valid
    input_date_carpark = data_carpark.get('date_carpark')
    input_time_carpark = data_carpark.get('time_carpark')

    # Validate the address
    if input_address_carpark not in data_df_carpark['address_carpark'].values:
        return jsonify({'error_carpark': 'Invalid or missing carpark address.'}), 400

    # Parse the date and time
    dt_carpark = datetime.strptime(f"{input_date_carpark} {input_time_carpark}", "%Y-%m-%d %H:%M")
    day_of_week_carpark = dt_carpark.isoweekday()  # Monday = 1, Sunday = 7
    is_weekend_carpark = 1 if day_of_week_carpark in [6, 7] else 0
    hour_carpark = dt_carpark.hour

    # Extract historical availability for lag features
    recent_data_carpark = data_df_carpark[data_df_carpark['address_carpark'] == input_address_carpark].tail(3)
    
    if len(recent_data_carpark) < 3:
        return jsonify({'error_carpark': 'Not enough historical data for this carpark.'}), 400

    lag_1_carpark = recent_data_carpark.iloc[-1]["available_lots"]
    lag_2_carpark = recent_data_carpark.iloc[-2]["available_lots"]
    rolling_avg_3_carpark = recent_data_carpark["available_lots"].mean()

    # Prepare input features (Must match model's feature set)
    features_carpark = pd.DataFrame({
        'hour_carpark': [hour_carpark],
        'day_of_week_carpark': [day_of_week_carpark],
        'is_weekend_carpark': [is_weekend_carpark],
        'lag_1_carpark': [lag_1_carpark],
        'lag_2_carpark': [lag_2_carpark],
        'rolling_avg_3_carpark': [rolling_avg_3_carpark]
    })

    # Predict using the model
    prediction_carpark = model_carpark.predict(features_carpark)
    predicted_lots_carpark = max(0, math.floor(prediction_carpark[0] / 10) * 10)  # Ensure non-negative

    # Find nearby carparks if available lots are low
    nearby_carparks_carpark = []
    if predicted_lots_carpark < 20:
        # Get the coordinates of the requested carpark
        carpark_row_carpark = carpark_locations_carpark[carpark_locations_carpark['address_carpark'] == input_address_carpark]
        
        if not carpark_row_carpark.empty:
            carpark_coords_carpark = carpark_row_carpark[['x_coord', 'y_coord']].values[0]

            # Find up to 10 nearest neighbors (to ensure we find at least 2 with 30+ lots)
            distances_carpark, indices_carpark = carpark_tree_carpark.query(carpark_coords_carpark, k=10)

            # Loop through nearest carparks and find two with at least 30 available lots
            for idx in indices_carpark[1:]:  # Skip the first index (itself)
                nearby_address_carpark = carpark_locations_carpark.iloc[idx]['address_carpark']
                
                # Get latest available lots for this carpark
                available_lots_carpark = data_df_carpark[data_df_carpark['address_carpark'] == nearby_address_carpark].tail(1)['available_lots'].values
                
                if len(available_lots_carpark) > 0 and available_lots_carpark[0] >= 30:
                    nearby_carparks_carpark.append({
                        "address_carpark": nearby_address_carpark,
                        "available_lots_carpark": int(available_lots_carpark[0])
                    })

                # Stop after finding 2 valid carparks
                if len(nearby_carparks_carpark) == 2:
                    break

    return jsonify({
        'prediction_carpark': predicted_lots_carpark,
        'nearby_carparks_carpark': nearby_carparks_carpark  # Only included if lots < 20
    })

@app.route('/predict/crowdlevel', methods=['POST'])
def predict_crowd_level():
    try:
        data = request.json
        station = data.get('station', '').strip().lower()
        day = data.get('day', '').strip().lower()
        hour = data.get('hour', None)

        result = process_crowd_level_prediction(station, day, hour)

        if "error" in result:
            return jsonify(result), 400

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


def get_station_context(station, day, hour, crowd_level=None):
    print(f"Processing: Station={station}, Day={day}, Hour={hour}, Crowd Level={crowd_level}")

    station = station.title().strip()
    day = day.lower().strip()

    # Clean station name (remove 'station' if present)
    station = re.sub(r'\s*station$', '', station, flags=re.IGNORECASE)
    hour = convert_to_24_hour(hour)

    matching_row = station_dictionary[
        (station_dictionary['station_name'].str.lower() == station.lower()) |
        (station_dictionary['Station'].str.lower() == station.lower())
    ]

    if matching_row.empty:
        return f"Sorry, we do not have data for '{station}' at {hour}:00 on {day}."

    station_code = matching_row['Station'].iloc[0]  
    station_name = matching_row['station_name'].iloc[0]  
    reasoning = matching_row['Reasoning'].iloc[0]

    print(f"Mapped Station Name '{station}' -> Station Code '{station_code}'")  

    response_data = process_crowd_level_prediction(station_code, day, hour)

    print("Prediction Function Response:", response_data)  
    if not isinstance(response_data, dict):
        return "Error: Prediction function returned an invalid format."

    if "error" in response_data:
        return f"Error: {response_data['error']}"

    if "prediction" not in response_data or "probabilities" not in response_data:
        return "Error: Missing prediction or probabilities from response."

    probabilities = response_data["probabilities"]
    
    most_probable_crowd_level = max(probabilities, key=probabilities.get)

    prob_response = ""
    crowd_mapping = {"low": "l", "medium": "m", "high": "h"}
    reverse_crowd_mapping = {v: k for k, v in crowd_mapping.items()}  

    if crowd_level and crowd_level.lower() in crowd_mapping:
        crowd_level = crowd_mapping[crowd_level.lower()]
        user_prob = probabilities[crowd_level]
        full_word_crowd_level = reverse_crowd_mapping[crowd_level]
        full_word_probable_crowd_level = reverse_crowd_mapping[most_probable_crowd_level]
        # Determine probability response
        if 0 <= user_prob < 10:
            prob_response = f"The chance of '{full_word_crowd_level}' happening is very low. It is likely not possible."
        elif 10 <= user_prob <= 25:
            prob_response = f"The chance of '{full_word_crowd_level}' happening is low. It can be possible but unlikely."
        elif 26 <= user_prob <= 40:
            prob_response = f"The chance of '{full_word_crowd_level}' happening is moderate. It happens, but not often."
        elif 41 <= user_prob <= 50:
            prob_response = f"The chance of '{full_word_crowd_level}' happening is somewhat likely. It happens occasionally."
        elif 51 <= user_prob <= 65:
            prob_response = f"The chance of '{full_word_crowd_level}' happening is decent. It has a fair chance of happening."
        elif 66 <= user_prob <= 75:
            prob_response = f"The chance of '{full_word_crowd_level}' happening is high. It is quite likely."
        else:
            prob_response = f"The chance of '{full_word_crowd_level}' happening is very high. It usually happens."

    response = f"{prob_response} To elaborate, {station_name} ({station_code.upper()}) Station is {reasoning.lower()}. Hence, the crowd level around this time is usually {full_word_probable_crowd_level}."
    return response

DIALOGFLOW_URL = "https://dialogflow.googleapis.com/v2/projects/crowd-explanation-bot-gych/agent/sessions/asdwdadsgaoiwfhaiwfhiwhwhabsf:detectIntent"
DIALOGFLOW_HEADERS = {
    "Authorization": f"Bearer {get_access_token()}",
    "Content-Type": "application/json"
}

@app.route('/webhook', methods=['POST'])
def webhook():
    req = request.get_json()
    print("Incoming Request:", req)  

    if 'queryInput' in req:
        query_text = req['queryInput']['text']['text']
        print(f"Processing raw input: {query_text}")

        payload = {
            "queryInput": {
                "text": {
                    "text": query_text,
                    "languageCode": "en"
                }
            }
        }

        # Send request to Dialogflow
        dialogflow_response = requests.post(DIALOGFLOW_URL, headers=DIALOGFLOW_HEADERS, json=payload)
        
        if dialogflow_response.status_code == 200:
            dialogflow_data = dialogflow_response.json()
            print("Dialogflow Response:", dialogflow_data)

            # Extract intent and parameters from Dialogflow response
            parameters = dialogflow_data['queryResult']['parameters']

            # Extract required parameters
            station = parameters.get('station')
            day = parameters.get('day')
            hour = parameters.get('time')
            crowd_level = parameters.get('crowd_level', None)

            # Generate response using get_station_context
            response_text = get_station_context(station, day, hour, crowd_level)

            # Send formatted response back to the frontend
            response = {
                "fulfillmentText": response_text
            }
        else:
            print("Error communicating with Dialogflow:", dialogflow_response.text)
            response = {"fulfillmentText": "Error processing the input via Dialogflow."}

        return jsonify(response)

    else:
        return jsonify({"fulfillmentText": "Invalid request structure."}), 400

# Route: Login Page
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        user = get_user_by_username(username)
        if user and check_password_hash(user[4], password):
            session.permanent = True
            session['logged_in'] = True
            session['username'] = user[2]  # Username
            session['email'] = user[3]  # Email

            flash(f"✅ Welcome, {user[1]}!", "success")
            return redirect(url_for("home"))

        flash("⚠️ Invalid username or password!", "danger")
    
    return render_template('login.html')

# Route: Signup Page
@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        fullname = request.form['fullname']
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        confirm_password = request.form['confirm_password']

        if password != confirm_password:
            flash("⚠️ Passwords do not match!", "danger")
            return redirect(url_for('signup'))

        hashed_password = generate_password_hash(password, method='pbkdf2:sha256')

        if register_user(fullname, username, email, hashed_password):
            flash("✅ Sign-up successful! Please log in.", "success")
            return redirect(url_for('login'))
        else:
            flash("⚠️ Username or Email already exists!", "danger")

    return render_template('signup.html')

# Route: Predict Weather and Show Result
@app.route('/predict', methods=['POST'])
def predict():
    if not session.get('logged_in'):
        return redirect(url_for('login'))

    city = request.form.get("city")
    date = request.form.get("date")
    time = request.form.get("time")
    user_email = session.get("email")

    if not city or not date or not time:
        flash("⚠️ Please fill in all fields!", "danger")
        return redirect(url_for('home'))

    # Process date
    date_obj = pd.to_datetime(date)
    prepared_features = {
        'year': date_obj.year,
        'month': date_obj.month,
        'day': date_obj.day,
        'day_of_week': date_obj.weekday(),
        'datetime_ordinal': date_obj.toordinal(),
        'windspeed': 5.0,
        'precipprob': 50.0,
        'precipcover': 0.3,
        'tempmax': 30.0,
        'tempmin': 25.0
    }

    # Convert data to DataFrame
    input_df = pd.DataFrame([prepared_features])
    dmatrix = xgb.DMatrix(input_df)

 
    # Generate predictions and round to 2 decimal places
    predictions = {target: round(models[target].predict(dmatrix)[0], 2) for target in targets}

    # Alert message if severe weather is predicted
    alert_message = ""
    if predictions['temp'] > 35:
        alert_message += "🔥 High Temperature Warning!\n"
    if predictions['precip'] > 50:
        alert_message += "🌧️ Heavy Rain Warning!\n"

    alert = bool(alert_message)
    if alert:
        send_email_alert(user_email, alert_message)

    return render_template('weather_result.html', predictions=predictions, alert=alert, message=alert_message)

# Route: Logout
@app.route('/logout')
def logout():
    session.clear()  # Completely clears the session
    flash("✅ Successfully logged out. Please log in again.", "success")
    return redirect(url_for('login'))

                              # ------------------ Traffic flow ------------------ #
# Check if we should skip XGBoost models
USE_XGBOOST = os.getenv("SKIP_XGBOOST", "False").lower() != "true"

if USE_XGBOOST:
    import xgboost as xgb  # Import XGBoost only if needed
# Function to load models safely
def load_model_safely(model_path):
    try:
        if not os.path.exists(model_path):
            print(f"⚠️ Model file {model_path} not found!")
            return None

        if USE_XGBOOST and model_path.endswith(".json"):  # If it's an XGBoost model
            model = xgb.Booster()
            model.load_model(model_path)
        else:  # ✅ Load TensorFlow model correctly
            model = tf.keras.models.load_model(model_path)

        print(f"✅ Loaded model: {model_path}")
        return model
    except Exception as e:
        print(f"⚠️ Error loading model {model_path}: {e}")
        return None
# Function to load camera data from JSON file
def load_camera_data():
    try:
        with open("models/traffic_model/camera.json", "r") as file:
            camera_data = json.load(file)
        return camera_data
    except Exception as e:
        print(f"Error loading camera data: {e}")
        return []  # Return an empty list if JSON loading fails


# Function to load ML model safely
def load_model_safely(model_path):
    try:
        if not os.path.exists(model_path):
            print(f"⚠️ Model file {model_path} not found!")
            return None

        if USE_XGBOOST and model_path.endswith(".json"):  # If it's an XGBoost model
            model = xgb.Booster()
            model.load_model(model_path)
        else:  # Load TensorFlow model
            model = tf.keras.models.load_model(model_path)

        print(f"✅ Loaded model: {model_path}")
        return model
    except Exception as e:
        print(f"⚠️ Error loading model {model_path}: {e}")
        return None


# Function to predict vehicle count
def predict_future_vehicle_count(camera_id, predicted_time):
    model_path = f"models/traffic_model/models/camera_{camera_id}_model.h5"
    scaler_path = "models/traffic_model/traffic_scaler.joblib"

    # Load Model & Scaler
    model = load_model_safely(model_path)
    if model is None:
        return None

    if not os.path.exists(scaler_path):
        print(f"⚠️ Scaler file {scaler_path} not found!")
        return None

    scaler = load(scaler_path)

    # Extract time features
    try:
        datetime_value = datetime.strptime(predicted_time, "%Y-%m-%dT%H:%M:%S")
        features = pd.DataFrame({
            "month": [datetime_value.month],
            "day": [datetime_value.day],
            "hour": [datetime_value.hour],
            "weekday_sin": [np.sin(2 * np.pi * datetime_value.weekday() / 7)],
            "weekday_cos": [np.cos(2 * np.pi * datetime_value.weekday() / 7)]
        })

        # Normalize and predict
        features = scaler.transform(features)
        prediction = model.predict(features)

        # Unload model from memory
        del model
        return float(prediction[0][0])  # Convert to Python float

    except Exception as e:
        print(f"⚠️ Prediction error: {e}")
        return None


# Route to predict traffic for a single camera
@app.route("/predict_traffic", methods=["POST"])
def predict_traffic():
    try:
        data = request.json
        camera_id = data.get("camera_id")
        predicted_time = data.get("predicted_time")

        if not camera_id or not predicted_time:
            return jsonify({"error": "Missing camera_id or predicted_time"}), 400

        camera_data = load_camera_data()
        camera_info = next((c for c in camera_data if c["camera_id"] == camera_id), None)

        if not camera_info:
            return jsonify({"error": "Camera not found"}), 404

        predicted_vehicles = predict_future_vehicle_count(camera_id, predicted_time)

        if predicted_vehicles is None:
            return jsonify({"error": "Prediction failed"}), 500

        return jsonify({
            "camera_id": camera_id,
            "predicted_time": predicted_time,
            "predicted_vehicles": predicted_vehicles,
            "lat": camera_info["lat"],
            "lng": camera_info["lng"]
        }), 200

    except Exception as e:
        print(f"⚠️ Error in predict_traffic: {e}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# Route to predict traffic for all cameras
@app.route("/getAllCameraPredict", methods=["GET"])
def get_cameraList():
    try:
        predicted_time = request.args.get("predicted_time", default=None)
        camera_data = load_camera_data()

        for camera in camera_data:
            camera_id = camera["camera_id"]
            predicted_vehicles = predict_future_vehicle_count(camera_id, predicted_time)

            if predicted_vehicles is not None:
                camera.update({"predicted_vehicles": predicted_vehicles})

        return jsonify(camera_data), 200

    except Exception as e:
        print("⚠️ Error in getAllCameraPredict:", str(e))
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

# ------------------ Auto Open Browser ------------------ #
def open_browser():
    webbrowser.open_new("http://127.0.0.1:5000/")

if __name__ == '__main__':
    if os.environ.get("WERKZEUG_RUN_MAIN") == "true":  # Prevent reloader from opening a browser
        threading.Timer(1.0, open_browser).start()
    app.run(debug=True)

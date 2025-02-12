from flask import Flask, request, render_template, jsonify
import joblib
import pandas as pd
from datetime import datetime
import math
from scipy.spatial import KDTree  # Fast nearest neighbor search

app = Flask(__name__)

# Load the dataset
data_df_carpark = pd.read_csv('filtered_processed_carpark_availability.csv')

# Load the carpark information with X, Y coordinates
carpark_info_df_carpark = pd.read_csv('filtered_HDB_Carpark_Information.csv')
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
model_carpark = joblib.load("model_carpark.joblib")

@app.route("/")
def home():
    return render_template("index_carpark.html")

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

if __name__ == "__main__":
    app.run(debug=True)

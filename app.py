from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import pandas as pd
import os

app = Flask(__name__)

# Load the trained model and preprocessing objects
MODEL_PATH = 'house_model.pkl'

try:
    with open(MODEL_PATH, 'rb') as f:
        model_data = pickle.load(f)
    model = model_data['model']
    scaler = model_data['scaler']
    ohe = model_data['ohe']
    poly = model_data['poly']
    top_locations = model_data['top_locations']
    print("✓ Model loaded successfully!")
except FileNotFoundError:
    print(f"ERROR: {MODEL_PATH} not found. Please run the notebook first.")
    model = None

def predict_house_price(location, total_sqft, bhk, bath, balcony=0):
    """Predict house price based on input features"""
    if model is None:
        return {"error": "Model not loaded"}
    
    try:
        # Create input dataframe
        input_df = pd.DataFrame({
            'location': [location],
            'total_sqft': [float(total_sqft)],
            'bhk': [int(bhk)],
            'bath': [int(bath)],          
            'has_balcony': [1 if int(balcony) > 0 else 0]
        })
        
        # Create engineered features
        input_df['sqft_per_bhk'] = input_df['total_sqft'] / input_df['bhk']
        input_df['bath_per_bhk'] = input_df['bath'] / input_df['bhk']

        # Replace unknown locations with 'other'
        input_df['location'] = input_df['location'].apply(
            lambda x: x if x in top_locations else 'other'
        )
        
        # Apply one-hot encoding to location
        location_encoded = ohe.transform(input_df[['location']])
        
        # Apply polynomial features to numeric columns
        num_features = ['total_sqft', 'bhk', 'bath', 'has_balcony', 'sqft_per_bhk', 'bath_per_bhk']
        input_num = poly.transform(input_df[num_features])
        
        # Scale numeric features
        input_scaled = scaler.transform(input_num)
        
        # Combine with location encoding
        input_scaled = np.hstack([input_scaled, location_encoded])
        
        # Make prediction
        price = model.predict(input_scaled)[0]
        return float(price)
    except Exception as e:
        return {"error": str(e)}

@app.route('/')
def index():
    """Serve the main HTML page"""
    return render_template('index.html', locations=list(top_locations))

@app.route('/api/predict', methods=['POST'])
def predict():
    """API endpoint for price prediction"""
    try:
        data = request.json
        
        # Validate input
        required_fields = ['location', 'total_sqft', 'bhk', 'bath']
        for field in required_fields:
            if field not in data:
                return jsonify({"error": f"Missing field: {field}"}), 400
        
        location = data['location']
        total_sqft = float(data['total_sqft'])
        bhk = int(data['bhk'])
        bath = int(data['bath'])
        balcony = int(data.get('balcony', 0))
        
        # Validate ranges
        if total_sqft < 300 or total_sqft > 30000:
            return jsonify({"error": "Total sqft should be between 300 and 30000"}), 400
        if bhk < 1 or bhk > 20:
            return jsonify({"error": "BHK should be between 1 and 20"}), 400
        if bath < 1 or bath > 15:
            return jsonify({"error": "Bathrooms should be between 1 and 15"}), 400
        
        # Get prediction
        price = predict_house_price(location, total_sqft, bhk, bath, balcony)
        
        if isinstance(price, dict) and "error" in price:
            return jsonify(price), 400
        
        return jsonify({
            "success": True,
            "predicted_price": round(price, 2),
            "currency": "₹ Lakh",
            "input": {
                "location": location,
                "total_sqft": total_sqft,
                "bhk": bhk,
                "bath": bath,
                "balcony": balcony > 0
            }
        })
    except ValueError as e:
        return jsonify({"error": f"Invalid input: {str(e)}"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/locations', methods=['GET'])
def get_locations():
    """Get list of all available locations"""
    return jsonify({"locations": sorted(list(top_locations))})

if __name__ == '__main__':
    print("=" * 50)
    print("House Price Prediction API")
    print("=" * 50)
    print("Starting Flask server...")
    print("Visit: http://localhost:5000")
    print("=" * 50)
    app.run(debug=True, port=5000)

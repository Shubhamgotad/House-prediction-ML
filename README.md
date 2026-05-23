# 🏠 House Price Prediction Application

A full-stack web application for predicting house prices in Bengaluru using Machine Learning.

## 📋 Features

- **Interactive UI**: User-friendly web interface to input house details
- **ML-Powered Predictions**: Uses trained Linear Regression model with polynomial features
- **Real-time Results**: Get price predictions instantly
- **Input Validation**: Validates all user inputs with helpful error messages
- **Location Support**: 100+ locations in Bengaluru

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- Flask
- pandas, numpy, scikit-learn

### Installation

1. **Install dependencies**:
```bash
pip install -r requirements.txt
```

2. **Run the notebook first** (if model file doesn't exist):
   - Open `Notebook_fixed.ipynb` in Jupyter
   - Execute all cells to train the model
   - This will generate `house_model.pkl` in the same directory

3. **Start the Flask server**:
```bash
python app.py
```

4. **Open in browser**:
   - Visit: `http://localhost:5000`
   - Enter house details
   - Click "Predict Price"

## 📁 Project Structure

```
House prediction ML/
├── app.py                    # Flask backend API
├── Notebook_fixed.ipynb      # ML model training & analysis
├── Bengaluru_House_Data.csv  # Training data
├── requirements.txt          # Python dependencies
├── house_model.pkl           # Trained model (generated)
└── templates/
    └── index.html            # Frontend UI
```

## 🔧 How to Use

### Using the Web Interface

1. **Select Location**: Choose from 100+ available locations in Bengaluru
2. **Enter BHK**: Number of bedrooms (1-20)
3. **Enter Area**: Total area in sqft (300-30000)
4. **Enter Bathrooms**: Number of bathrooms (1-15)
5. **Check Balcony**: Select if the property has a balcony
6. **Click Predict**: Get the predicted price in Lakhs (₹)

### Using the API

**Endpoint**: `POST /api/predict`

**Request**:
```json
{
  "location": "Whitefield",
  "total_sqft": 1500,
  "bhk": 3,
  "bath": 2,
  "balcony": 1
}
```

**Response**:
```json
{
  "success": true,
  "predicted_price": 67.50,
  "currency": "₹ Lakh",
  "input": {
    "location": "Whitefield",
    "total_sqft": 1500,
    "bhk": 3,
    "bath": 2,
    "balcony": true
  }
}
```

### Get Available Locations

**Endpoint**: `GET /api/locations`

**Response**:
```json
{
  "locations": ["Whitefield", "Koramangala", "Electronic City", ...]
}
```

## 🤖 Model Details

- **Algorithm**: Linear Regression with Polynomial Features (degree 3)
- **Training Data**: Bengaluru house prices from CSV
- **Features Used**:
  - Location (one-hot encoded)
  - Total Sqft
  - BHK (bedrooms)
  - Bathrooms
  - Balcony presence
  - Engineered features (sqft_per_bhk, bath_per_bhk)
  
- **Preprocessing**:
  - StandardScaler for numeric features
  - OneHotEncoder for location
  - PolynomialFeatures for non-linear relationships
  - Outlier removal & data cleaning

## 📊 Model Performance

Check `Notebook_fixed.ipynb` for detailed metrics:
- RMSE (Root Mean Square Error)
- R² Score
- Residual analysis
- Best-fit visualizations

## 🐛 Troubleshooting

### "Model not loaded" error
- Ensure `house_model.pkl` exists
- Run the notebook cells to generate the model file

### Port already in use
- Change port in `app.py`: `app.run(debug=True, port=5001)`

### Import errors
- Reinstall dependencies: `pip install --upgrade -r requirements.txt`

## 📝 Notes

- Price predictions are in **₹ Lakh** (100,000 rupees)
- Unknown locations are treated as 'other'
- The model uses training data patterns - predictions may vary from actual market prices
- For best results, use realistic input values within the trained range

## 🔄 Workflow

```
1. Train Model (Notebook)
   ↓
2. Export Model (house_model.pkl)
   ↓
3. Run Flask App
   ↓
4. Access Web UI (http://localhost:5000)
   ↓
5. Make Predictions
```

## 📄 License

This project is for educational purposes.

---

**Happy Predicting! 🏡✨**

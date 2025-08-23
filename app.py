from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import json
import random
from datetime import datetime, timedelta

app = Flask(__name__)

# Generate synthetic crime data for demonstration
def generate_crime_data():
    np.random.seed(42)
    
    # Years from 2018 to 2024
    years = list(range(2018, 2025))
    
    # Districts in both states
    ap_districts = ['Visakhapatnam', 'Vijayawada', 'Guntur', 'Nellore', 'Kurnool', 
                   'Rajahmundry', 'Tirupati', 'Anantapur', 'Kadapa', 'Chittoor']
    
    ts_districts = ['Hyderabad', 'Warangal', 'Nizamabad', 'Khammam', 'Karimnagar',
                   'Mahbubnagar', 'Nalgonda', 'Adilabad', 'Medak', 'Rangareddy']
    
    crime_types = ['Theft', 'Burglary', 'Assault', 'Fraud', 'Drug Related', 'Cyber Crime']
    
    data = []
    
    for state, districts in [('Andhra Pradesh', ap_districts), ('Telangana', ts_districts)]:
        for district in districts:
            for year in years:
                # Base crime rate varies by district (urban areas higher)
                base_rate = random.uniform(50, 200) if district in ['Hyderabad', 'Visakhapatnam', 'Vijayawada'] else random.uniform(20, 100)
                
                # Add yearly trend (slight increase over time)
                yearly_factor = 1 + (year - 2018) * 0.02
                
                # Add some randomness
                crime_rate = base_rate * yearly_factor * random.uniform(0.8, 1.2)
                
                # Population factor
                population = random.randint(500000, 5000000)
                
                # Economic indicator (GDP per capita proxy)
                gdp_per_capita = random.randint(80000, 300000)
                
                # Unemployment rate
                unemployment = random.uniform(2, 12)
                
                # Literacy rate
                literacy = random.uniform(60, 95)
                
                data.append({
                    'state': state,
                    'district': district,
                    'year': year,
                    'crime_rate': round(crime_rate, 2),
                    'population': population,
                    'gdp_per_capita': gdp_per_capita,
                    'unemployment_rate': round(unemployment, 2),
                    'literacy_rate': round(literacy, 2)
                })
    
    return pd.DataFrame(data)

# Load and prepare data
df = generate_crime_data()

# Prepare features for ML
def prepare_features(data):
    # Create dummy variables for categorical data
    state_dummies = pd.get_dummies(data['state'], prefix='state')
    district_dummies = pd.get_dummies(data['district'], prefix='district')
    
    # Combine features
    features = pd.concat([
        data[['year', 'population', 'gdp_per_capita', 'unemployment_rate', 'literacy_rate']],
        state_dummies,
        district_dummies
    ], axis=1)
    
    return features

# Train models
X = prepare_features(df)
y = df['crime_rate']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Random Forest Model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Linear Regression Model
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/crime_data')
def get_crime_data():
    # Group data by state and year for visualization
    grouped = df.groupby(['state', 'year'])['crime_rate'].mean().reset_index()
    
    ap_data = grouped[grouped['state'] == 'Andhra Pradesh']
    ts_data = grouped[grouped['state'] == 'Telangana']
    
    return jsonify({
        'andhra_pradesh': {
            'years': ap_data['year'].tolist(),
            'crime_rates': ap_data['crime_rate'].round(2).tolist()
        },
        'telangana': {
            'years': ts_data['year'].tolist(),
            'crime_rates': ts_data['crime_rate'].round(2).tolist()
        }
    })

@app.route('/api/district_data/<state>')
def get_district_data(state):
    state_data = df[df['state'] == state]
    latest_year = state_data['year'].max()
    latest_data = state_data[state_data['year'] == latest_year]
    
    return jsonify({
        'districts': latest_data['district'].tolist(),
        'crime_rates': latest_data['crime_rate'].round(2).tolist()
    })

@app.route('/api/predict', methods=['POST'])
def predict_crime():
    data = request.json
    
    # Create a sample for prediction
    sample_data = pd.DataFrame([{
        'year': data['year'],
        'population': data['population'],
        'gdp_per_capita': data['gdp_per_capita'],
        'unemployment_rate': data['unemployment_rate'],
        'literacy_rate': data['literacy_rate'],
        'state': data['state'],
        'district': data['district']
    }])
    
    # Prepare features
    sample_features = prepare_features(sample_data)
    
    # Ensure all columns are present
    for col in X.columns:
        if col not in sample_features.columns:
            sample_features[col] = 0
    
    sample_features = sample_features[X.columns]
    
    # Make predictions
    rf_prediction = rf_model.predict(sample_features)[0]
    lr_prediction = lr_model.predict(sample_features)[0]
    
    return jsonify({
        'random_forest': round(rf_prediction, 2),
        'linear_regression': round(lr_prediction, 2),
        'average': round((rf_prediction + lr_prediction) / 2, 2)
    })

@app.route('/api/model_performance')
def get_model_performance():
    # Get predictions for test set
    rf_pred = rf_model.predict(X_test)
    lr_pred = lr_model.predict(X_test)
    
    # Calculate metrics
    rf_r2 = r2_score(y_test, rf_pred)
    lr_r2 = r2_score(y_test, lr_pred)
    rf_mse = mean_squared_error(y_test, rf_pred)
    lr_mse = mean_squared_error(y_test, lr_pred)
    
    return jsonify({
        'random_forest': {
            'r2_score': round(rf_r2, 3),
            'mse': round(rf_mse, 3)
        },
        'linear_regression': {
            'r2_score': round(lr_r2, 3),
            'mse': round(lr_mse, 3)
        }
    })

if __name__ == '__main__':
    app.run(debug=True)
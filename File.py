import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from flask import Flask, render_template, request, jsonify
import joblib
import os

# Data Generation and Collection Module
class DataCollector:
    def __init__(self, file_path='crop_data.csv'):
        self.file_path = file_path
    
    def generate_data(self):
        np.random.seed(42)
        num_samples = 1000
        start_date = datetime(2010, 1, 1)
        
        dates = [start_date + timedelta(days=i) for i in range(num_samples)]
        temperature = np.random.normal(25, 5, num_samples)
        rainfall = np.random.normal(1000, 300, num_samples)
        soil_quality = np.random.uniform(0.3, 1, num_samples)
        
        base_yield = 5000
        temp_effect = (temperature - 25) * 50
        rain_effect = (rainfall - 1000) * 2
        soil_effect = (soil_quality - 0.65) * 5000
        
        crop_yield = base_yield + temp_effect + rain_effect + soil_effect
        crop_yield = np.maximum(crop_yield, 0)
        crop_yield += np.random.normal(0, 500, num_samples)
        
        df = pd.DataFrame({
            'date': dates,
            'temperature': temperature,
            'rainfall': rainfall,
            'soil_quality': soil_quality,
            'crop_yield': crop_yield
        })
        
        df.to_csv(self.file_path, index=False)
        print(f"Generated sample data and saved to {self.file_path}")
    
    def load_data(self):
        if not os.path.exists(self.file_path):
            print(f"Data file {self.file_path} not found. Generating sample data...")
            self.generate_data()
        return pd.read_csv(self.file_path)
    
    def preprocess_data(self, data):
        data = data.dropna()
        data['date'] = pd.to_datetime(data['date'])
        data['month'] = data['date'].dt.month
        data['year'] = data['date'].dt.year
        return data

# Machine Learning Module
class CropYieldPredictor:
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
    
    def prepare_features(self, data):
        features = ['temperature', 'rainfall', 'soil_quality', 'month', 'year']
        X = data[features]
        y = data['crop_yield']
        return X, y
    
    def train(self, X, y):
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
    
    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def evaluate(self, X, y):
        predictions = self.predict(X)
        mse = mean_squared_error(y, predictions)
        r2 = r2_score(y, predictions)
        return mse, r2

# Optimization Module
class CropOptimizer:
    def __init__(self, model, scaler):
        self.model = model
        self.scaler = scaler
    
    def optimize_planting(self, temperature, rainfall, soil_quality, month, year):
        input_data = np.array([[temperature, rainfall, soil_quality, month, year]])
        input_scaled = self.scaler.transform(input_data)
        prediction = self.model.predict(input_scaled)
        return prediction[0]
    
    def suggest_improvements(self, current_conditions):
        suggestions = []
        if current_conditions['soil_quality'] < 0.6:
            suggestions.append("Improve soil quality through organic matter addition and proper fertilization.")
        if current_conditions['rainfall'] < 800:
            suggestions.append("Consider implementing irrigation systems to supplement rainfall.")
        return suggestions

# Visualization Module
class DataVisualizer:
    @staticmethod
    def plot_feature_importance(model, feature_names):
        importance = model.feature_importances_
        indices = np.argsort(importance)[::-1]
        
        plt.figure(figsize=(10, 6))
        plt.title("Feature Importances")
        plt.bar(range(len(importance)), importance[indices])
        plt.xticks(range(len(importance)), [feature_names[i] for i in indices], rotation=45)
        plt.tight_layout()
        plt.savefig('feature_importance.png')
    
    @staticmethod
    def plot_predictions(actual, predicted):
        plt.figure(figsize=(10, 6))
        plt.scatter(actual, predicted, alpha=0.5)
        plt.plot([actual.min(), actual.max()], [actual.min(), actual.max()], 'r--', lw=2)
        plt.xlabel('Actual Yield')
        plt.ylabel('Predicted Yield')
        plt.title('Actual vs Predicted Crop Yield')
        plt.tight_layout()
        plt.savefig('yield_prediction.png')

# Main Application
def main():
    # Initialize modules
    data_collector = DataCollector()
    predictor = CropYieldPredictor()
    visualizer = DataVisualizer()
    
    # Load and preprocess data
    data = data_collector.load_data()
    processed_data = data_collector.preprocess_data(data)
    
    # Prepare features and target
    X, y = predictor.prepare_features(processed_data)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    predictor.train(X_train, y_train)
    
    # Evaluate model
    mse, r2 = predictor.evaluate(X_test, y_test)
    print(f"Model Performance - MSE: {mse:.2f}, R2: {r2:.2f}")
    
    # Visualize results
    visualizer.plot_feature_importance(predictor.model, X.columns)
    visualizer.plot_predictions(y_test, predictor.predict(X_test))
    
    # Save model for later use
    joblib.dump(predictor.model, 'crop_yield_model.joblib')
    joblib.dump(predictor.scaler, 'scaler.joblib')

    print("Model and visualizations saved. Starting web application...")

    # Initialize Flask app
    app = Flask(__name__)

    # Load saved model and scaler
    loaded_model = joblib.load('crop_yield_model.joblib')
    loaded_scaler = joblib.load('scaler.joblib')
    optimizer = CropOptimizer(loaded_model, loaded_scaler)

    @app.route('/')
    def home():
        return render_template('index.html')

    @app.route('/predict', methods=['POST'])
    def predict():
        data = request.json
        prediction = optimizer.optimize_planting(
            data['temperature'],
            data['rainfall'],
            data['soil_quality'],
            data['month'],
            data['year']
        )
        suggestions = optimizer.suggest_improvements(data)
        return jsonify({
            'predicted_yield': prediction,
            'suggestions': suggestions
        })

    app.run(debug=True)

if __name__ == "__main__":
    main()
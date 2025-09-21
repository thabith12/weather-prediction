from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
from datetime import datetime
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

# Global variables
df = None
model = None
label_encoder = None

def load_and_clean_data():
    global df
    try:
        # Load the dataset
        df = pd.read_csv('1000 datasets.csv')
        
        # Data Cleaning
        # Convert date column
        df['Formatted Date'] = pd.to_datetime(df['Formatted Date'], errors='coerce')
        
        # Handle missing values
        df['Precip Type'].fillna('None', inplace=True)
        numeric_cols = ['Temperature (C)', 'Apparent Temperature (C)', 'Humidity', 
                       'Wind Speed (km/h)', 'Wind Bearing (degrees)', 'Visibility (km)', 
                       'Pressure (millibars)']
        
        for col in numeric_cols:
            df[col].fillna(df[col].median(), inplace=True)
        
        # Remove constant columns
        df.drop(columns=['Loud Cover'], inplace=True, errors='ignore')
        
        # Feature engineering
        df['Hour'] = df['Formatted Date'].dt.hour
        df['Day'] = df['Formatted Date'].dt.day
        df['Month'] = df['Formatted Date'].dt.month
        df['Year'] = df['Formatted Date'].dt.year
        
        return True
    except Exception as e:
        print(f"Error loading data: {e}")
        return False

def create_visualizations():
    visualizations = {}
    
    # Temperature distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(df['Temperature (C)'], kde=True)
    plt.title('Temperature Distribution')
    plt.xlabel('Temperature (°C)')
    plt.ylabel('Frequency')
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    visualizations['temp_dist'] = base64.b64encode(img.getvalue()).decode()
    plt.close()
    
    # Humidity vs Temperature
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='Temperature (C)', y='Humidity', alpha=0.6)
    plt.title('Temperature vs Humidity')
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    visualizations['temp_humidity'] = base64.b64encode(img.getvalue()).decode()
    plt.close()
    
    # Weather summary count
    plt.figure(figsize=(12, 6))
    df['Summary'].value_counts().head(10).plot(kind='bar')
    plt.title('Top 10 Weather Conditions')
    plt.xticks(rotation=45)
    plt.tight_layout()
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    visualizations['weather_count'] = base64.b64encode(img.getvalue()).decode()
    plt.close()
    
    # Correlation heatmap
    plt.figure(figsize=(12, 8))
    numeric_df = df.select_dtypes(include=[np.number])
    correlation_matrix = numeric_df.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
    plt.title('Feature Correlation Heatmap')
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    visualizations['correlation'] = base64.b64encode(img.getvalue()).decode()
    plt.close()
    
    return visualizations

def train_model():
    global model, label_encoder
    try:
        # Prepare data for modeling
        features = ['Humidity', 'Wind Speed (km/h)', 'Wind Bearing (degrees)', 
                   'Visibility (km)', 'Pressure (millibars)', 'Hour', 'Day', 'Month']
        
        X = df[features].copy()
        
        # Encode the target variable
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(df['Summary'])
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Train Random Forest classifier
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        
        # Get class names
        class_names = label_encoder.classes_
        
        return {
            'accuracy': accuracy,
            'class_names': list(class_names),
            'feature_importance': dict(zip(X.columns, model.feature_importances_))
        }
    except Exception as e:
        print(f"Error training model: {e}")
        return None

@app.route('/')
def index():
    if df is None:
        success = load_and_clean_data()
        if not success:
            return "Error loading data. Please check your dataset."
    
    # Basic statistics
    stats = {
        'total_records': len(df),
        'date_range': f"{df['Formatted Date'].min().date()} to {df['Formatted Date'].max().date()}",
        'avg_temp': df['Temperature (C)'].mean(),
        'avg_humidity': df['Humidity'].mean()
    }
    
    # Create visualizations
    visualizations = create_visualizations()
    
    # Train model and get metrics
    model_metrics = train_model()
    
    return render_template('index.html', 
                         stats=stats, 
                         visualizations=visualizations,
                         model_metrics=model_metrics)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        features = [
            float(data['humidity']),
            float(data['wind_speed']),
            float(data['wind_bearing']),
            float(data['visibility']),
            float(data['pressure']),
            int(data['hour']),
            int(data['day']),
            int(data['month'])
        ]
        
        prediction = model.predict([features])[0]
        predicted_class = label_encoder.inverse_transform([prediction])[0]
        
        return jsonify({'prediction': predicted_class})
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/data_preview')
def data_preview():
    preview = df.head(10).to_dict('records')
    columns = list(df.columns)
    return jsonify({'preview': preview, 'columns': columns})

if __name__ == '__main__':
    load_and_clean_data()
    app.run(debug=True)
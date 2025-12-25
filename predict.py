import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from datetime import datetime
from data_pipeline import DataPipeline
from mongo_utils import mongodb_manager

def load_model_and_preprocessor():
    """Load the trained model and preprocessor"""
    model = joblib.load('models/house_price_model.joblib')
    pipeline = DataPipeline()
    pipeline.load_preprocessor('models/preprocessor.joblib')
    return model, pipeline

def predict_house_price(area, bedrooms, bathrooms, stories, parking, neighborhood, house_style):
    """
    Predict house price using the trained model
    
    Args:
        area (int): Area in square feet
        bedrooms (int): Number of bedrooms
        bathrooms (float): Number of bathrooms
        stories (int): Number of stories
        parking (int): Number of parking spaces
        neighborhood (str): Neighborhood code (A, B, C, or D)
        house_style (str): Type of house (Apartment, Villa, or Townhouse)
        
    Returns:
        dict: Prediction results with confidence interval
    """
    model, pipeline = load_model_and_preprocessor()
    
    # Create input DataFrame
    input_data = {
        'area': [area],
        'bedrooms': [bedrooms],
        'bathrooms': [bathrooms],
        'stories': [stories],
        'parking': [parking],
        'neighborhood': [neighborhood],
        'house_style': [house_style]
    }
    input_df = pd.DataFrame(input_data)
    
    # Preprocess input
    X = pipeline.preprocessor.transform(input_df)
    
    # Make prediction
    prediction = model.predict(X)[0]
    
    # Calculate confidence interval (using standard deviation of predictions from training)
    train_preds = model.predict(pipeline.preprocessor.transform(load_sample_data().drop(columns=['price'])))
    std_dev = np.std(train_preds)
    lower_bound = prediction - (1.96 * std_dev)
    upper_bound = prediction + (1.96 * std_dev)
    
    # Prepare result
    result = {
        'predicted_price': round(prediction, 2),
        'confidence_interval': {
            'lower': round(lower_bound, 2),
            'upper': round(upper_bound, 2)
        },
        'confidence_level': 0.95
    }
    
    # Save to MongoDB
    try:
        input_data = {
            'area': area,
            'bedrooms': bedrooms,
            'bathrooms': bathrooms,
            'stories': stories,
            'parking': parking,
            'neighborhood': neighborhood,
            'house_style': house_style,
            'timestamp': datetime.utcnow()
        }
        mongodb_manager.save_prediction(input_data, result)
    except Exception as e:
        print(f"Warning: Could not save prediction to MongoDB: {e}")
    
    return result

def create_prediction_vs_actual_plot():
    """Create a plot comparing predicted vs actual prices"""
    # Load the model and preprocessor
    model, pipeline = load_model_and_preprocessor()
    
    # Load sample data (or your test data)
    df = load_sample_data()
    X = df.drop(columns=['price'])
    y = df['price']
    
    # Preprocess and predict
    X_processed = pipeline.preprocessor.transform(X)
    y_pred = model.predict(X_processed)
    
    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.scatter(y, y_pred, alpha=0.5)
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
    plt.xlabel('Actual Price')
    plt.ylabel('Predicted Price')
    plt.title('Actual vs Predicted House Prices')
    plt.grid(True)
    
    # Save the plot
    os.makedirs('reports/figures', exist_ok=True)
    plt.savefig('reports/figures/prediction_vs_actual.png')
    plt.close()

def load_sample_data():
    """Generate sample housing data for demonstration"""
    np.random.seed(42)
    n_samples = 1000
    
    data = {
        'area': np.random.normal(1500, 500, n_samples).astype(int),
        'bedrooms': np.random.randint(1, 6, n_samples),
        'bathrooms': np.random.randint(1, 4, n_samples),
        'stories': np.random.randint(1, 4, n_samples),
        'parking': np.random.randint(0, 3, n_samples),
        'neighborhood': np.random.choice(['A', 'B', 'C', 'D'], n_samples),
        'house_style': np.random.choice(['Apartment', 'Villa', 'Townhouse'], n_samples),
    }
    
    # Create a realistic price based on features with some noise
    price = (
        data['area'] * 100 + 
        data['bedrooms'] * 50000 + 
        data['bathrooms'] * 30000 +
        (np.array(data['neighborhood']) == 'A') * 100000 +
        (np.array(data['house_style']) == 'Villa') * 150000 +
        np.random.normal(0, 50000, n_samples)
    )
    
    data['price'] = price.astype(int)
    return pd.DataFrame(data)

if __name__ == "__main__":
    # Example prediction
    prediction = predict_house_price(
        area=2000,
        bedrooms=3,
        bathrooms=2.5,
        stories=2,
        parking=1,
        neighborhood='B',
        house_style='Apartment'
    )
    
    print("\nPrediction Result:")
    print(f"The prediction result is generated directly from the trained ML model stored in a serialized .joblib file.")
    print(f"Predicted Price: ${prediction['predicted_price']:,.2f}")
    print(f"95% Confidence Interval: ${prediction['confidence_interval']['lower']:,.2f} - ${prediction['confidence_interval']['upper']:,.2f}")
    
    # Generate the prediction vs actual plot
    create_prediction_vs_actual_plot()
    print("\nGenerated visualization: reports/figures/prediction_vs_actual.png")

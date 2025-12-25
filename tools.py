import joblib
import numpy as np
import pandas as pd
from data_pipeline import DataPipeline

def predict_house_price(area, bedrooms, bathrooms, stories, parking, neighborhood, house_style):
    """
    Predict house price based on input features.
    
    Args:
        area (int): Area of the house in square feet
        bedrooms (int): Number of bedrooms
        bathrooms (float): Number of bathrooms
        stories (int): Number of stories
        parking (int): Number of parking spaces
        neighborhood (str): Neighborhood code (A, B, C, or D)
        house_style (str): Type of house (Apartment, Villa, or Townhouse)
        
    Returns:
        dict: Predicted price and confidence interval
    """
    try:
        # Load the model and preprocessor
        model = joblib.load('models/house_price_model.joblib')
        pipeline = DataPipeline()
        pipeline.load_preprocessor('models/preprocessor.joblib')
        
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
        
        # Calculate confidence interval (simplified example)
        confidence = 0.95
        std_dev = prediction * 0.1  # 10% of prediction as std dev
        lower_bound = prediction - (1.96 * std_dev)
        upper_bound = prediction + (1.96 * std_dev)
        
        return {
            'predicted_price': round(prediction, 2),
            'confidence_interval': {
                'lower': round(lower_bound, 2),
                'upper': round(upper_bound, 2)
            },
            'confidence_level': confidence
        }
        
    except Exception as e:
        return {'error': str(e)}

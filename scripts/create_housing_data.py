import pandas as pd
import os
import numpy as np
from sklearn.datasets import fetch_california_housing

def create_housing_data():
    try:
        # Create data directory if it doesn't exist
        os.makedirs('data', exist_ok=True)
        
        # Try loading with as_frame=True (for newer scikit-learn versions)
        try:
            california = fetch_california_housing(as_frame=True)
            df = california.frame  # This gives us the complete DataFrame
            if 'MedHouseVal' not in df.columns:
                df['MedHouseVal'] = california.target
        
        # Fall back to manual DataFrame creation if as_frame is not supported
        except (TypeError, AttributeError):
            # Get data and feature names
            X, y = fetch_california_housing(return_X_y=True)
            feature_names = [
                'MedInc', 'HouseAge', 'AveRooms', 'AveBedrms',
                'Population', 'AveOccup', 'Latitude', 'Longitude'
            ]
            
            # Create DataFrame
            df = pd.DataFrame(X, columns=feature_names)
            df['MedHouseVal'] = y
        
        # Save to CSV
        output_path = os.path.join('data', 'housing_data.csv')
        df.to_csv(output_path, index=False)
        
        # Print success message and dataset info
        print(f"Housing dataset created successfully at '{output_path}'")
        print(f"Dataset shape: {df.shape}")
        print("\nFirst 5 rows of the dataset:")
        print(df.head())
        
        return True
        
    except Exception as e:
        print(f"Failed to create dataset: {str(e)}")
        print("\nPlease check your scikit-learn installation or provide the dataset manually.")
        return False

if __name__ == "__main__":
    create_housing_data()

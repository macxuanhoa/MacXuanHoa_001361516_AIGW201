import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from data_pipeline import DataPipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import joblib
import numpy as np

def train_model(data_path='data/housing_data.csv'):
    """
    Train a house price prediction model and save it to disk.
    
    Args:
        data_path (str): Path to the housing data CSV file
    """
    # Initialize the data pipeline
    pipeline = DataPipeline()
    
    # Load and clean the data
    print("Loading and cleaning data...")
    df = pipeline.load_data(data_path)
    df = pipeline.clean_data(df)
    
    # Prepare training data (this will also create and fit the preprocessor)
    print("Preparing training data...")
    X_train, X_test, y_train, y_test = pipeline.prepare_data(df)
    
    # Train the model
    print("Training model...")
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    
    # Fit the model
    model.fit(X_train, y_train)
    
    # Evaluate the model
    print("\nModel Evaluation:")
    metrics = pipeline.evaluate_model(model, X_test, y_test)
    
    for metric, value in metrics.items():
        print(f"{metric.upper()}: {value:.4f}")
    
    # Create reports directory if it doesn't exist
    os.makedirs('reports/figures', exist_ok=True)
    
    # 1. Feature Importance Plot
    feature_importance = pd.DataFrame({
        'feature': pipeline.numerical_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x='importance', y='feature', data=feature_importance)
    plt.title('Feature Importance')
    plt.tight_layout()
    plt.savefig('reports/figures/feature_importance.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Prediction vs Actual Plot
    y_pred = model.predict(X_test)
    
    plt.figure(figsize=(8, 8))
    plt.scatter(y_test, y_pred, alpha=0.3)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Predicted vs Actual House Prices')
    plt.tight_layout()
    plt.savefig('reports/figures/prediction_vs_actual.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save the model and preprocessor
    print("\nSaving model and preprocessor...")
    pipeline.save_pipeline(model)
    
    print("\nVisualizations saved to reports/figures/")
    print("Training completed successfully!")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train a house price prediction model')
    parser.add_argument('--data', type=str, default='data/housing_data.csv',
                      help='Path to the housing data CSV file')
    
    args = parser.parse_args()
    
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    train_model(args.data)

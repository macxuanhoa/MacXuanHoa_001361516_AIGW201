import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import joblib
import os

class DataPipeline:
    def __init__(self):
        self.preprocessor = None
        # Update column names to match the actual dataset
        self.numerical_cols = ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'Latitude', 'Longitude']
        self.target_col = 'MedHouseVal'  # The target column in the dataset
        self.categorical_cols = []  # No categorical columns in this dataset
        
    def load_data(self, filepath):
        """Load the housing dataset"""
        return pd.read_csv(filepath)
    
    def clean_data(self, df):
        """Handle missing values and outliers"""
        # Ensure column names are stripped of any whitespace
        df.columns = df.columns.str.strip()
        
        # Drop rows with missing target values
        df = df.dropna(subset=[self.target_col])
        
        # Fill missing values in numerical columns
        for col in self.numerical_cols:
            if col in df.columns:
                df[col] = df[col].fillna(df[col].median())
        
        # Remove outliers (using IQR method for numerical columns)
        for col in self.numerical_cols + [self.target_col]:
            if col in df.columns:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                df = df[~((df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR)))]
        
        return df
    
    def feature_engineering(self, df):
        """Create new features for the California housing dataset"""
        # Create rooms per household feature
        if 'AveRooms' in df.columns and 'AveOccup' in df.columns:
            df['RoomsPerHousehold'] = df['AveRooms'] / df['AveOccup']
            
        # Create bedrooms ratio feature
        if 'AveBedrms' in df.columns and 'AveRooms' in df.columns:
            df['BedroomRatio'] = df['AveBedrms'] / df['AveRooms']
            
        # Create population per household feature
        if 'Population' in df.columns and 'AveOccup' in df.columns:
            df['PopulationPerHousehold'] = df['Population'] / df['AveOccup']
            
        # Update numerical columns if we added new features
        new_features = ['RoomsPerHousehold', 'BedroomRatio', 'PopulationPerHousehold']
        self.numerical_cols = [col for col in self.numerical_cols if col not in new_features] + new_features
            
        return df
        
    def create_preprocessor(self, X=None):
        """
        Create and return a preprocessing pipeline for both numerical and categorical features.
        
        Args:
            X: Optional DataFrame to fit the preprocessor. If None, only the preprocessor
               will be created but not fitted.
               
        Returns:
            ColumnTransformer: Configured preprocessor
        """
        # Numerical features pipeline
        numerical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        # Categorical features pipeline
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])
        
        # Combine preprocessing steps
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, self.numerical_cols),
                ('cat', categorical_transformer, self.categorical_cols)
            ])
            
        # Fit the preprocessor if data is provided
        if X is not None:
            self.preprocessor.fit(X)
            
        return self.preprocessor
        
    def prepare_data(self, df, test_size=0.2, random_state=42):
        """Prepare train and test datasets"""
        # Apply feature engineering
        df = self.feature_engineering(df)
        
        # Ensure we only use the columns we've defined
        features = [col for col in self.numerical_cols + self.categorical_cols if col in df.columns]
        X = df[features]
        y = df[self.target_col]
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # Create and fit the preprocessor
        self.create_preprocessor(X_train)
        
        # Transform the features
        X_train = self.preprocessor.transform(X_train)
        X_test = self.preprocessor.transform(X_test)
        
        return X_train, X_test, y_train, y_test
        
    def evaluate_model(self, model, X_test, y_test):
        """Evaluate the model and return metrics"""
        from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
        import numpy as np
        
        y_pred = model.predict(X_test)
        
        # Calculate RMSE manually to avoid version compatibility issues
        mse = mean_squared_error(y_test, y_pred)
        
        metrics = {
            'mse': mse,
            'rmse': np.sqrt(mse),  # Calculate RMSE manually
            'mae': mean_absolute_error(y_test, y_pred),
            'r2': r2_score(y_test, y_pred)
        }
        
        return metrics
        
    def save_pipeline(self, model, output_dir='models', df=None):
        """
        Save the model and preprocessor
        
        Args:
            model: Trained model to save
            output_dir: Directory to save the model and preprocessor
            df: Optional DataFrame to return (for method chaining)
            
        Returns:
            DataFrame if df is provided, else None
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Save model
        joblib.dump(model, os.path.join(output_dir, 'house_price_model.joblib'))
        
        # Save preprocessor if it exists
        if hasattr(self, 'preprocessor'):
            joblib.dump(self.preprocessor, os.path.join(output_dir, 'preprocessor.joblib'))
        
        print(f"Model and preprocessor saved to {output_dir}/")
        return df if df is not None else None
    
    def prepare_features(self, df):
        """Prepare features and target variable"""
        X = df.drop(columns=[self.target_col], errors='ignore')
        y = df[self.target_col]
        return X, y
    
        
        return self.preprocessor
    
    def save_preprocessor(self, filepath):
        """Save the preprocessor to disk"""
        if self.preprocessor:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            joblib.dump(self.preprocessor, filepath)
    
    def load_preprocessor(self, filepath):
        """Load a preprocessor from disk"""
        if os.path.exists(filepath):
            self.preprocessor = joblib.load(filepath)
        return self.preprocessor

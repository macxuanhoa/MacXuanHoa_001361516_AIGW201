import os
import json
import requests
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv
from tools import predict_house_price

# Load environment variables
load_dotenv()

# API Configuration
OLLAMA_API_URL = os.getenv('OLLAMA_API_URL', 'http://localhost:11434/api/chat')
OLLAMA_MODEL = os.getenv('OLLAMA_MODEL', 'llama3')

# Define available functions
FUNCTIONS = [
    {
        "name": "predict_house_price",
        "description": "Predict the price of a house based on its features. Use this when the user asks about house prices, property values, or wants to estimate the cost of a house.",
        "parameters": {
            "type": "object",
            "required": ["area", "bedrooms", "bathrooms"],
            "properties": {
                "area": {
                    "type": "integer",
                    "description": "Area of the house in square feet (e.g., 2000)",
                    "minimum": 100,
                    "maximum": 10000
                },
                "bedrooms": {
                    "type": "integer",
                    "description": "Number of bedrooms (e.g., 3)",
                    "minimum": 1,
                    "maximum": 10
                },
                "bathrooms": {
                    "type": "number",
                    "description": "Number of bathrooms (e.g., 2.5)",
                    "minimum": 1,
                    "maximum": 10
                },
                "floors": {
                    "type": "integer",
                    "description": "Number of floors in the house (default: 1)",
                    "minimum": 1,
                    "maximum": 4,
                    "default": 1
                },
                "stories": {
                    "type": "integer",
                    "description": "Number of stories (default: 1)",
                    "minimum": 1,
                    "maximum": 5,
                    "default": 1
                },
                "parking": {
                    "type": "integer",
                    "description": "Number of parking spaces (default: 1)",
                    "minimum": 0,
                    "default": 1
                },
                "neighborhood": {
                    "type": "string",
                    "enum": ["A", "B", "C", "D"],
                    "description": "Neighborhood code (default: 'A')",
                    "default": "A"
                },
                "house_style": {
                    "type": "string",
                    "enum": ["Apartment", "Villa", "Townhouse"],
                    "description": "Type of house (default: 'Apartment')",
                    "default": "Apartment"
                }
            },
            "required": ["area", "bedrooms", "bathrooms", "stories", "parking", "neighborhood", "house_style"]
        }
    }
]

def run_conversation(user_input):
    # Prepare the initial prompt
    prompt = f"""You are a helpful real estate assistant. You can help predict house prices based on their features.
    
    Available function:
    - predict_house_price: Predict house price based on area, bedrooms, bathrooms, stories, parking, neighborhood, and house style.
    
    User: {user_input}
    
    If the user is asking to predict a house price, please respond with a function call in the following format:
    ```
    FUNCTION: predict_house_price
    ARGS: {{"area": 2000, "bedrooms": 3, "bathrooms": 2, "stories": 2, "parking": 1, "neighborhood": "A", "house_style": "Villa"}}
    ```
    
    Otherwise, respond normally to the user's query."""
    
    # Send the prompt to Ollama
    response = requests.post(
        OLLAMA_API_URL,
        json={
            "model": OLLAMA_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False
        }
    )
    
    if response.status_code != 200:
        return f"Error: Failed to get response from Ollama: {response.text}"
    
    response_text = response.json().get('message', {}).get('content', '').strip()
    
    # Check if the response contains a function call
    if 'FUNCTION:' in response_text and 'ARGS:' in response_text:
        try:
            # Extract function name and arguments
            func_name = response_text.split('FUNCTION:')[1].split('\n')[0].strip()
            args_str = response_text.split('ARGS:')[1].strip()
            
            if func_name == 'predict_house_price':
                # Parse the arguments
                function_args = json.loads(args_str)
                
                # Call the function with the provided arguments
                function_response = predict_house_price(**function_args)
                
                # Format the response
                price = function_response.get('predicted_price', 'N/A')
                low = function_response.get('confidence_interval', {}).get('lower', 'N/A')
                high = function_response.get('confidence_interval', {}).get('upper', 'N/A')
                
                return f"""Based on the trained ML model, the predicted price is:
                - Estimated Price: ${price:,.2f}
                - 95% Confidence Interval: ${low:,.2f} - ${high:,.2f}"""
                
        except Exception as e:
            return f"Error processing function call: {str(e)}"
    
    return response_text

def main():
    print("House Price Prediction Assistant")
    print("Type 'quit' to exit")
    print("\nYou can ask questions like:")
    print("- What would be the price of a 2000 sqft, 3 bedroom, 2 bathroom villa in neighborhood A?")
    print("- Predict the price for a 1500 sqft apartment with 2 bedrooms and 1 bathroom in area C")
    print("- I have a 4 bedroom, 3.5 bathroom townhouse with 2 parking spaces in neighborhood B, what would it cost?")
    
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() == 'quit':
            break
            
        response = run_conversation(user_input)
        print(f"\nAssistant: {response}")

if __name__ == "__main__":
    main()

import streamlit as st
import ollama
import json
import logging
import os
import sys
import time
import uuid
import joblib
import pandas as pd
from datetime import datetime
from typing import Dict, List, Any, Optional, Union, Tuple
from dotenv import load_dotenv
from pymongo.errors import ConnectionFailure

# Add the parent directory to the path to allow importing from components
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import local modules
from components.sidebar import Sidebar
from ml_tools import get_model_metrics

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('app.log')
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Configuration
class Config:
    OLLAMA_MODEL = os.getenv('OLLAMA_MODEL', 'llama3')
    MAX_CHAT_HISTORY = int(os.getenv('MAX_CHAT_HISTORY', '20'))
    DEFAULT_NEIGHBORHOOD = os.getenv('DEFAULT_NEIGHBORHOOD', 'A')
    DEFAULT_HOUSE_STYLE = os.getenv('DEFAULT_HOUSE_STYLE', 'Apartment')
    DEFAULT_STORIES = int(os.getenv('DEFAULT_STORIES', '1'))
    DEFAULT_PARKING = int(os.getenv('DEFAULT_PARKING', '1'))
    
    @classmethod
    def validate(cls):
        """Validate configuration values"""
        if not cls.OLLAMA_MODEL:
            logger.warning("OLLAMA_MODEL not set, using default 'llama3'")
        
        if cls.MAX_CHAT_HISTORY <= 0:
            logger.warning(f"Invalid MAX_CHAT_HISTORY: {cls.MAX_CHAT_HISTORY}, using default 20")
            cls.MAX_CHAT_HISTORY = 20

# Initialize configuration
Config.validate()

# Import dependencies
try:
    from mongo_utils import mongodb_manager
    from ml_tools import get_house_price_prediction
except ImportError as e:
    logger.error(f"Failed to import required modules: {e}")
    mongodb_manager = None
    get_house_price_prediction = None

def get_tools_definition() -> List[Dict[str, Any]]:
    """
    Return the tools definition for function calling
    
    Returns:
        List of tool definitions for the AI assistant
    """
    return [
        {
            "type": "function",
            "function": {
                "name": "get_house_price_prediction",
                "description": "Get a house price prediction based on property features",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "area": {
                            "type": "integer", 
                            "description": "Area of the house in square feet (required)",
                            "minimum": 1
                        },
                        "bedrooms": {
                            "type": "integer", 
                            "description": "Number of bedrooms (required)",
                            "minimum": 1
                        },
                        "bathrooms": {
                            "type": "number", 
                            "description": "Number of bathrooms (required)",
                            "minimum": 0.5
                        },
                        "stories": {
                            "type": "integer", 
                            "description": f"Number of stories (default: {Config.DEFAULT_STORIES})",
                            "default": Config.DEFAULT_STORIES,
                            "minimum": 1
                        },
                        "parking": {
                            "type": "integer", 
                            "description": f"Number of parking spaces (default: {Config.DEFAULT_PARKING})",
                            "default": Config.DEFAULT_PARKING,
                            "minimum": 0
                        },
                        "neighborhood": {
                            "type": "string", 
                            "description": f"Neighborhood code (A, B, C, or D, default: '{Config.DEFAULT_NEIGHBORHOOD}')",
                            "default": Config.DEFAULT_NEIGHBORHOOD,
                            "enum": ["A", "B", "C", "D"]
                        },
                        "house_style": {
                            "type": "string", 
                            "description": f"Type of house (Apartment, Villa, or Townhouse, default: '{Config.DEFAULT_HOUSE_STYLE}')",
                            "default": Config.DEFAULT_HOUSE_STYLE,
                            "enum": ["Apartment", "Villa", "Townhouse"]
                        }
                    },
                    "required": ["area", "bedrooms", "bathrooms"]
                }
            }
        }
    ]

def setup_page() -> None:
    """
    Set up the Streamlit page configuration and custom styles
    """
    st.set_page_config(
        page_title="AI Chat Assistant",
        page_icon="💬",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("💬 AI Chat")
    
    # Custom CSS for better UI/UX
    st.markdown("""
    <style>
        /* Chat container styling */
        .stChatFloatingInputContainer {
            bottom: 20px;
        }
        
        /* Input area styling */
        .stChatInputContainer {
            position: fixed;
            left: 0;
            right: 0;
            bottom: 0;
            padding: 1rem;
            background: white;
            z-index: 1000;
            box-shadow: 0 -2px 10px rgba(0, 0, 0, 0.1);
        }
        
        /* Message styling */
        .stChatMessage {
            padding: 1rem;
            border-radius: 0.5rem;
            margin-bottom: 0.5rem;
        }
        
        /* User message styling */
        [data-testid="stChatMessage"][data-message-author-role="user"] {
            background-color: #f0f2f6;
            margin-left: 20%;
        }
        
        /* Assistant message styling */
        [data-testid="stChatMessage"][data-message-author-role="assistant"] {
            background-color: #ffffff;
            margin-right: 20%;
            border: 1px solid #e0e0e0;
        }
        
        /* Make markdown tables responsive */
        .markdown-table {
            width: 100%;
            overflow-x: auto;
        }
    </style>
    """, unsafe_allow_html=True)

def initialize_session():
    """Initialize session state and load chat history"""
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.user_id = "anonymous"
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
        load_chat_history()

def save_message(role: str, content: str, is_welcome: bool = False) -> None:
    """
    Save message to MongoDB if available
    
    Args:
        role: 'user' or 'assistant'
        content: Message content
        is_welcome: Whether this is the welcome message
    """
    if not mongodb_manager:
        logger.warning("MongoDB manager not available. Message not saved.")
        return
        
    try:
        message_data = {
            "session_id": st.session_state.session_id,
            "user_id": st.session_state.get("user_id", "anonymous"),
            "role": role,
            "content": content,
            "metadata": {
                'app_version': '1.1.0',
                'model': Config.OLLAMA_MODEL,
                        'is_welcome': is_welcome
            }
        }
        
        # Only save non-welcome messages or if it's the first welcome message
        if not is_welcome or not mongodb_manager.chat_history.find_one({
            "session_id": st.session_state.session_id, 
            "metadata.is_welcome": True
        }):
            mongodb_manager.save_chat_message(message_data)
            
    except Exception as e:
        logger.error(f"Error saving message to MongoDB: {e}", exc_info=True)

def load_chat_history():
    """Load chat history from MongoDB or initialize with welcome message"""
    if not mongodb_manager:
        logger.warning("MongoDB manager not available. Using empty chat history.")
        st.session_state.messages = []
        return
        
    try:
        saved_messages = mongodb_manager.get_chat_history(
            session_id=st.session_state.session_id,
            limit=Config.MAX_CHAT_HISTORY
        )
        
        if saved_messages:
            # Sort messages by timestamp in ascending order
            saved_messages.sort(key=lambda x: x.get('timestamp', ''))
            st.session_state.messages = [
                {"role": msg.get("role", "user"), "content": msg.get("content", "")}
                for msg in saved_messages
            ]
        else:
            # Start with empty chat history
            st.session_state.messages = []
    except ConnectionFailure as e:
        logger.error(f"MongoDB connection failed: {e}")
        st.session_state.messages = [{"role": "assistant", "content": "Welcome! Note: Chat history won't be saved due to database connection issues."}]
    except Exception as e:
        logger.error(f"Error loading chat history: {e}", exc_info=True)
        st.session_state.messages = [{"role": "assistant", "content": "Welcome! There was an error loading chat history."}]

def format_prediction_result(result: Dict) -> str:
    """
    Format prediction result into a user-friendly markdown string
    
    Args:
        result: Dictionary containing prediction results
        
    Returns:
        str: Formatted markdown string with prediction results
    """
    if not isinstance(result, dict):
        return "❌ Error: Invalid prediction result format"
        
    if 'error' in result:
        return f"❌ Error: {result['error']}"
    
    try:
        # Format currency values
        def format_currency(value):
            try:
                return "${:,.2f}".format(float(value))
            except (ValueError, TypeError):
                return "N/A"
            
        price = format_currency(result.get('predicted_price', 0))
        
        # Handle confidence interval
        confidence = result.get('confidence_interval', {})
        lower = format_currency(confidence.get('lower', 0))
        upper = format_currency(confidence.get('upper', 0))
        
        # Format features
        features = result.get('features_used', {})
        features_text = "\n".join([f"- **{k.replace('_', ' ').title()}:** {v}" 
                                 for k, v in features.items() if v is not None])
        
        return (
            "## 🏠 House Price Prediction\n\n"
            f"**Estimated Price:** {price}\n"
            f"**Confidence Interval (95%):** {lower} - {upper}\n\n"
            "### Features Used\n"
            f"{features_text}\n\n"
            "*Note: This is an estimate and should not be considered as financial advice.*"
        )
    except Exception as e:
        logger.error(f"Error formatting prediction result: {e}")
        return "❌ Error formatting prediction result. Please try again."

def display_chat_messages():
    """Display chat messages from the session state"""
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"], unsafe_allow_html=True)


def predict_house_price(area: float, bedrooms: int, bathrooms: int, 
                       floors: int = 1, stories: int = 1, 
                       parking: int = 1, location: str = 'Suburbs') -> dict:
    """
    Predict house price based on input parameters
    
    Args:
        area: Area in square meters (m²)
        bedrooms: Number of bedrooms
        bathrooms: Number of bathrooms
        floors: Number of floors (default: 1)
        stories: Number of stories (default: 1)
        parking: Number of parking spaces (default: 1)
        location: Property location (default: 'Suburbs')
        
    Returns:
        dict: Prediction result and related information
    """
    try:
        # Path to model and preprocessor
        script_dir = os.path.dirname(os.path.abspath(__file__))
        preprocessor_path = os.path.join(script_dir, 'models', 'preprocessor.joblib')
        model_path = os.path.join(script_dir, 'models', 'house_price_model.joblib')
        
        # Load model and preprocessor
        preprocessor = joblib.load(preprocessor_path)
        model = joblib.load(model_path)
        
        # Prepare input data
        input_data = {
            'area': [area],
            'bedrooms': [bedrooms],
            'bathrooms': [bathrooms],
            'floors': [floors],
            'stories': [stories],
            'parking': [parking],
            'location': [location]
        }
        
        # Preprocess and predict
        input_df = pd.DataFrame(input_data)
        X_processed = preprocessor.transform(input_df)
        predicted_price = model.predict(X_processed)[0]
        
        return {
            'success': True,
            'price': int(predicted_price),
            'features': input_data
        }
        
    except Exception as e:
        logger.error(f"Error predicting house price: {str(e)}")
        return {
            'success': False,
            'error': str(e)
        }


def handle_chat_message(prompt: str) -> None:
    """
    Handle chat message with Ollama
    
    Args:
        prompt: User's input message
    """
    if not prompt.strip():
        return
        
    # Add user message to chat history
    save_message("user", prompt)
    
    # Display user message on the right
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Generate assistant response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
    
    try:
        # Get the selected model from session state or use default
        model = st.session_state.get('selected_model', Config.OLLAMA_MODEL)
        
        # Get the temperature from session state or use default
        temperature = st.session_state.get('temperature', 0.7)
            
        # Enhanced system prompt to enforce strict JSON output
        system_prompt = """You are a helpful AI assistant for real estate. Follow these rules STRICTLY:
        
        FOR HOUSE PRICE PREDICTIONS:
        1. You MUST output ONLY the JSON object with NO additional text
        2. DO NOT write any introductory text like 'Here is the JSON' or 'I can help'
        3. Strictly NO text before or after the JSON
        4. Use this exact JSON format:
        
        {
            "tool": "predict_house_price",
            "args": {
                "location": "Suburbs",  # Default if not specified
                "area": 0,             # Required
                "bedrooms": 0,         # Required
                "bathrooms": 0,        # Required
                "floors": 1,           # Default to 1 if not specified
                "stories": 1,          # Default to 1 if not specified
                "parking": 1           # Default to 1 if not specified
            }
        }
        
        FOR ALL OTHER QUERIES:
        - Respond normally with text
        - DO NOT include any JSON"""
        
        # Prepare messages for the model
        messages = [
            {"role": "system", "content": system_prompt},
            *[{"role": msg["role"], "content": msg["content"]} 
              for msg in st.session_state.messages[-Config.MAX_CHAT_HISTORY:]],
            {"role": "user", "content": prompt}
                        ]
                        
        # Get response from Ollama
        response = ollama.chat(
            model=model,
            messages=messages,
            options={"temperature": temperature}
        )
        
        # Get the full response content
        full_response = response['message']['content']
        
        # Display the response in the assistant's message
        message_placeholder.markdown(full_response)
        
        # Check if this is a function call by looking for JSON in the response
        import json
        import re
        
        def extract_json(text):
            """Extract JSON from text by finding the first { and last }"""
            try:
                start_idx = text.find('{')
                end_idx = text.rfind('}')
                
                if start_idx == -1 or end_idx == -1 or end_idx < start_idx:
                    return None
                    
                json_str = text[start_idx:end_idx+1]
                
                # Clean up common JSON issues
                json_str = re.sub(r',\s*([\]}])', r'\1', json_str)  # Remove trailing commas
                json_str = re.sub(r'\n', ' ', json_str)  # Remove newlines
                json_str = re.sub(r'\s+', ' ', json_str).strip()  # Normalize whitespace
                
                return json.loads(json_str)
            except Exception as e:
                return None
            
        try:
            # Try to extract and parse JSON from the response
            response_data = extract_json(full_response)
            
            # Check if this is a valid function call for house price prediction
            if response_data and isinstance(response_data, dict) and response_data.get('tool') == 'predict_house_price':
                args = response_data.get('args', {})
                st.sidebar.info(f"🛠️ Calling Tool: predict_house_price({args})")
                
                # Call the prediction function
                prediction = predict_house_price(
                    area=args.get('area', 0),
                    bedrooms=args.get('bedrooms', 0),
                    bathrooms=args.get('bathrooms', 0),
                    floors=args.get('floors', 1),
                    stories=args.get('stories', 1),
                    parking=args.get('parking', 1),
                    location=args.get('location', 'Suburbs')
                )
                
                # Process the prediction result
                if prediction['success']:
                    features = prediction['features']
                    formatted_response = f"""## 🏠 Dự đoán giá nhà (Mô hình ML)
- **Vị trí**: {features['location'][0].capitalize()}
- **Diện tích**: {features['area'][0]} m²
- **Phòng ngủ**: {features['bedrooms'][0]}
- **Phòng tắm**: {features['bathrooms'][0]}
- **Tầng**: {features['floors'][0]}
- **Lầu**: {features['stories'][0]}
- **Chỗ đỗ xe**: {features['parking'][0]} {'chỗ' if features['parking'][0] > 1 else 'chỗ'}
### Giá dự đoán: **${prediction['price']:,}**"""
                else:
                    # Fallback calculation if prediction fails
                    area = args.get('area', 0)
                    bedrooms = args.get('bedrooms', 0)
                    bathrooms = args.get('bathrooms', 0)
                    floors = args.get('floors', 1)
                    # Improved fallback formula with more realistic price estimation
                    base_price_per_sqm = 200  # Base price per square meter
                    bedroom_value = 25000     # Value per bedroom
                    bathroom_value = 15000    # Value per bathroom
                    floor_bonus = 0.05        # 5% bonus per floor above ground
                    
                    # Calculate base price
                    base_price = area * base_price_per_sqm
                    
                    # Add value for bedrooms and bathrooms
                    total_price = base_price + (bedrooms * bedroom_value) + (bathrooms * bathroom_value)
                    
                    # Add floor bonus (5% per floor above ground)
                    if floors > 1:
                        total_price *= (1 + (floors - 1) * floor_bonus)
                    
                    final_price = int(total_price)
                    
                    formatted_response = f"""## ⚠️ House Price Estimation (Fallback Method)
- **Note**: Using fallback formula as the ML model could not be loaded
- **Area**: {area} m²
- **Bedrooms**: {bedrooms}
- **Bathrooms**: {bathrooms}
- **Floors**: {floors}
- **Estimated Price**: ${final_price:,}

*Note: This is an estimated price using a fallback formula. The actual price may vary based on location, market conditions, and other factors.*"""
                
                # Display the response
                message_placeholder.markdown(formatted_response)
                save_message("assistant", formatted_response)
                st.stop()  # Stop further processing to prevent duplicate responses
            else:
                # If not a function call, display as normal
                message_placeholder.markdown(full_response)
                save_message("assistant", full_response)
                
        except json.JSONDecodeError:
            # If response is not JSON, display as normal
            message_placeholder.markdown(full_response)
            save_message("assistant", full_response)
        except Exception as e:
            # Log any other errors and display a generic message
            logger.error(f"Error processing response: {str(e)}")
            error_msg = "I encountered an error processing your request. Please try again."
            message_placeholder.markdown(error_msg)
            save_message("assistant", error_msg)
            
    except Exception as e:
        error_msg = f"Error generating response: {str(e)}"
        logger.error(error_msg, exc_info=True)
        st.error(error_msg)
        save_message("assistant", f"I'm sorry, I encountered an error: {str(e)}")
        return


def check_and_install_model() -> Tuple[bool, str]:
    """
    Check if any models are available and install the default one if not
    
    Returns:
        Tuple[bool, str]: (success, message) - Whether the check/install was successful and a status message
    """
    try:
        # Check if any models are available
        models = ollama.list()
        if models and 'models' in models and models['models']:
            return True, f"Found {len(models['models'])} model(s) available"
            
        # If no models found, install the default one
        st.warning("No LLaMA models found. Installing LLaMA 3 (8B) model...")
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # Start the model download in a separate thread to avoid blocking
            import threading
            
            def download_model():
                try:
                    # This will show a spinner while downloading
                    ollama.pull('llama3')
                    progress_bar.progress(100, "Model downloaded successfully!")
                    time.sleep(1)  # Let the user see the success message
                except Exception as e:
                    status_text.error(f"Failed to install LLaMA 3 model: {str(e)}")
                    return False, f"Failed to install model: {str(e)}"
                finally:
                    progress_bar.empty()
                    status_text.empty()
                return True, "Successfully installed LLaMA 3 model!"
            
            # Start the download in a separate thread
            download_thread = threading.Thread(target=download_model)
            download_thread.start()
            
            # Show a spinner while downloading
            with st.spinner("Downloading LLaMA 3 model (this may take several minutes, ~4.7GB)..."):
                download_thread.join()  # Wait for the download to complete
                
            # Verify the model was installed
            models = ollama.list()
            if models and 'models' in models and models['models']:
                return True, "Successfully installed LLaMA 3 model!"
            else:
                return False, "Failed to verify model installation. Please try again."
            
        except Exception as e:
            progress_bar.empty()
            status_text.error(f"Failed to install LLaMA 3 model: {str(e)}")
            return False, f"Failed to install model: {str(e)}"
            
    except Exception as e:
        return False, f"Error checking/installing models: {str(e)}"

def main():
    """Main application function"""
    setup_page()
    
    # Check for and install model if needed
    with st.spinner("Checking for LLaMA models..."):
        success, message = check_and_install_model()
        if not success:
            st.error(f"Error: {message}")
            if st.button("Try Again"):
                st.rerun()
            return
    
    initialize_session()
    load_chat_history()
    
    # Initialize ML model status with real metrics if available
    ml_model_status = {
        'model_name': 'House Price Predictor',
        'metrics': {'r2': 'N/A', 'mae': 'N/A'},
        'last_prediction': None
    }
    
    # Try to load model metrics if available
    try:
        import joblib
        
        model_path = os.path.join('models', 'house_price_model.joblib')
        if os.path.exists(model_path):
            # Get actual model metrics if available
            metrics = get_model_metrics()
            if metrics:
                ml_model_status['metrics'] = metrics
    except Exception as e:
        logger.warning(f"Could not load model metrics: {e}")
    
    # Initialize and render sidebar
    sidebar = Sidebar(mongodb_manager=mongodb_manager, ml_model_status=ml_model_status)
    sidebar.render()
    
    # Display chat messages
    display_chat_messages()
    
    # Chat input
    if prompt := st.chat_input("What would you like to know?"):
        # Update ML model status with last prediction if available
        if 'last_prediction' in st.session_state:
            ml_model_status['last_prediction'] = st.session_state.last_prediction
            
        # Handle the chat message
        handle_chat_message(prompt)
        
        # Rerun to update the UI
        st.rerun()


if __name__ == "__main__":
    main()


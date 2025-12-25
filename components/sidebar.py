import streamlit as st
import logging
import ollama
import uuid
from typing import Dict, Any, Optional
from datetime import datetime

class Sidebar:
    """
    Sidebar component for the application
    """
    
    def __init__(self, mongodb_manager=None, ml_model_status: Optional[Dict[str, Any]] = None):
        """
        Initialize the sidebar
        
        Args:
            mongodb_manager: MongoDB manager instance for chat history
            ml_model_status: Dictionary containing ML model status information
        """
        self.mongodb_manager = mongodb_manager
        self.ml_model_status = ml_model_status or {}
        self.logger = logging.getLogger(__name__)
    
    def _render_ml_model_status(self) -> None:
        """
        Render the ML Model Status section with evaluation metrics
        """
        st.sidebar.markdown("## 📊 ML Model Status")
        
        with st.sidebar.expander("House Price Prediction Model", expanded=True):
            if not self.ml_model_status:
                st.warning("ML model not loaded")
                return
                
            st.markdown("### 🏠 House Price Prediction")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("R2 Score", f"{self.ml_model_status.get('metrics', {}).get('r2', 0):.2f}")
            with col2:
                st.metric("MAE", f"${self.ml_model_status.get('metrics', {}).get('mae', 0):,.0f}")
            
            if 'model_name' in self.ml_model_status:
                st.caption(f"Model: {self.ml_model_status['model_name']}")
                
            if 'last_trained' in self.ml_model_status:
                st.caption(f"Last trained: {self.ml_model_status['last_trained']}")
            
            st.success("✅ Model is ready for predictions")
    
    def _render_model_config(self) -> None:
        """
        Render the Model Configuration section with model selection and settings
        """
        st.sidebar.markdown("## ⚙️ Model Configuration")
        
        with st.sidebar.expander("Model Settings", expanded=True):
            try:
                # Get available models
                models_response = ollama.list()
                
                # Check if we got a valid response with models
                if not models_response or 'models' not in models_response or not models_response['models']:
                    st.warning("No LLaMA models found. Please install a model to get started.")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("🔄 Check Again", help="Check for installed models"):
                            st.rerun()
                    with col2:
                        if st.button("⬇️ Install LLaMA 3 (8B)", help="Download the LLaMA 3 8B model"):
                            with st.spinner("Downloading LLaMA 3 8B model (this may take several minutes, ~4.7GB)..."):
                                try:
                                    ollama.pull('llama3')
                                    st.success("✅ Successfully installed LLaMA 3 model!")
                                    st.rerun()
                                except Exception as pull_error:
                                    st.error(f"❌ Failed to install model: {str(pull_error)}")
                    
                    st.info("💡 You can also install models manually using the command: `ollama pull <model_name>`")
                    return
                
                # Extract model names and details
                models = []
                model_details = {}
                
                # First pass: collect all models
                for model in models_response['models']:
                    if 'name' in model:
                        model_name = model['name']
                        # Store both full name and base name (without tag)
                        base_name = model_name.split(':')[0]
                        models.append(model_name)
                        model_details[model_name] = {
                            'base_name': base_name,
                            'size': model.get('size', 0),
                            'modified': model.get('modified_at', ''),
                            'digest': model.get('digest', '')[:8]  # Show first 8 chars of digest
                        }
                
                # Remove duplicates while preserving order
                seen = set()
                unique_models = []
                for model in models:
                    if model not in seen:
                        seen.add(model)
                        unique_models.append(model)
                
                models = unique_models
                
                if not models:
                    st.warning("No valid models found. Please install a model.")
                    return
                
                # Set default model if not set
                if 'selected_model' not in st.session_state:
                    st.session_state['selected_model'] = models[0]
                
                # Model selection with details
                st.markdown("### Available LLaMA Models")
                
                # Create a more informative display of models
                model_options = []
                for model in models:
                    details = model_details[model]
                    size_gb = details['size'] / (1024**3)
                    modified = details['modified'].split('T')[0] if details['modified'] else 'unknown'
                    model_options.append({
                        'label': f"{details['base_name']} ({size_gb:.1f}GB, {modified})",
                        'value': model
                    })
                
                # Default selection
                default_index = 0
                if 'selected_model' in st.session_state:
                    for i, model in enumerate(models):
                        if model == st.session_state.selected_model:
                            default_index = i
                            break
                
                # Display model selection
                selected_model = st.selectbox(
                    "Select Model",
                    models,
                    index=default_index,
                    format_func=lambda x: f"{model_details[x]['base_name']} ({model_details[x]['size']/(1024**3):.1f}GB)",
                    help="Select the LLaMA model to use for generating responses"
                )
                
                # Show detailed model information
                if selected_model in model_details:
                    details = model_details[selected_model]
                    st.markdown("### Model Details")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Model Name", details['base_name'])
                        st.metric("Version", selected_model.split(':')[1] if ':' in selected_model else 'latest')
                    with col2:
                        st.metric("Size", f"{details['size'] / (1024**3):.1f} GB")
                        st.metric("Last Updated", details['modified'].split('T')[0] if details['modified'] else 'unknown')
                    
                    if 'digest' in details:
                        st.caption(f"Digest: {details['digest']}")
                
                # Add a button to install new models
                if st.button("⬇️ Install New Model"):
                    with st.expander("Install New Model", expanded=True):
                        st.markdown("### Install LLaMA Models")
                        model_name = st.text_input("Model name (e.g., llama3, llama2, mistral)", "llama3")
                        
                        if st.button("Install"):
                            if model_name:
                                with st.spinner(f"Downloading {model_name} model (this may take several minutes)..."):
                                    try:
                                        ollama.pull(model_name)
                                        st.success(f"✅ Successfully installed {model_name} model!")
                                        st.rerun()
                                    except Exception as e:
                                        st.error(f"❌ Failed to install model: {str(e)}")
                            else:
                                st.warning("Please enter a model name")
                
                st.session_state['selected_model'] = selected_model
                
                # Model parameters
                st.markdown("### Model Parameters")
                
                # Temperature slider
                if 'temperature' not in st.session_state:
                    st.session_state['temperature'] = 0.7
                    
                temperature = st.slider(
                    "Temperature",
                    min_value=0.0,
                    max_value=2.0,
                    value=st.session_state.temperature,
                    step=0.1,
                    help="Controls randomness in the model's responses. Lower = more deterministic."
                )
                st.session_state['temperature'] = temperature
                
                # Add a button to refresh model list
                if st.button("🔄 Refresh Models"):
                    st.rerun()
                
            except Exception as e:
                self.logger.error("Error in model configuration: %s", str(e), exc_info=True)
                st.error("Error loading model configuration")
                st.code(f"Error details: {str(e)}")
                
                # Set default values if we can't load models
                if 'selected_model' not in st.session_state:
                    st.session_state['selected_model'] = 'llama3'
                if 'temperature' not in st.session_state:
                    st.session_state['temperature'] = 0.7
    
    def _render_ml_status(self) -> None:
        """Render the ML Model Status section (legacy, kept for backward compatibility)"""
        pass  # This method is kept for backward compatibility but is no longer used
    
    def _render_chat_history(self) -> None:
        """
        Render the Chat History section with session management
        """
        st.sidebar.markdown("## 💬 Chat History")
        
        # Initialize current_session if not exists
        if 'current_session' not in st.session_state:
            st.session_state.current_session = str(uuid.uuid4())
            st.session_state.messages = []
        
        # Show current session info with a button to start new chat
        col1, col2 = st.sidebar.columns([3, 1])
        col1.caption(f"Current Session: {st.session_state.current_session[:8]}...")
        if col2.button("💬 New"):
            st.session_state.current_session = str(uuid.uuid4())
            st.session_state.messages = []
            st.rerun()
        
        if not self.mongodb_manager:
            st.sidebar.info("Chat history will be temporarily stored in memory")
            return
            
        try:
            # Get recent chat sessions
            sessions = []
            try:
                sessions = self.mongodb_manager.get_chat_sessions(limit=10) or []
            except Exception as db_error:
                self.logger.warning("Could not load chat history: %s", str(db_error))
            
            if not sessions:
                st.sidebar.info("No chat history available")
                return
                
            # Add a divider before the list
            st.sidebar.markdown("---")
            st.sidebar.caption("Recent Sessions")
            
            # Display session list with delete option
            for session in sessions:
                if not session or not isinstance(session, dict):
                    continue
                    
                session_id = session.get('_id')
                if not session_id:
                    continue
                
                # Create columns for session info and delete button
                col1, col2 = st.sidebar.columns([4, 1])
                
                # Format timestamp and preview
                timestamp = session.get('timestamp', datetime.now())
                if isinstance(timestamp, str):
                    try:
                        timestamp = datetime.fromisoformat(timestamp)
                    except (ValueError, TypeError):
                        timestamp = datetime.now()
                
                preview = str(session.get('preview', 'New chat'))
                display_text = f"{timestamp.strftime('%d/%m %H:%M')} - {preview[:25]}{'...' if len(preview) > 25 else ''}"
                
                # Session button
                if col1.button(display_text, key=f"session_btn_{session_id}", use_container_width=True):
                    if session_id != st.session_state.current_session:
                        st.session_state.current_session = session_id
                        # Load messages for this session
                        try:
                            session_data = self.mongodb_manager.get_chat_session(session_id)
                            st.session_state.messages = session_data.get('messages', [])
                        except Exception as e:
                            self.logger.error(f"Error loading session {session_id}: {str(e)}")
                            st.session_state.messages = []
                        st.rerun()
                
                # Delete button
                if col2.button("🗑️", key=f"del_{session_id}", help="Delete this session"):
                    try:
                        self.mongodb_manager.delete_chat_session(session_id)
                        st.rerun()
                    except Exception as e:
                        st.sidebar.error(f"Error deleting session: {str(e)}")
        
        except Exception as e:
            self.logger.error("Error in chat history: %s", str(e), exc_info=True)
            st.sidebar.warning("Failed to load chat history")
    
    def _render_developer_options(self) -> None:
        """
        Render the Developer Options section with enhanced function call logging
        """
        if not st.sidebar.checkbox("🛠️ Developer Options", False):
            return
            
        st.sidebar.markdown("### Activity Log")
        
        # Initialize function_calls if not exists
        if 'function_calls' not in st.session_state:
            st.session_state.function_calls = []
        
        # Filter options
        st.sidebar.markdown("**Filters:**")
        col1, col2 = st.sidebar.columns(2)
        with col1:
            show_errors = st.checkbox("Errors", value=True, key="show_errors")
        with col2:
            show_success = st.checkbox("Success", value=True, key="show_success")
        
        # Search box
        search_term = st.sidebar.text_input("Search in logs:", "")
        
        # Display function calls
        if st.session_state.function_calls:
            filtered_calls = []
            for call in reversed(st.session_state.function_calls):
                # Apply filters
                if not show_errors and call.get('status') == 'error':
                    continue
                if not show_success and call.get('status') == 'success':
                    continue
                if search_term and search_term.lower() not in str(call).lower():
                    continue
                filtered_calls.append(call)
            
            st.sidebar.markdown(f"**Function Call History ({len(filtered_calls)}/{len(st.session_state.function_calls)})**")
            
            if not filtered_calls:
                st.sidebar.info("No matching results")
            else:
                for call in filtered_calls[:10]:  # Limit to 10 most recent filtered calls
                    call_name = call.get('name', 'unknown')
                    call_time = call.get('timestamp', '')
                    status = call.get('status', 'unknown')
                    call_time_str = ''  # Initialize with default value
                    
                    # Format timestamp
                    if call_time:
                        try:
                            if isinstance(call_time, str):
                                call_time = datetime.fromisoformat(call_time)
                            call_time_str = call_time.strftime('%H:%M:%S')
                        except (ValueError, TypeError):
                            call_time_str = 'Invalid time'
                    
                    # Status indicator
                    status_emoji = "✅" if status == 'success' else "❌" if status == 'error' else "ℹ️"
                    
                    # Create expander with status and time
                    with st.sidebar.expander(f"{status_emoji} {call_time_str} - {call_name}", expanded=False):
                        # Show function call details
                        st.markdown(f"**Function:** `{call_name}`")
                        
                        # Show execution time if available
                        if 'execution_time' in call:
                            st.caption(f"⏱️ Execution time: {call['execution_time']:.2f}s")
                        
                        # Show arguments
                        if 'args' in call or 'kwargs' in call:
                            st.markdown("**Parameters:**")
                            args = call.get('args', [])
                            kwargs = call.get('kwargs', {})
                            if args or kwargs:
                                args_str = ', '.join([repr(a) for a in args])
                                kwargs_str = ', '.join([f"{k}={repr(v)}" for k, v in kwargs.items()])
                                params = ', '.join(filter(None, [args_str, kwargs_str]))
                                st.code(f"{call_name}({params})")
                        
                        # Show result or error
                        if status == 'error':
                            st.error(f"Error: {call.get('error', 'Unknown error')}")
                        elif 'result' in call:
                            st.markdown("**Result:**")
                            st.json(call['result'])
                        
                        # Show raw data for debugging
                        with st.expander("📝 Raw Data", False):
                            st.json(call)
        else:
            st.sidebar.info("No activity logs available")
            
        # Add buttons for log management
        col1, col2 = st.sidebar.columns(2)
        if col1.button("🔄 Refresh"):
            st.rerun()
        if col2.button("🗑️ Clear Logs"):
            st.session_state.function_calls = []
            st.rerun()
    
    def render(self) -> None:
        """
        Render the sidebar with all components
        """
        self._render_ml_model_status()
        st.sidebar.markdown("---")
        self._render_model_config()
        
        # Add chat history and developer options
        st.sidebar.markdown("---")
        self._render_chat_history()
        
        # Add developer options at the bottom
        st.sidebar.markdown("---")
        self._render_developer_options()
        
        # Add some spacing and footer
        st.sidebar.markdown("\n\n")
        st.sidebar.markdown("---")
        st.sidebar.markdown("*Powered by Ollama*")

# Example usage:
# sidebar = Sidebar(mongodb_manager, ml_model_status={
#     'model_name': 'House Price Predictor',
#     'metrics': {'r2': 0.85, 'mae': 25000},
#     'last_prediction': {'predicted_price': 350000, 'confidence': 0.92}
# })
# sidebar.render()

import os
from pymongo import MongoClient, IndexModel, ASCENDING, DESCENDING
from pymongo.errors import ConnectionFailure, DuplicateKeyError
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from uuid import uuid4
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class MongoDBManager:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(MongoDBManager, cls).__new__(cls)
            cls._instance._initialize_connection()
        return cls._instance
    
    def _initialize_connection(self):
        """Initialize MongoDB connection and collections"""
        try:
            mongodb_uri = os.getenv('MONGODB_URI', 'mongodb://localhost:27017/')
            db_name = os.getenv('MONGODB_DB_NAME', 'ai_manager_assistant')
            
            print(f"Connecting to MongoDB at: {mongodb_uri}")
            print(f"Using database: {db_name}")
            
            # Create connection with a short timeout for testing
            self.client = MongoClient(
                mongodb_uri,
                serverSelectionTimeoutMS=5000  # 5 second timeout
            )
            
            # Test the connection
            print("Testing MongoDB connection...")
            self.client.admin.command('ping')
            print("MongoDB connection successful!")
            
            # List all databases before creating new one
            print("\nCurrent databases:")
            for db in self.client.list_database_names():
                print(f"- {db}")
            
            # Initialize database and collections
            print(f"\nInitializing database: {db_name}")
            self.db = self.client[db_name]
            
            # Initialize collections
            print("Initializing collections...")
            self._init_chat_history()
            self._init_predictions()
            self._init_analysis_results()
            self._init_raw_data()
            
            # Create test data to ensure collections are created
            print("\nCreating test data...")
            self._create_test_data()
            
            # List collections after initialization
            print("\nCollections in database:")
            for col in self.db.list_collection_names():
                print(f"- {col}")
            
            print("\nMongoDB initialization completed successfully!")
            
        except Exception as e:
            print(f"\nMONGODB CONNECTION ERROR:")
            print(f"- Please check if MongoDB is running")
            print(f"- Verify connection string: {mongodb_uri}")
            print(f"- Error details: {e}")
            raise
    
    def _create_test_data(self):
        """Create test data to ensure collections are created"""
        try:
            # Test chat history
            self.chat_history.insert_one({
                '_id': 'test_chat_001',
                'session_id': 'test_session',
                'user_id': 'test_user',
                'role': 'user',
                'content': 'Đây là tin nhắn kiểm tra',
                'timestamp': datetime.utcnow(),
                'metadata': {'test': True}
            })
            
            # Test prediction
            self.predictions.insert_one({
                'input': {'test': 'data'},
                'prediction': {'result': 'test_prediction'},
                'timestamp': datetime.utcnow(),
                'model_version': '1.0.0'
            })
            
            print("Test data created successfully in MongoDB")
            
        except Exception as e:
            print(f"Failed to create test data: {e}")
    
    def _init_chat_history(self):
        """Initialize chat history collection with indexes"""
        self.chat_history = self.db['chat_history']
        # Create indexes for faster querying
        self.chat_history.create_indexes([
            IndexModel([('session_id', 1)]),
            IndexModel([('timestamp', -1)]),
            IndexModel([('user_id', 1)])
        ])
    
    def _init_predictions(self):
        """Initialize predictions collection with indexes"""
        self.predictions = self.db['predictions']
        self.predictions.create_index([('timestamp', -1)])
    
    def _init_analysis_results(self):
        """Initialize analysis results collection with indexes"""
        self.analysis_results = self.db['analysis_results']
        self.analysis_results.create_indexes([
            IndexModel([('analysis_id', 1)], unique=True),
            IndexModel([('timestamp', -1)]),
            IndexModel([('model_name', 1)])
        ])
    
    def _init_raw_data(self):
        """Initialize raw data collection with indexes"""
        self.raw_data = self.db['raw_data']
        self.raw_data.create_index([('data_source', 1), ('timestamp', -1)])

    # ===== Chat History Methods =====
    def save_chat_message(self, message: Dict[str, Any]) -> str:
        """
        Save a chat message to the database
        
        Args:
            message: {
                'session_id': str,       # Unique session identifier
                'user_id': str,          # User identifier
                'role': str,             # 'user' or 'assistant'
                'content': str,          # The message content
                'metadata': dict,        # Any additional metadata
            }
            
        Returns:
            str: ID of the inserted document
        """
        message.update({
            '_id': str(uuid4()),
            'timestamp': datetime.utcnow()
        })
        
        result = self.chat_history.insert_one(message)
        return str(result.inserted_id)
    
    def get_chat_history(self, session_id: str, limit: int = 20) -> List[Dict]:
        """
        Retrieve chat history for a specific session
        
        Args:
            session_id: The session ID to retrieve history for
            limit: Maximum number of messages to return
            
        Returns:
            List of chat messages, most recent first
        """
        return list(self.chat_history
                  .find({'session_id': session_id})
                  .sort('timestamp', -1)
                  .limit(limit))
    
    # ===== Data Pipeline Methods =====
    def save_raw_data(self, data_source: str, data: Dict) -> str:
        """
        Save raw data to the database for processing in the data pipeline
        
        Args:
            data_source: Identifier for the data source
            data: The raw data to store
            
        Returns:
            str: ID of the inserted document
        """
        document = {
            '_id': str(uuid4()),
            'data_source': data_source,
            'data': data,
            'timestamp': datetime.utcnow(),
            'status': 'pending'
        }
        
        result = self.raw_data.insert_one(document)
        return str(result.inserted_id)
    
    def get_pending_raw_data(self, data_source: str, limit: int = 100) -> List[Dict]:
        """
        Retrieve raw data that needs processing
        
        Args:
            data_source: Filter by data source
            limit: Maximum number of documents to return
            
        Returns:
            List of pending raw data documents
        """
        return list(self.raw_data
                  .find({
                      'data_source': data_source,
                      'status': 'pending'
                  })
                  .sort('timestamp', 1)
                  .limit(limit))
    
    def update_raw_data_status(self, doc_id: str, status: str, **updates) -> bool:
        """
        Update the status of a raw data document
        
        Args:
            doc_id: Document ID to update
            status: New status (e.g., 'processing', 'processed', 'error')
            updates: Additional fields to update
            
        Returns:
            bool: True if update was successful
        """
        updates['status'] = status
        updates['last_updated'] = datetime.utcnow()
        
        result = self.raw_data.update_one(
            {'_id': doc_id},
            {'$set': updates}
        )
        
        return result.modified_count > 0
    
    # ===== Analysis Results Methods =====
    def save_analysis_result(self, result: Dict) -> str:
        """
        Save analysis results to the database
        
        Args:
            result: {
                'analysis_id': str,      # Unique analysis identifier
                'model_name': str,       # Name/version of the model
                'input_data': dict,      # Input data used for analysis
                'results': dict,         # Analysis results
                'metadata': dict,        # Additional metadata
            }
            
        Returns:
            str: ID of the inserted document
        """
        result.update({
            '_id': str(uuid4()),
            'timestamp': datetime.utcnow()
        })
        
        # Ensure analysis_id is unique
        if 'analysis_id' not in result:
            result['analysis_id'] = f"analysis_{uuid4().hex[:8]}"
        
        try:
            self.analysis_results.insert_one(result)
            return result['analysis_id']
        except DuplicateKeyError:
            # If analysis_id already exists, generate a new one and retry
            result['analysis_id'] = f"analysis_{uuid4().hex[:8]}"
            self.analysis_results.insert_one(result)
            return result['analysis_id']
    
    def get_analysis_result(self, analysis_id: str) -> Optional[Dict]:
        """
        Retrieve analysis results by ID
        
        Args:
            analysis_id: The analysis ID to retrieve
            
        Returns:
            The analysis document or None if not found
        """
        return self.analysis_results.find_one({'analysis_id': analysis_id})
    
    def get_recent_analyses(self, model_name: str = None, limit: int = 10) -> List[Dict]:
        """
        Get recent analysis results
        
        Args:
            model_name: Filter by model name (optional)
            limit: Maximum number of results to return
            
        Returns:
            List of analysis results, most recent first
        """
        query = {}
        if model_name:
            query['model_name'] = model_name
            
        return list(self.analysis_results
                  .find(query)
                  .sort('timestamp', -1)
                  .limit(limit))
    
    # ===== Prediction Methods (existing functionality) =====
    def save_prediction(self, input_data: Dict, prediction_result: Dict) -> str:
        """
        Save prediction to MongoDB
        
        Args:
            input_data: Input data used for prediction
            prediction_result: Prediction result from the model
            
        Returns:
            str: ID of the inserted document
        """
        document = {
            'input': input_data,
            'prediction': prediction_result,
            'timestamp': datetime.utcnow(),
            'model_version': '1.0.0'
        }
        
        result = self.predictions.insert_one(document)
        return str(result.inserted_id)
    
    def get_recent_predictions(self, limit: int = 10) -> List[Dict]:
        """
        Get recent predictions from the database
        
        Args:
            limit: Number of recent predictions to return
            
        Returns:
            List of recent prediction documents
        """
        return list(self.predictions
                  .find()
                  .sort('timestamp', -1)
                  .limit(limit))

# Create a singleton instance
mongodb_manager = MongoDBManager()

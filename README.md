# 🏠 AI Real Estate Assistant

<div align="center">
  <p>An intelligent chatbot that helps users search for information and predict house prices using Local LLM and Machine Learning</p>
  <img src="https://img.shields.io/badge/Python-3.11-blue" alt="Python 3.11">
  <img src="https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white" alt="Streamlit">
  <img src="https://img.shields.io/badge/Ollama-LLM-7F52FF" alt="Ollama LLM">
  <img src="https://img.shields.io/badge/Scikit--learn-F7931E?logo=scikitlearn&logoColor=white" alt="Scikit-learn">
</div>

## 🌟 Project Overview
AI Real Estate Assistant is a Minimum Viable Product (MVP) that combines the power of Local Large Language Models (LLaMA 3) with Machine Learning to provide an intelligent real estate consultation experience. The application allows users to have natural conversations about real estate and get AI-powered house price predictions based on property features.

## ✨ Key Features

### 💬 Natural Conversation
- Chat naturally in English with LLaMA 3 running locally via Ollama
- No external API calls required, ensuring data privacy

### 🧠 Context Awareness
- Built-in memory system to maintain conversation context
- Remembers previous interactions for more coherent conversations

### 🏠 House Price Prediction
- Automatic Function Calling to trigger ML model predictions
- Get instant price estimates based on property features
- Visual representation of prediction results

### 📊 Data Analysis
- Based on the California Housing Dataset
- Provides insights into real estate market trends

## 🛠️ Tech Stack
- **Frontend**: Streamlit
- **LLM Engine**: Ollama (LLaMA 3)
- **Machine Learning**: Scikit-learn (Random Forest Regressor)
- **Model Persistence**: Joblib
- **Data Processing**: Pandas, NumPy
- **Visualization**: Matplotlib, Plotly

## 🚀 Installation Guide

### Prerequisites
- Python 3.11
- Ollama installed and running

### Setup Instructions
1. **Clone the repository**
   ```bash
   git clone https://github.com/macxuanhoa/MacXuanHoa_001361516_AIGW201.git
   cd MacXuanHoa_001361516_AIGW201
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up Ollama**
   - Download and install Ollama from [ollama.ai](https://ollama.ai/)
   - Pull the LLaMA 3 model:
     ```bash
     ollama pull llama3
     ```

4. **Run the application**
   ```bash
   streamlit run app.py
   ```

## 🤖 Model Details

### Dataset
- **Name**: California Housing Dataset
- **Source**: Scikit-learn
- **Features**:
  - MedInc: Median income in block group
  - HouseAge: Median house age in block group
  - AveRooms: Average number of rooms per household
  - AveBedrms: Average number of bedrooms per household
  - Population: Block group population
  - AveOccup: Average number of household members
  - Latitude: Block group latitude
  - Longitude: Block group longitude

### Model Configuration & Performance
- **Algorithm**: Random Forest Regressor
- **Hyperparameters**:
  - `n_estimators`: 100
  - `max_depth`: 10
  - `random_state`: 42
- **Performance Metrics**:
  - **R² Score**: 0.7439
  - **MAE**: 0.3245
  - **RMSE**: 0.4630
- **Model Location**: `models/house_price_model.joblib`

### 🔄 Data Pipeline
The system processes raw data through a rigorous pipeline:
1. **Data Cleaning**: Handling missing values and outliers.
2. **Feature Engineering**: Creating derived features like `Rooms_per_Household`.
3. **Preprocessing**: Scaling numerical features using StandardScaler.
4. **Training**: Fitting the Random Forest model on processed data.

## 📁 Project Structure
```
MacXuanHoa_001361516_AIGW201/
├── .env                    # Environment variables
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── README.md             # This file
├── data/                  # Data files
│   └── housing_data.csv   # Sample housing data
├── models/                # Trained models
│   ├── house_price_model.joblib
│   └── preprocessor.joblib
├── reports/               # Reports and figures
│   └── figures/           # Visualization outputs
├── scripts/               # Utility scripts
└── tests/                 # Test files
```

## 📝 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Author
- **Mac Xuan Hoa** - GCD220422

## 🙏 Acknowledgments
- Scikit-learn for the California Housing Dataset
- Ollama for the local LLM infrastructure
- Streamlit for the web interface
- The open-source community for invaluable tools and libraries
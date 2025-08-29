# Sales AI Analyst

An AI-powered sales data analysis tool that enables natural language querying and automated insights generation using Google's Gemini API.

## 🎯 Features

- **Natural Language Queries**: Ask questions about your sales data in plain English
- **AI-Powered Analysis**: Gemini API generates Python code for data analysis
- **Interactive Visualizations**: Dynamic charts and graphs using Plotly
- **Sales Forecasting**: Time series forecasting with Prophet
- **Safe Code Execution**: Sandboxed environment for AI-generated code
- **File Upload Support**: CSV and Excel file uploads
- **Chat Interface**: Streamlit-based conversational interface

## 🛠️ Tech Stack

- **Frontend**: Streamlit
- **AI Engine**: Google Gemini API
- **Data Processing**: Pandas, NumPy
- **Visualizations**: Plotly
- **Forecasting**: Prophet
- **Environment**: Python 3.11+

## 📋 Requirements

- Python 3.11+
- Gemini API key (from Google AI Studio)
- 2GB+ RAM recommended
- Internet connection for API calls

## 🚀 Quick Start

### Windows (PowerShell)

```powershell
# Clone or download the project
# Navigate to project directory

# Create virtual environment
python -m venv venv
venv\Scripts\activate

# Install dependencies
pip install streamlit google-genai pandas numpy plotly prophet python-dotenv pydantic pydantic-settings

# Setup environment
copy .env.example .env
# Edit .env file and add your GEMINI_API_KEY

# Run the application
python -m streamlit run app.py

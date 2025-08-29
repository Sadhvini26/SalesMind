# 🤖 SalesMind – Agentic AI Sales Data Analyst

SalesMind is an **Agentic AI-powered sales data analyst** that transforms raw sales data into actionable insights. It combines **Google Gemini API** with a **multi-agent AI architecture** to understand natural language queries and generate **business-ready analysis** with charts, tables, and insights.

---

## 🚀 Features

✅ **Agentic AI Architecture** – Autonomous decision engine routes queries to specialized agents  
✅ **Natural Language Queries** – Just ask questions like *"Which customers contribute the most revenue?"*  
✅ **Multi-Agent System**:
- 📈 **Trends Agent** → Monthly/seasonal sales trends  
- 🏆 **Ranking Agent** → Top products, customers, categories  
- 🌍 **Regional Agent** → Regional performance & insights  
- 🔮 **Forecasting Agent** → Future sales predictions with Prophet  

✅ **Error Recovery** – Automatically fixes data formatting issues  
✅ **Interactive Visuals** – Clear charts and summaries in the UI  
✅ **Secure Sandbox Execution** – Ensures safe analysis without risky code execution

---

## 🛠️ Tech Stack

- **Core AI**: Google Gemini API (Natural Language → Analysis Plans)  
- **Agentic System**: Custom-built specialized agents in Python  
- **Forecasting**: Facebook Prophet for time-series forecasting  
- **Frontend**: Streamlit interactive UI  
- **Data Handling**: Pandas, PyArrow  
- **Security**: AST-based sandbox for safe execution

---

## 📋 Requirements

- Python 3.11+
- Gemini API key (from Google AI Studio)
- 2GB+ RAM recommended
- Internet connection for API calls

---

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
```

### macOS/Linux

```bash
# Clone or download the project
# Navigate to project directory

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install streamlit google-genai pandas numpy plotly prophet python-dotenv pydantic pydantic-settings

# Setup environment
cp .env.example .env
# Edit .env file and add your GEMINI_API_KEY

# Run the application
python -m streamlit run app.py
```

---


## 🔧 Configuration

1. **Get Gemini API Key**:
   - Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
   - Create a new API key
   - Copy the key

2. **Setup Environment**:
   ```bash
   cp .env.example .env
   ```
   
3. **Edit `.env` file**:
   ```env
   GEMINI_API_KEY=your_gemini_api_key_here
   ```

---

## 📊 Usage Examples

### Sample Queries

- **Trends**: *"Show me monthly sales trends for 2024"*
- **Rankings**: *"Which are the top 10 products by revenue?"*
- **Regional**: *"Compare sales performance across regions"*
- **Forecasting**: *"Predict next quarter's sales"*

---

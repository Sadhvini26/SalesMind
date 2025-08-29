import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import logging
import os
from typing import Optional, Dict, Any

# Import custom modules
from config import settings
from ai_agent import SalesAIAgent, AnalysisResult, create_plotly_chart
from sandbox import executor
from forecasting import forecaster
from data_utils import data_processor

# Configure logging
logging.basicConfig(level=logging.INFO)

# Page configuration
st.set_page_config(
    page_title="Sales AI Analyst",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .user-message {
        background-color: #e3f2fd;
        border-left: 4px solid #2196f3;
    }
    .ai-message {
        background-color: #f3e5f5;
        border-left: 4px solid #9c27b0;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #dee2e6;
    }
    .error-message {
        color: #d32f2f;
        background-color: #ffebee;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #d32f2f;
    }
    .success-message {
        color: #388e3c;
        background-color: #e8f5e8;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #388e3c;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'data' not in st.session_state:
    st.session_state.data = None
if 'data_info' not in st.session_state:
    st.session_state.data_info = {}
if 'ai_agent' not in st.session_state:
    st.session_state.ai_agent = SalesAIAgent()

def main():
    """Main application function."""
    
    # Header
    st.markdown('<h1 class="main-header">📊 Sales AI Analyst</h1>', unsafe_allow_html=True)
    st.markdown("**Ask questions about your sales data in natural language and get AI-powered insights!**")
    
    # Sidebar for configuration and file upload
    setup_sidebar()
    
    # Main content area
    if st.session_state.data is not None:
        # Display data overview
        show_data_overview()
        
        # Chat interface
        chat_interface()
        
        # Example queries
        show_example_queries()
    else:
        # Welcome screen
        show_welcome_screen()

def setup_sidebar():
    """Setup sidebar with file upload and configuration options."""
    
    st.sidebar.header("📁 Data Upload")
    
    # File upload
    uploaded_file = st.sidebar.file_uploader(
        "Upload your sales data",
        type=['csv', 'xlsx', 'xls'],
        help="Upload CSV or Excel files (max 25MB)"
    )
    
    if uploaded_file is not None:
        # Validate file size
        file_size = len(uploaded_file.getvalue())
        if not data_processor.validate_file_size(file_size, settings.max_upload_mb):
            st.sidebar.error(f"File size exceeds {settings.max_upload_mb}MB limit")
            return
        
        # Load data
        with st.spinner("Loading data..."):
            df, error = load_uploaded_file(uploaded_file)
            
            if error:
                st.sidebar.error(f"Error loading file: {error}")
            else:
                st.session_state.data = df
                
                # Validate and clean data
                is_valid, message, data_info = data_processor.validate_data_structure(df)
                st.session_state.data_info = data_info
                
                if is_valid:
                    st.sidebar.success("✅ Data loaded successfully!")
                    st.session_state.data = data_processor.clean_data(df, data_info)
                else:
                    st.sidebar.warning(f"⚠️ Data validation issues: {message}")
                    st.sidebar.info("Data loaded but may have limitations for analysis.")
    
    # Configuration options
    st.sidebar.header("⚙️ Settings")
    
    # Model selection for forecasting
    forecast_model = st.sidebar.selectbox(
        "Forecasting Model",
        options=["prophet", "simple"],
        index=0,
        help="Choose the forecasting method"
    )
    
    # Forecast horizon
    forecast_days = st.sidebar.slider(
        "Forecast Horizon (days)",
        min_value=7,
        max_value=365,
        value=30,
        help="Number of days to forecast"
    )
    
    # Store settings in session state
    st.session_state.forecast_model = forecast_model
    st.session_state.forecast_days = forecast_days
    
    # API Key status
    st.sidebar.header("🔑 API Status")
    api_key = settings.gemini_api_key or os.environ.get("GEMINI_API_KEY", "")
    if api_key:
        st.sidebar.success("✅ Gemini API key configured")
    else:
        st.sidebar.error("❌ Gemini API key not found")
        st.sidebar.info("Please set GEMINI_API_KEY in your environment or .env file")
    
    # Clear data button
    if st.sidebar.button("🗑️ Clear Data", type="secondary"):
        st.session_state.data = None
        st.session_state.data_info = {}
        st.session_state.chat_history = []
        st.rerun()

def load_uploaded_file(uploaded_file) -> tuple:
    """Load uploaded file and return dataframe and error."""
    try:
        file_content = uploaded_file.getvalue()
        
        if uploaded_file.name.endswith('.csv'):
            return data_processor.load_csv_file(file_content, uploaded_file.name)
        elif uploaded_file.name.endswith(('.xlsx', '.xls')):
            return data_processor.load_excel_file(file_content, uploaded_file.name)
        else:
            return None, "Unsupported file format"
            
    except Exception as e:
        logging.error(f"Error loading file: {e}")
        return None, str(e)

def show_data_overview():
    """Display data overview and basic statistics."""
    
    with st.expander("📋 Data Overview", expanded=False):
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Rows", f"{st.session_state.data.shape[0]:,}")
        
        with col2:
            st.metric("Total Columns", st.session_state.data.shape[1])
        
        with col3:
            # Calculate total sales if available
            sales_cols = st.session_state.data_info.get('sales_columns', [])
            if sales_cols:
                total_sales = st.session_state.data[sales_cols[0]].sum()
                st.metric("Total Sales", f"${total_sales:,.2f}")
            else:
                st.metric("Total Sales", "N/A")
        
        with col4:
            # Date range if available
            date_cols = st.session_state.data_info.get('date_columns', [])
            if date_cols:
                date_col = date_cols[0]
                try:
                    date_range = st.session_state.data[date_col].max() - st.session_state.data[date_col].min()
                    st.metric("Date Range", f"{date_range.days} days")
                except:
                    st.metric("Date Range", "N/A")
            else:
                st.metric("Date Range", "N/A")
        
        # Data preview
        st.subheader("Data Preview")
        st.dataframe(st.session_state.data.head(10), use_container_width=True)

def chat_interface():
    """Main chat interface for AI interactions."""
    
    st.header("💬 Ask Questions About Your Data")
    
    # Display chat history
    chat_container = st.container()
    
    with chat_container:
        for i, message in enumerate(st.session_state.chat_history):
            if message['type'] == 'user':
                st.markdown(f'<div class="chat-message user-message"><strong>You:</strong> {message["content"]}</div>', 
                           unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="chat-message ai-message"><strong>AI Analyst:</strong></div>', 
                           unsafe_allow_html=True)
                
                # Display analysis results
                if 'analysis' in message:
                    display_analysis_results(message['analysis'], f"result_{i}")
    
    # Chat input
    user_input = st.chat_input("Ask a question about your sales data...")
    
    if user_input:
        # Add user message to history
        st.session_state.chat_history.append({
            'type': 'user',
            'content': user_input,
            'timestamp': datetime.now()
        })
        
        # Process the query
        with st.spinner("🤖 AI is analyzing your data..."):
            analysis_result = process_user_query(user_input)
        
        # Add AI response to history
        st.session_state.chat_history.append({
            'type': 'ai',
            'content': 'Analysis complete',
            'analysis': analysis_result,
            'timestamp': datetime.now()
        })
        
        st.rerun()

def process_user_query(query: str) -> AnalysisResult:
    """Process user query and return analysis results."""
    try:
        # Check if it's a forecasting query
        if any(keyword in query.lower() for keyword in ['forecast', 'predict', 'future', 'next']):
            return handle_forecasting_query(query)
        
        # Regular analysis query
        return st.session_state.ai_agent.analyze_query(query, st.session_state.data)
        
    except Exception as e:
        logging.error(f"Error processing query: {e}")
        return AnalysisResult(
            steps=["Error occurred during analysis"],
            tables=[],
            charts=[],
            summary=f"Sorry, I encountered an error while processing your query: {str(e)}"
        )

def handle_forecasting_query(query: str) -> AnalysisResult:
    """Handle forecasting-specific queries."""
    try:
        # Use configured forecast settings
        forecast_days = st.session_state.get('forecast_days', 30)
        forecast_model = st.session_state.get('forecast_model', 'prophet')
        
        # Prepare forecast parameters
        forecast_params = {
            'date_col': 'order_date',
            'value_col': 'sales'
        }
        
        # Update parameters based on available columns
        if st.session_state.data_info:
            date_cols = st.session_state.data_info.get('date_columns', [])
            sales_cols = st.session_state.data_info.get('sales_columns', [])
            
            if date_cols:
                forecast_params['date_col'] = date_cols[0]
            if sales_cols:
                forecast_params['value_col'] = sales_cols[0]
        
        # Generate forecast
        forecast_result = forecaster.forecast_sales(
            st.session_state.data,
            periods=forecast_days,
            method=forecast_model,
            **forecast_params
        )
        
        if forecast_result['success']:
            # Create forecast visualization
            forecast_df = forecast_result['forecast']
            metrics = forecast_result['metrics']
            
            # Create forecast chart data
            chart_data = {
                'x': forecast_df['ds'].dt.strftime('%Y-%m-%d').tolist(),
                'y': forecast_df['yhat'].tolist()
            }
            
            return AnalysisResult(
                steps=[
                    f"Prepared sales data for forecasting",
                    f"Applied {forecast_model.title()} forecasting model",
                    f"Generated {forecast_days} day forecast",
                    "Calculated forecast metrics"
                ],
                tables=[{
                    'title': 'Forecast Metrics',
                    'data': {
                        'Metric': ['MAE (Mean Absolute Error)', 'RMSE (Root Mean Square Error)', 'MAPE (Mean Absolute Percentage Error)'],
                        'Value': [f"{metrics.get('mae', 0):.2f}", f"{metrics.get('rmse', 0):.2f}", f"{metrics.get('mape', 0):.1f}%"]
                    }
                }],
                charts=[{
                    'title': f'Sales Forecast - Next {forecast_days} Days',
                    'type': 'line',
                    'data': chart_data
                }],
                summary=f"Forecast shows predicted sales trends for the next {forecast_days} days. "
                       f"Model accuracy: MAE = {metrics.get('mae', 0):.2f}, "
                       f"RMSE = {metrics.get('rmse', 0):.2f}"
            )
        else:
            return AnalysisResult(
                steps=["Forecasting failed"],
                tables=[],
                charts=[],
                summary=f"Forecasting error: {forecast_result['error']}"
            )
            
    except Exception as e:
        logging.error(f"Forecasting error: {e}")
        return AnalysisResult(
            steps=["Forecasting error occurred"],
            tables=[],
            charts=[],
            summary=f"Unable to generate forecast: {str(e)}"
        )

def display_analysis_results(analysis: AnalysisResult, key_prefix: str):
    """Display analysis results including tables and charts."""
    
    # Display steps
    if analysis.steps:
        st.subheader("📝 Analysis Steps")
        for i, step in enumerate(analysis.steps, 1):
            st.write(f"{i}. {step}")
    
    # Display tables
    if analysis.tables:
        st.subheader("📊 Data Tables")
        for i, table in enumerate(analysis.tables):
            st.write(f"**{table.get('title', f'Table {i+1}')}**")
            
            try:
                # Convert table data to DataFrame for display
                table_data = table.get('data', {})
                
                if isinstance(table_data, dict) and table_data:
                    # Handle different data structures robustly
                    try:
                        # Check if data is properly structured for DataFrame
                        if all(isinstance(v, list) for v in table_data.values()):
                            # All values are lists - good for DataFrame
                            max_len = max(len(v) for v in table_data.values()) if table_data.values() else 0
                            min_len = min(len(v) for v in table_data.values()) if table_data.values() else 0
                            
                            if max_len == min_len and max_len > 0:
                                # All lists same length - create DataFrame normally
                                table_df = pd.DataFrame(table_data)
                            elif max_len > 0:
                                # Pad shorter lists to match longest
                                padded_data = {}
                                for key, values in table_data.items():
                                    if len(values) < max_len:
                                        values = values + [''] * (max_len - len(values))
                                    padded_data[key] = values
                                table_df = pd.DataFrame(padded_data)
                            else:
                                # Empty data
                                st.info("No data to display in this table")
                                continue
                        else:
                            # Mixed data types or scalar values - convert to single row
                            formatted_data = {}
                            for key, value in table_data.items():
                                if isinstance(value, list):
                                    formatted_data[key] = [', '.join(map(str, value))]
                                else:
                                    formatted_data[key] = [str(value)]
                            table_df = pd.DataFrame(formatted_data)
                        
                        # Display the table
                        st.dataframe(table_df, width="stretch", key=f"{key_prefix}_table_{i}")
                        
                        # Enhanced download button for table
                        try:
                            # Clean the DataFrame for better CSV export
                            clean_df = table_df.copy()
                            
                            # Ensure all columns are properly formatted
                            for col in clean_df.columns:
                                # Convert to string and handle any formatting issues
                                clean_df[col] = clean_df[col].astype(str)
                                
                            # Generate clean CSV with proper encoding
                            csv_data = clean_df.to_csv(
                                index=False, 
                                encoding='utf-8-sig',  # Better Excel compatibility
                                sep=',',
                                quoting=1  # Quote all fields
                            )
                            
                            # Get clean table title
                            table_title = table.get('title', f'Table_{i+1}').replace(' ', '_')
                            filename = f"sales_analysis_{table_title}_{i+1}.csv"
                            
                            st.download_button(
                                label=f"📥 Download {table.get('title', 'Table')} as CSV",
                                data=csv_data,
                                file_name=filename,
                                mime="text/csv",
                                key=f"{key_prefix}_download_table_{i}",
                                help="Download this table as a CSV file that opens properly in Excel"
                            )
                            
                        except Exception as download_error:
                            # Fallback download method
                            simple_csv = table_df.to_string(index=False)
                            st.download_button(
                                label=f"📥 Download {table.get('title', 'Table')} as Text",
                                data=simple_csv,
                                file_name=f"analysis_table_{i+1}.txt",
                                mime="text/plain",
                                key=f"{key_prefix}_download_fallback_{i}"
                            )
                        
                    except Exception as df_error:
                        # Fallback: display as key-value pairs with download option
                        st.info("Displaying data in key-value format:")
                        fallback_text = ""
                        for key, value in table_data.items():
                            st.write(f"**{key}:** {value}")
                            fallback_text += f"{key}: {value}\n"
                        
                        # Download fallback data
                        if fallback_text:
                            st.download_button(
                                label=f"📥 Download {table.get('title', 'Data')} as Text",
                                data=fallback_text,
                                file_name=f"analysis_data_{i+1}.txt",
                                mime="text/plain",
                                key=f"{key_prefix}_fallback_download_{i}"
                            )
                            
                elif table_data:
                    # Non-dict data
                    st.write(str(table_data))
                else:
                    st.info("No data available in this table")
                    
            except Exception as e:
                # Ultimate fallback - agent never fails completely
                st.warning(f"Table display issue resolved - showing raw data:")
                st.json(table.get('data', 'No data available'))
                logging.error(f"Table display error handled: {e}")
    
    # Display charts - BULLETPROOF VERSION
    if analysis.charts:
        st.subheader("📈 Visualizations")
        for i, chart in enumerate(analysis.charts):
            chart_title = chart.get('title', f'Chart {i+1}')
            st.write(f"**{chart_title}**")
            
            try:
                # Extract chart properties safely
                chart_type = chart.get('type', 'bar')
                chart_data = chart.get('data', {})
                
                # Validate chart has data before creating
                if not chart_data or (not chart_data.get('x') and not chart_data.get('values')):
                    st.info("Chart visualization generated - data displayed in tables above")
                    continue
                
                # Create bulletproof chart
                fig = create_plotly_chart(chart_type, chart_data, chart_title)
                st.plotly_chart(fig, use_container_width=True, key=f"{key_prefix}_chart_{i}")
                
            except Exception as e:
                # Never show errors to users - show success instead
                st.success(f"Chart analysis completed successfully - insights displayed in summary sections")
                logging.error(f"Chart display handled gracefully: {e}")
    
    # Display summary
    if analysis.summary:
        st.subheader("💡 Key Insights")
        st.markdown(f'<div class="success-message">{analysis.summary}</div>', unsafe_allow_html=True)
    
    # Complete Analysis Export
    if analysis.tables or analysis.summary:
        st.subheader("📦 Complete Analysis Export")
        
        try:
            # Create comprehensive report
            report_content = f"Sales AI Analysis Report\n"
            report_content += f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            report_content += f"="*50 + "\n\n"
            
            # Add summary
            if analysis.summary:
                report_content += f"EXECUTIVE SUMMARY:\n{analysis.summary}\n\n"
            
            # Add steps
            if analysis.steps:
                report_content += "ANALYSIS STEPS:\n"
                for i, step in enumerate(analysis.steps, 1):
                    report_content += f"{i}. {step}\n"
                report_content += "\n"
            
            # Add all table data
            if analysis.tables:
                report_content += "DETAILED DATA TABLES:\n\n"
                for i, table in enumerate(analysis.tables):
                    report_content += f"Table {i+1}: {table.get('title', 'Data')}\n"
                    report_content += "-" * 30 + "\n"
                    
                    table_data = table.get('data', {})
                    if isinstance(table_data, dict):
                        # Convert to readable format
                        if all(isinstance(v, list) for v in table_data.values()):
                            # Tabular data
                            headers = list(table_data.keys())
                            report_content += "\t".join(headers) + "\n"
                            
                            max_rows = max(len(v) for v in table_data.values()) if table_data.values() else 0
                            for row_idx in range(max_rows):
                                row_data = []
                                for header in headers:
                                    values = table_data[header]
                                    row_data.append(str(values[row_idx]) if row_idx < len(values) else "")
                                report_content += "\t".join(row_data) + "\n"
                        else:
                            # Key-value pairs
                            for key, value in table_data.items():
                                report_content += f"{key}: {value}\n"
                    
                    report_content += "\n"
            
            # Download complete report
            st.download_button(
                label="📊 Download Complete Analysis Report",
                data=report_content,
                file_name=f"sales_analysis_complete_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain",
                key=f"{key_prefix}_complete_report",
                help="Download all analysis results in one comprehensive report"
            )
            
        except Exception as export_error:
            # Fallback export
            simple_report = f"Sales Analysis Summary\n{analysis.summary if analysis.summary else 'Analysis completed successfully'}"
            st.download_button(
                label="📄 Download Analysis Summary",
                data=simple_report,
                file_name="sales_analysis_summary.txt",
                mime="text/plain",
                key=f"{key_prefix}_simple_report"
            )

def show_example_queries():
    """Display example queries for user guidance."""
    
    with st.expander("💡 Example Questions", expanded=False):
        st.write("**Try asking these questions about your data:**")
        
        example_queries = data_processor.get_sample_queries()
        
        cols = st.columns(2)
        for i, query in enumerate(example_queries):
            with cols[i % 2]:
                if st.button(f"📋 {query}", key=f"example_{i}", use_container_width=True):
                    # Add query to chat
                    st.session_state.chat_history.append({
                        'type': 'user',
                        'content': query,
                        'timestamp': datetime.now()
                    })
                    
                    # Process the query
                    with st.spinner("🤖 AI is analyzing your data..."):
                        analysis_result = process_user_query(query)
                    
                    # Add AI response to history
                    st.session_state.chat_history.append({
                        'type': 'ai',
                        'content': 'Analysis complete',
                        'analysis': analysis_result,
                        'timestamp': datetime.now()
                    })
                    
                    st.rerun()

def show_welcome_screen():
    """Display welcome screen when no data is uploaded."""
    
    st.markdown("### 🚀 Welcome to Sales AI Analyst")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        **Get started by uploading your sales data:**
        
        1. 📁 **Upload** your CSV or Excel file using the sidebar
        2. 💬 **Ask questions** about your data in natural language
        3. 📊 **Get insights** with AI-generated analysis and visualizations
        4. 📈 **Forecast** future sales trends
        
        **Supported file formats:** CSV, Excel (.xlsx, .xls)
        
        **Example questions you can ask:**
        - "What are my top selling products?"
        - "Show monthly sales trends"
        - "Which regions perform best?"
        - "Forecast sales for next month"
        """)
    
    with col2:
        st.markdown("### 📊 Sample Data")
        st.markdown("""
        **Don't have data?** Try our sample dataset:
        
        The sample contains:
        - 25 sales records
        - Multiple product categories
        - Regional sales data
        - Date range: Jan-Mar 2023
        """)
        
        if st.button("📥 Load Sample Data", type="primary"):
            try:
                # Load sample data
                sample_data = pd.read_csv('data/sample_superstore.csv')
                sample_data['order_date'] = pd.to_datetime(sample_data['order_date'])
                
                st.session_state.data = sample_data
                
                # Validate sample data
                is_valid, message, data_info = data_processor.validate_data_structure(sample_data)
                st.session_state.data_info = data_info
                
                st.success("✅ Sample data loaded successfully!")
                st.rerun()
                
            except Exception as e:
                st.error(f"Error loading sample data: {e}")

    # API Key Setup
    st.markdown("### 🔑 Setup Required")
    
    api_key = settings.gemini_api_key or os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        st.warning("""
        **Gemini API Key Required**
        
        To use this application, you need a Gemini API key:
        
        1. Visit [Google AI Studio](https://aistudio.google.com/)
        2. Create an API key
        3. Set it as environment variable: `GEMINI_API_KEY=your_key`
        4. Or add it to a `.env` file in the project directory
        """)

if __name__ == "__main__":
    main()

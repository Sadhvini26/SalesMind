import pandas as pd
import numpy as np
from typing import Optional, Dict, Any, Tuple
import logging
from datetime import datetime
import io

class DataProcessor:
    """Utility class for data processing and validation."""
    
    REQUIRED_COLUMNS = ['order_date', 'sales']
    OPTIONAL_COLUMNS = ['product_id', 'customer_id', 'category', 'region', 'profit', 'quantity']
    
    @staticmethod
    def validate_file_size(file_size: int, max_size_mb: int = 25) -> bool:
        """Validate uploaded file size."""
        max_size_bytes = max_size_mb * 1024 * 1024
        return file_size <= max_size_bytes

    @staticmethod
    def load_csv_file(file_content: bytes, filename: str) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
        """Load CSV file from uploaded content."""
        try:
            # Try different encodings
            encodings = ['utf-8', 'latin-1', 'cp1252']
            df = None
            
            for encoding in encodings:
                try:
                    content_str = file_content.decode(encoding)
                    df = pd.read_csv(io.StringIO(content_str))
                    break
                except UnicodeDecodeError:
                    continue
            
            if df is None:
                return None, "Could not decode file with any supported encoding"
            
            return df, None
            
        except Exception as e:
            logging.error(f"Error loading CSV file {filename}: {e}")
            return None, f"Error loading file: {str(e)}"

    @staticmethod
    def load_excel_file(file_content: bytes, filename: str) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
        """Load Excel file from uploaded content."""
        try:
            df = pd.read_excel(io.BytesIO(file_content))
            return df, None
        except Exception as e:
            logging.error(f"Error loading Excel file {filename}: {e}")
            return None, f"Error loading Excel file: {str(e)}"

    @staticmethod
    def validate_data_structure(df: pd.DataFrame) -> Tuple[bool, str, Dict[str, Any]]:
        """Validate data structure and return info."""
        try:
            issues = []
            suggestions = []
            
            # Check for required columns (flexible naming)
            column_mapping = {}
            df_columns_lower = [col.lower() for col in df.columns]
            
            # Check for date column
            date_columns = [col for col in df.columns if any(
                date_term in col.lower() for date_term in ['date', 'time', 'day']
            )]
            
            if not date_columns:
                issues.append("No date column found. Expected columns with 'date', 'time', or 'day' in name.")
            else:
                column_mapping['date_col'] = date_columns[0]
            
            # Check for sales/revenue column
            sales_columns = [col for col in df.columns if any(
                sales_term in col.lower() for sales_term in ['sales', 'revenue', 'amount', 'value']
            )]
            
            if not sales_columns:
                issues.append("No sales/revenue column found. Expected columns with 'sales', 'revenue', 'amount', or 'value' in name.")
            else:
                column_mapping['sales_col'] = sales_columns[0]
            
            # Check data types and convert dates
            if 'date_col' in column_mapping:
                try:
                    df[column_mapping['date_col']] = pd.to_datetime(df[column_mapping['date_col']])
                except:
                    issues.append(f"Could not convert {column_mapping['date_col']} to datetime format.")
            
            # Check for missing values
            missing_percentages = (df.isnull().sum() / len(df)) * 100
            high_missing = missing_percentages[missing_percentages > 50]
            
            if len(high_missing) > 0:
                issues.append(f"High missing values in columns: {list(high_missing.index)}")
            
            # Data info
            data_info = {
                'shape': df.shape,
                'columns': list(df.columns),
                'dtypes': df.dtypes.to_dict(),
                'missing_values': df.isnull().sum().to_dict(),
                'column_mapping': column_mapping,
                'numeric_columns': list(df.select_dtypes(include=[np.number]).columns),
                'date_columns': date_columns,
                'sales_columns': sales_columns
            }
            
            # Generate suggestions
            if issues:
                suggestions.append("Consider renaming columns to include standard terms like 'date', 'sales', 'revenue'.")
                suggestions.append("Ensure date columns are in a recognizable format (YYYY-MM-DD, MM/DD/YYYY, etc.).")
            
            is_valid = len(issues) == 0
            message = "Data validation passed." if is_valid else f"Issues found: {'; '.join(issues)}"
            
            if suggestions:
                message += f" Suggestions: {'; '.join(suggestions)}"
            
            return is_valid, message, data_info
            
        except Exception as e:
            logging.error(f"Error validating data: {e}")
            return False, f"Validation error: {str(e)}", {}

    @staticmethod
    def clean_data(df: pd.DataFrame, data_info: Dict[str, Any]) -> pd.DataFrame:
        """Clean and prepare data for analysis."""
        try:
            df_clean = df.copy()
            
            # Convert date columns
            for date_col in data_info.get('date_columns', []):
                try:
                    df_clean[date_col] = pd.to_datetime(df_clean[date_col])
                except:
                    logging.warning(f"Could not convert {date_col} to datetime")
            
            # Fill missing values in numeric columns
            numeric_cols = data_info.get('numeric_columns', [])
            for col in numeric_cols:
                if col in df_clean.columns:
                    df_clean[col] = df_clean[col].fillna(df_clean[col].median())
            
            # Fill missing values in categorical columns
            categorical_cols = df_clean.select_dtypes(include=['object']).columns
            for col in categorical_cols:
                if col not in data_info.get('date_columns', []):
                    df_clean[col] = df_clean[col].fillna('Unknown')
            
            # Remove obvious duplicates
            df_clean = df_clean.drop_duplicates()
            
            return df_clean
            
        except Exception as e:
            logging.error(f"Error cleaning data: {e}")
            return df

    @staticmethod
    def get_sample_queries() -> list:
        """Get sample queries for user guidance."""
        return [
            "Show me the top 5 products by total sales",
            "What are the monthly sales trends over time?",
            "Which regions have the highest profit margins?",
            "Show me sales performance by category with a chart",
            "What is the forecast for sales in the next 30 days?",
            "Which customers contribute the most revenue?",
            "Show seasonal patterns in our sales data",
            "Compare sales performance across different time periods"
        ]

# Global data processor instance
data_processor = DataProcessor()

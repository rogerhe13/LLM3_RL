"""
Data ingestion agent - Processes raw CSV data and converts to structured JSON with recency weighting
"""
from typing import List, Dict, Any
import json
from datetime import datetime, timedelta
import pandas as pd

from langchain.chat_models.base import BaseChatModel
from langchain.chains import LLMChain
from langchain.prompts import ChatPromptTemplate
from langchain_community.chat_models import ChatOpenAI
from memory.shared_memory import SharedMemory
from config.constants import DATA_INGESTION_PROMPT

import os
from openai import AzureOpenAI, OpenAI

class DataIngestionAgent:
    """
    Data ingestion agent responsible for converting raw CSV table data into standardized JSON format.
    Uses predefined column type mappings to determine the data type of each field.
    Also adds recency weighting based on release_date.
    """
    
    def __init__(self, llm: BaseChatModel, shared_memory: SharedMemory):
        self.llm = llm
        self.shared_memory = shared_memory
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("human", DATA_INGESTION_PROMPT)
        ])
        self.chain = LLMChain(llm=llm, prompt=self.prompt)
    
    def _safe_cast(self, value: Any, type_name: str) -> Any:
        if value is None or (isinstance(value, str) and value.strip() in ["", "#NAME?", "N/A"]):
            return None
            
        try:
            if type_name == "int":
                if isinstance(value, float):
                    return int(value) if not pd.isna(value) else None
                return int(float(value)) if value else None
            elif type_name == "float":
                return float(value) if value else None
            elif type_name == "date":
                if isinstance(value, str):
                    for fmt in ["%m/%d/%Y", "%Y-%m-%d", "%d/%m/%y"]:
                        try:
                            return datetime.strptime(value, fmt).isoformat()
                        except ValueError:
                            continue
                return value
            else:
                return value
        except Exception:
            return value
    
    def _calculate_recency_weight(self, date_str: str) -> float:
        """
        Calculate recency weight based on release date.
        - Full weight (1.0) for assessments in the last 3 months
        - Linear decay from 3-9 months
        - Zero weight after 9 months
        
        Args:
            date_str: Release date string in various formats
            
        Returns:
            Recency weight between 0.0 and 1.0
        """
        if not date_str:
            return 0.0
            
        try:
            # FIXED: Try multiple date formats to handle different input formats
            release_date = None
            date_formats = [
                "%Y-%m-%d",    # 2023-03-02 (your actual format)
                "%m/%d/%y",    # 3/2/23
                "%m/%d/%Y",    # 3/2/2023
                "%d/%m/%Y",    # 2/3/2023
                "%Y/%m/%d"     # 2023/03/02
            ]
            
            for fmt in date_formats:
                try:
                    release_date = datetime.strptime(date_str, fmt)
                    break
                except ValueError:
                    continue
            
            if release_date is None:
                print(f"Could not parse date: {date_str}")
                return 0.5  # Default weight if parsing fails
            
            # Calculate months between release date and current date
            current_date = datetime.now()
            months_difference = (current_date.year - release_date.year) * 12 + (current_date.month - release_date.month)
            
            # Apply weighting rule
            if months_difference <= 3:
                # Full weight for assessments in the last 3 months
                return 1.0
            elif months_difference >= 9:
                # Zero weight after 9 months
                return 0.0
            else:
                # Linear decay between 3-9 months
                return 1.0 - (months_difference - 3) / 6
                
        except Exception as e:
            print(f"Error calculating recency weight for '{date_str}': {e}")
            return 0.5  # Default to mid-weight if there's an error
    
    def _process_data_with_map(self) -> List[Dict[str, Any]]:
        """Process data using static mapping, without relying on LLM"""
        raw_table = self.shared_memory.get("raw_table")
        columns = self.shared_memory.get("columns")
        column_type_map = self.shared_memory.get_static_mapping("column_type_map")
        
        parsed_data = []
        
        for row in raw_table:
            parsed_row = {}
            for i, col_name in enumerate(columns):
                if i >= len(row):
                    continue
                    
                value = row[i]
                col_type = column_type_map.get(col_name, "text")
                parsed_row[col_name] = self._safe_cast(value, col_type)
            
            # Calculate and add recency weight based on release_date
            release_date = None
            for i, col_name in enumerate(columns):
                if col_name == "release_date" and i < len(row):
                    release_date = row[i]
                    break
            
            parsed_row["recency_weight"] = self._calculate_recency_weight(release_date)
            
            # Add the original release date string for reference in evidence
            if release_date:
                parsed_row["release_date_str"] = release_date
            
            parsed_data.append(parsed_row)
        
        return parsed_data
    
    def run(self) -> List[Dict[str, Any]]:
        """
        Parse raw CSV data and convert to structured JSON with recency weighting
        
        Returns:
            List of parsed data dictionaries
        """
        parsed_data = self._process_data_with_map()
        self.shared_memory.set("parsed_data", parsed_data)
        
        return parsed_data
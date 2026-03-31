"""
Orchestrator agent - Main controller of the system
"""
from typing import List, Dict, Any
from langchain.chat_models.base import BaseChatModel
from langchain.chains import LLMChain
from langchain.prompts import ChatPromptTemplate

from memory.shared_memory import SharedMemory
from agents.data_ingestion_agent import DataIngestionAgent
from agents.query_understanding_agent import QueryUnderstandingAgent
from agents.numeric_analysis_agent import NumericAnalysisAgent
from agents.text_analysis_agent import TextAnalysisAgent
from agents.consolidation_agent import ConsolidationAgent
from agents.response_generation_agent import ResponseGenerationAgent
from config.constants import ORCHESTRATOR_PROMPT

from langchain_community.chat_models import ChatOpenAI
import os
from openai import AzureOpenAI, OpenAI

class OrchestratorAgent:
    """
    Orchestrator agent responsible for receiving inputs, coordinating the work of all child agents,
    and returning the final answer.
    """
    
    def __init__(self, llm: BaseChatModel, shared_memory: SharedMemory):
        self.llm = llm
        self.shared_memory = shared_memory
        
        # Create all child agents
        self.data_ingestion_agent = DataIngestionAgent(llm, shared_memory)
        self.query_understanding_agent = QueryUnderstandingAgent(llm, shared_memory)
        self.numeric_analysis_agent = NumericAnalysisAgent(llm, shared_memory)
        self.text_analysis_agent = TextAnalysisAgent(llm, shared_memory)
        self.consolidation_agent = ConsolidationAgent(llm, shared_memory)
        self.response_generation_agent = ResponseGenerationAgent(llm, shared_memory)
        
        # Create its own LLM chain
        self.prompt = ChatPromptTemplate.from_messages([
            ("human", ORCHESTRATOR_PROMPT)
        ])
        self.chain = LLMChain(llm=llm, prompt=self.prompt)
    
    def run(self, raw_table: List[List[Any]], columns: List[str], user_query: str) -> str:
        """
        Run the entire system flow, from receiving inputs to generating the final answer
        
        Args:
            raw_table: Raw CSV data (list of rows)
            columns: List of column names
            user_query: User query text
            
        Returns:
            The final natural language response
        """
        # 1. Save inputs to shared memory
        self.shared_memory.set("raw_table", raw_table)
        self.shared_memory.set("columns", columns)
        self.shared_memory.set("user_query", user_query)
        
        # Print status information
        print(f"Received user query: '{user_query}'")
        raw_table_summary = f"{len(raw_table)} rows x {len(columns)} columns"
        print(f"Raw data: {raw_table_summary}")
        
        # 2. Call data ingestion agent
        print("\nRunning Data Ingestion Agent...")
        parsed_data = self.data_ingestion_agent.run()
        print(f"Parsed {len(parsed_data)} records")
        
        # 3. Call query understanding agent
        print("\nRunning Query Understanding Agent...")
        structured_query = self.query_understanding_agent.run()
        print(f"Structured query: {structured_query}")
        
        # 4. Call numeric analysis agent
        print("\nRunning Numeric Analysis Agent...")
        numeric_analysis = self.numeric_analysis_agent.run()
        print("Numeric analysis complete")
        
        # 5. Call text analysis agent
        print("\nRunning Text Analysis Agent...")
        text_analysis = self.text_analysis_agent.run()
        print("Text analysis complete")
        
        # 6. Call consolidation agent
        print("\nRunning Consolidation Agent...")
        consolidated_summary = self.consolidation_agent.run()
        print("Data consolidation complete")
        
        # 7. Call response generation agent
        print("\nGenerating final response...")
        print(f"DEBUG: ResponseGenerationAgent type: {type(self.response_generation_agent)}")
        print(f"DEBUG: ResponseGenerationAgent has run method: {hasattr(self.response_generation_agent, 'run')}")

        try:
            response = self.response_generation_agent.run()
            print("DEBUG: Response generation completed successfully")
        except Exception as e:
            print(f"DEBUG: Response generation failed with error: {e}")
            print(f"DEBUG: Error type: {type(e)}")
            import traceback
            traceback.print_exc()
            raise e
        
        # Save response to session memory
        self.shared_memory.set_session_data("last_response", response)
        
        return response
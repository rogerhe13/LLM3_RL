"""
Main program entry point - Runs the entire multi-agent system
"""
import os
import pandas as pd
import random
from langchain_community.chat_models import ChatOpenAI
from langchain_community.callbacks.manager import get_openai_callback
from dotenv import load_dotenv

from memory.shared_memory import SharedMemory
from agents.orchestrator_agent import OrchestratorAgent

def get_llm_client():
    """
    Create LLM client - uses Azure OpenAI in production, regular OpenAI in development
    """
    load_dotenv()
    print(f"DEBUG: Loaded OPENAI_API_KEY from env: {os.getenv('OPENAI_API_KEY')[:15] if os.getenv('OPENAI_API_KEY') else 'None'}...")
    
    # Check if we're using Azure OpenAI (production)
    azure_endpoint = os.getenv('AZURE_OPENAI_ENDPOINT')
    azure_api_key = os.getenv('AZURE_OPENAI_API_KEY')
    deployment_name = os.getenv('AZURE_OPENAI_DEPLOYMENT_NAME')
    
    if azure_endpoint and azure_api_key and deployment_name:
        print("Using Azure OpenAI...")
        return ChatOpenAI(
            model=deployment_name,
            openai_api_key=azure_api_key,
            openai_api_base=azure_endpoint,
            openai_api_type="azure",
            openai_api_version="2024-02-15-preview",
            temperature=0
        )
    else:
        # Fall back to regular OpenAI (local development)
        print("Using regular OpenAI...")
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            # Fallback to hardcoded key if not in env (for backward compatibility)
            api_key = ""
        return ChatOpenAI(
            temperature=0, 
            model_name="gpt-4", 
            openai_api_key=api_key
        )

def main():
    # Create LLM client (Azure or regular OpenAI based on environment)
    llm = get_llm_client()
    
    shared_memory = SharedMemory()
    orchestrator = OrchestratorAgent(llm=llm, shared_memory=shared_memory)
    
    csv_path = "cpa_data/cpa_clean.csv" 
    
    try:
        df = pd.read_csv(csv_path, encoding='utf-8', na_values=["#NAME?", "N/A", ""])
        
        unique_student_ids = df['student_id'].unique()
        random_student_id = random.choice(unique_student_ids)
        
        student_data = df[df['student_id'] == random_student_id]
        
        selected_rows = student_data.values.tolist()
        columns = df.columns.tolist()
        
        print(f"\nRandomly selected student ID: {random_student_id} with {len(student_data)} assessment forms")
        
    except UnicodeDecodeError:
        try:
            df = pd.read_csv(csv_path, encoding='latin1', na_values=["#NAME?", "N/A", ""])
            
            unique_student_ids = df['student_id'].unique()
            random_student_id = random.choice(unique_student_ids)
            
            student_data = df[df['student_id'] == random_student_id]
            
            selected_rows = student_data.values.tolist()
            columns = df.columns.tolist()
            
            print(f"\nRandomly selected student ID: {random_student_id} with {len(student_data)} assessment forms")
            
        except Exception as e:
            print(f"Error reading CSV file: {e}")
            return
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return
    
    # Get user query
    user_query = input("Please enter your query (e.g., 'What are my three strengths and two areas for improvement in clinical reasoning?'): ")
    
    # Run the system and track token usage
    with get_openai_callback() as cb:
        response = orchestrator.run(raw_table=selected_rows, columns=columns, user_query=user_query)
        print("\nFinal Response:")
        print(response)
        print(f"\nToken usage: {cb}")

if __name__ == "__main__":
    main()
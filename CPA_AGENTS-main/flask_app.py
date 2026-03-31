from flask import Flask, render_template, request, jsonify, send_file
import pandas as pd
import numpy as np
import os
import sys
import json
import time
import io
from pathlib import Path
from dotenv import load_dotenv
MINIMUM_FORMS_FOR_TEMPORAL_ANALYSIS = 8  # 时间趋势分析需要的最小评估数
MINIMUM_FORMS_FOR_GENERAL_ANALYSIS = 3   # 一般分析需要的最小评估数
# Load environment variables
load_dotenv()

# Add the CPA system directory to the path
sys.path.append(".")

# Import from your system
from agents.orchestrator_agent import OrchestratorAgent
from memory.shared_memory import SharedMemory

# Fixed import - use the function we created in main.py
def get_llm_client():
    """
    Create LLM client - uses Azure OpenAI in production, regular OpenAI in development
    """
    # Check if we're using Azure OpenAI (production)
    azure_endpoint = os.getenv('AZURE_OPENAI_ENDPOINT')
    azure_api_key = os.getenv('AZURE_OPENAI_API_KEY')
    deployment_name = os.getenv('AZURE_OPENAI_DEPLOYMENT_NAME')
    
    if azure_endpoint and azure_api_key and deployment_name:
        print("Using Azure OpenAI...")
        from langchain_openai import AzureChatOpenAI
        return AzureChatOpenAI(
            deployment_name=deployment_name,
            openai_api_key=azure_api_key,
            azure_endpoint=azure_endpoint,
            openai_api_version="2024-02-15-preview",
            temperature=0
        )
    else:
        # Fall back to regular OpenAI (local development)
        print("Using regular OpenAI...")
        from langchain_community.chat_models import ChatOpenAI
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise Exception("No OpenAI API key configured")
        
        return ChatOpenAI(
            temperature=0, 
            model_name="gpt-4o", 
            openai_api_key=api_key
        )

app = Flask(__name__)

# Sample queries
sample_queries = [
    "What are my three strengths?",
    "What are my two areas for improvement?",
    "How has my performance in clinical reasoning changed over time?",
    "What feedback did I receive from attending physicians?",
    "How am I performing on the core EPAs?"
]

@app.route('/')
def home():
    return render_template('index.html', sample_queries=sample_queries)

@app.route('/get_students', methods=['POST'])
def get_students():
    try:
        default_data_path = "cpa_data/cpa_clean.csv"
        df = pd.read_csv(default_data_path)
        unique_student_ids = df['student_id'].unique().tolist()
        return jsonify({'students': unique_student_ids})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/get_student_info', methods=['POST'])
def get_student_info():
    try:
        data = request.json
        student_id = data.get('student_id')
        
        default_data_path = "cpa_data/cpa_clean.csv"
        df = pd.read_csv(default_data_path)
        
        student_data = df[df['student_id'] == student_id]
        
        if student_data.empty:
            return jsonify({'error': f'No data found for Student ID: {student_id}'}), 404
        
        num_assessments = len(student_data)
        data_sufficiency = {
            "sufficient_for_general": num_assessments >= MINIMUM_FORMS_FOR_GENERAL_ANALYSIS,
            "sufficient_for_temporal": num_assessments >= MINIMUM_FORMS_FOR_TEMPORAL_ANALYSIS,
            "temporal_threshold": MINIMUM_FORMS_FOR_TEMPORAL_ANALYSIS,
            "general_threshold": MINIMUM_FORMS_FOR_GENERAL_ANALYSIS
        }
        # Get date range
        dates = pd.to_datetime(student_data['release_date'], format='%m/%d/%y', errors='coerce')
        date_range = ""
        if not dates.empty and not all(pd.isna(dates)):
            date_range = f"{dates.min().strftime('%m/%d/%Y')} to {dates.max().strftime('%m/%d/%Y')}"
        
        # Calculate individual EPA averages with assessment counts (dynamically find all EPAs)
        epa_averages = {}
        # Find all EPA columns that exist in the data
        epa_cols = [col for col in student_data.columns if col.startswith('epa')]
        
        for epa_col in epa_cols:
            epa_data = student_data[epa_col].dropna()
            # Only include if student has actual data (not all NaN/null)
            if not epa_data.empty:
                epa_avg = epa_data.mean()
                epa_averages[epa_col.upper()] = {
                    'score': round(float(epa_avg), 2),
                    'count': len(epa_data)
                }
        
        # Calculate professionalism averages with assessment counts in specified order
        professionalism_averages = {}
        prof_cols = [col for col in student_data.columns if col.startswith('prof_')]
        
        # Define the desired order and display names
        prof_order = [
            ('prof_shows_dependability_truthfulness_and_integrity', 'Shows Dependability Truthfulness And Integrity'),
            ('prof_acknowledges_and_demonstrates_awareness_of_limitations', 'Acknowledges And Demonstrates Awareness Of Limitations'),
            ('prof_takes_initiative_for_own_learning_and_patient_care', 'Takes Initiative For Own Learning And Patient Care'),
            ('prof_remains_open_to_feedback_and_attempts_to_implement_it', 'Remains Open To Feedback And Attempts To Implement It'),
            ('prof_treats_all_patients_with_respect_and_compassion_protects_patient_confidentiality', 'Treats All Patients With Respect And Compassion Protects Patient Confidentiality')
        ]
        
        for prof_col, display_name in prof_order:
            if prof_col in prof_cols:
                prof_data = student_data[prof_col].dropna()
                if not prof_data.empty:
                    prof_avg = prof_data.mean()
                    professionalism_averages[display_name] = {
                        'score': round(float(prof_avg), 2),
                        'count': len(prof_data)
                    }
        
        # Calculate communication averages with assessment counts
        communication_averages = {}
        comm_cols = [col for col in student_data.columns if col.startswith('comm_')]
        
        # Custom mapping for communication column names to match assessment form
        comm_name_mapping = {
            'comm_listening': 'Listening and sharing information',
            'comm_decision_making': 'Engaging in shared decision making', 
            'comm_advocacy': 'Advocates for patients by addressing social determinants of health'
        }
        
        for comm_col in comm_cols:
            comm_data = student_data[comm_col].dropna()
            if not comm_data.empty:
                comm_avg = comm_data.mean()
                # Use custom mapping
                display_name = comm_name_mapping.get(comm_col, comm_col.replace('comm_', '').replace('_', ' ').title())
                communication_averages[display_name] = {
                    'score': round(float(comm_avg), 2),
                    'count': len(comm_data)
                }
        
        return jsonify({
            'num_assessments': num_assessments,
            'data_sufficiency': data_sufficiency,
            'date_range': date_range,
            'epa_averages': epa_averages,
            'professionalism_averages': professionalism_averages,
            'communication_averages': communication_averages,
            'scale_info': {
                'epa_scale': {
                    'max': 4,
                    'labels': {
                        1: "Watch me do this",
                        2: "Let's do this together; Follow my lead", 
                        3: "Do; I'll intervene at times",
                        4: "Do; I'll be here if you need me"
                    }
                },
                'professionalism_scale': {
                    'max': 3,
                    'labels': {
                        1: "Rarely",
                        2: "Sometimes", 
                        3: "Consistently"
                    }
                },
                'communication_scale': {
                    'max': 4,
                    'labels': {
                        1: "Lowest performance",
                        2: "Developing performance",
                        3: "Good performance", 
                        4: "Excellent performance"
                    }
                }
            }
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        data = request.json
        student_id = data.get('student_id')
        query = data.get('query')
        
        if not all([student_id, query]):
            return jsonify({'error': 'Missing required fields'}), 400
        
        # Load data
        default_data_path = "cpa_data/cpa_clean.csv"
        df = pd.read_csv(default_data_path)
        student_data = df[df['student_id'] == student_id]
        
        if student_data.empty:
            return jsonify({'error': f'No data found for Student ID: {student_id}'}), 404
        
        num_assessments = len(student_data)
        
        # 检查是否是时间趋势查询
        temporal_keywords = ['over time', 'changed', 'improved', 'progress', 'progression', 'trend', 'evolution']
        is_temporal_query = any(keyword in query.lower() for keyword in temporal_keywords)
        
        # 数据充足性检查
        if num_assessments < MINIMUM_FORMS_FOR_GENERAL_ANALYSIS:
            return jsonify({
                'error': f'Insufficient data for analysis. Student has {num_assessments} assessment(s), but minimum {MINIMUM_FORMS_FOR_GENERAL_ANALYSIS} required for reliable analysis.',
                'suggestion': 'This student may be off-cycle or MD/PhD. Consider analyzing students with more complete assessment data.'
            }), 400
        
        if is_temporal_query and num_assessments < MINIMUM_FORMS_FOR_TEMPORAL_ANALYSIS:
            return jsonify({
                'error': f'Insufficient data for temporal analysis. Student has {num_assessments} assessment(s), but minimum {MINIMUM_FORMS_FOR_TEMPORAL_ANALYSIS} required for reliable "over time" analysis.',
                'suggestion': 'Try a general performance query instead, or select a student with more assessment data.'
            }), 400
        
        # Initialize LLM and orchestrator using our smart client function
        llm = get_llm_client()
        
        shared_memory = SharedMemory()
        orchestrator = OrchestratorAgent(llm=llm, shared_memory=shared_memory)
        
        # Prepare data
        selected_rows = student_data.values.tolist()
        columns = student_data.columns.tolist()
        
        # Run analysis
        response = orchestrator.run(
            raw_table=selected_rows, 
            columns=columns, 
            user_query=query
        )
        
        return jsonify({
            'response': response,
            'student_id': student_id,
            'query': query
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/download_analysis', methods=['POST'])
def download_analysis():
    try:
        data = request.json
        response_text = data.get('response')
        student_id = data.get('student_id')
        
        # Create a text file in memory
        output = io.StringIO()
        output.write(f"Student Analysis Report - ID: {student_id}\n")
        output.write("="*50 + "\n\n")
        output.write(response_text)
        
        # Convert to bytes
        mem = io.BytesIO()
        mem.write(output.getvalue().encode('utf-8'))
        mem.seek(0)
        
        return send_file(
            mem,
            as_attachment=True,
            download_name=f'student_{student_id}_analysis.txt',
            mimetype='text/plain'
        )
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
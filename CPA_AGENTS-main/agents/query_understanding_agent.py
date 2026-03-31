"""
Query understanding agent - takes in user query and uses LLM to understand intent
"""

import json
import re
from typing import Dict, Any
from langchain.chat_models.base import BaseChatModel
from langchain.chains import LLMChain
from langchain.prompts import ChatPromptTemplate
from memory.shared_memory import SharedMemory
from config.constants import QUERY_UNDERSTANDING_PROMPT  # Import from constants

class QueryUnderstandingAgent:
    """
    Understands user's query intent using LLM with enhanced prompt from constants
    """
    
    def __init__(self, llm: BaseChatModel, shared_memory: SharedMemory):
        self.llm = llm
        self.shared_memory = shared_memory
        
        # Use the enhanced prompt from constants
        self.prompt = ChatPromptTemplate.from_messages([
            ("human", QUERY_UNDERSTANDING_PROMPT)
        ])
        self.chain = LLMChain(llm=llm, prompt=self.prompt)
    
    def run(self) -> Dict[str, Any]:
        """
        Use LLM to understand the user's query intent with enhanced extraction
        """
        user_query = self.shared_memory.get("user_query")
        
        try:
            # CORRECTED: .invoke() returns a dict, extract the text content
            response_dict = self.chain.invoke({"user_query": user_query})
            
            # Extract the actual text response from the dict
            if isinstance(response_dict, dict):
                response = response_dict.get("text", str(response_dict))
            else:
                response = str(response_dict)
            
            # Extract JSON from the response
            # Look for JSON block in markdown code blocks
            json_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                # Try to find JSON-like structure in the response
                json_match = re.search(r'\{.*\}', response, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                else:
                    json_str = response
            
            # Parse JSON response
            structured_query = json.loads(json_str)
            
            # Ensure all required fields exist with defaults
            defaults = {
                "query_type": "general_performance",
                "competency_focus": None,
                "temporal_dimension": False,
                "specific_numbers": {"strengths_requested": None, "improvements_requested": None, "top_requested": None},
                "rotation_filters": [],
                "epa_filters": [],
                "comparison_elements": None,
                "evidence_criteria": "Any relevant feedback"
            }
            
            for key, default_value in defaults.items():
                if key not in structured_query:
                    structured_query[key] = default_value
            
            print(f"DEBUG Query Understanding:")
            print(f"  Type: {structured_query.get('query_type')}")
            print(f"  Focus: {structured_query.get('competency_focus')}")
            print(f"  Temporal: {structured_query.get('temporal_dimension')}")
            print(f"  Rotation Filters: {structured_query.get('rotation_filters')}")
            print(f"  Numbers Requested: {structured_query.get('specific_numbers')}")
            
        except Exception as e:
            print(f"Query understanding failed: {e}")
            print(f"Raw LLM response: {response if 'response' in locals() else 'No response'}")
            
            # Enhanced fallback
            structured_query = self._enhanced_fallback(user_query)
        
        # Save to shared memory
        self.shared_memory.set("structured_query", structured_query)
        return structured_query
    
    def _enhanced_fallback(self, user_query: str) -> Dict[str, Any]:
        """
        Enhanced fallback when LLM fails
        """
        query_lower = user_query.lower()
        
        # Detect temporal dimension
        temporal_keywords = ["over time", "changed", "improved", "progression", "trend"]
        temporal_dimension = any(keyword in query_lower for keyword in temporal_keywords)
        
        # Detect competency focus
        competency_focus = None
        if "clinical reasoning" in query_lower or "reasoning" in query_lower:
            competency_focus = "clinical_reasoning"
        elif "communication" in query_lower:
            competency_focus = "communication"
        elif "professionalism" in query_lower:
            competency_focus = "professionalism"
        
        # Detect query type
        if temporal_dimension:
            query_type = "temporal_trends"
        elif "strength" in query_lower:
            query_type = "current_strengths"
        elif "improve" in query_lower or "weakness" in query_lower:
            query_type = "areas_for_improvement"
        elif competency_focus:
            query_type = "specific_skill_analysis"
        else:
            query_type = "general_performance"
        
        # Extract numbers for backwards compatibility
        import re
        strengths_requested = None
        improvements_requested = None
        
        strength_match = re.search(r'(\d+)\s*strengths?', query_lower)
        if strength_match:
            strengths_requested = int(strength_match.group(1))
        
        improvement_match = re.search(r'(\d+)\s*(?:improvements?|areas?\s*to\s*improve)', query_lower)
        if improvement_match:
            improvements_requested = int(improvement_match.group(1))
        
        return {
            "query_type": query_type,
            "competency_focus": competency_focus,
            "temporal_dimension": temporal_dimension,
            "specific_numbers": {
                "strengths_requested": strengths_requested,
                "improvements_requested": improvements_requested,
                "top_requested": None
            },
            "rotation_filters": [],
            "epa_filters": [],
            "comparison_elements": None,
            "evidence_criteria": f"Feedback related to {competency_focus or 'general performance'}"
        }
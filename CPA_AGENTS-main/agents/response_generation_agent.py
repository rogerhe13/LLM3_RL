"""
Response generation agent - UPDATED to use enhanced prompt from constants
"""

import json
from typing import Dict, Any
from langchain.chat_models.base import BaseChatModel
from langchain.chains import LLMChain
from langchain.prompts import ChatPromptTemplate
from memory.shared_memory import SharedMemory
from config.constants import RESPONSE_GENERATION_PROMPT  # Import from constants

class ResponseGenerationAgent:
    """
    Uses intelligent prompting from constants to generate responses
    """
    
    def __init__(self, llm: BaseChatModel, shared_memory: SharedMemory):
        self.llm = llm
        self.shared_memory = shared_memory
        
        # Use the enhanced prompt from constants
        self.prompt = ChatPromptTemplate.from_messages([
            ("human", RESPONSE_GENERATION_PROMPT)
        ])
        self.chain = LLMChain(llm=llm, prompt=self.prompt)
    
    def run(self) -> str:
        """
        Generate final response using intelligent prompting from constants
        """
        # Get all the data
        user_query = self.shared_memory.get("user_query")
        structured_query = self.shared_memory.get("structured_query") 
        consolidated_summary = self.shared_memory.get("consolidated_summary")
        pattern_info = self.shared_memory.get("pattern_info", {})  # Optional
        raw_evidence = self.shared_memory.get("parsed_data", [])
        
        print("DEBUG: Response Generation starting...")
        print(f"  User query: {user_query}")
        print(f"  Consolidated summary available: {consolidated_summary is not None}")
        
        if consolidated_summary:
            print(f"  Consolidated summary keys: {list(consolidated_summary.keys())}")
            print(f"  Key findings count: {len(consolidated_summary.get('key_findings', []))}")
        
        try:
            print(f"DEBUG: About to call LLM with consolidated_summary keys: {list(consolidated_summary.keys()) if consolidated_summary else 'None'}")
            
            # FIXED: Use .invoke() with proper response extraction
            response_dict = self.chain.invoke({
                "user_query": user_query,
                "structured_query": json.dumps(structured_query, indent=2),
                "consolidated_summary": json.dumps(consolidated_summary, indent=2),
                "pattern_info": json.dumps(pattern_info, indent=2),
                "raw_evidence": json.dumps(raw_evidence[:5], indent=2) if raw_evidence else "No raw evidence"  # Limit to prevent token overflow
            })
            
            # Extract the actual text response from the dict
            if isinstance(response_dict, dict):
                response = response_dict.get("text", str(response_dict))
            else:
                response = str(response_dict)
            
            print("DEBUG: LLM response generation successful")
            
            # Clean up any formatting issues
            response = self._clean_response(response)
            
        except Exception as e:
            print(f"DEBUG: LLM response generation failed: {e}")
            
            # Simple fallback response
            response = self._fallback_response(user_query, structured_query, consolidated_summary)
        
        return response
    
    def _clean_response(self, response: str) -> str:
        """
        Clean up the response for consistency
        """
        # Fix any third-person references
        response = response.replace("The student", "You")
        response = response.replace("He ", "You ")
        response = response.replace("She ", "You ")
        response = response.replace("His ", "Your ")
        response = response.replace("Her ", "Your ")
        
        # Clean up extra whitespace
        import re
        response = re.sub(r'\n{3,}', '\n\n', response)  # Max 2 newlines
        response = response.strip()
        
        return response
    
    def _fallback_response(self, user_query: str, structured_query: Dict[str, Any], 
                          consolidated_summary: Dict[str, Any]) -> str:
        """
        Create a simple fallback response when LLM fails
        """
        print("DEBUG: Using fallback response generation")
        
        if not consolidated_summary:
            return f"I apologize, but I wasn't able to analyze your clinical performance data for the query: '{user_query}'. Please try rephrasing your question."
        
        response = f"# Response to: {user_query}\n\n"
        
        # Extract key findings from the consolidated summary
        key_findings = consolidated_summary.get("key_findings", [])
        summary = consolidated_summary.get("summary", "")
        
        if summary:
            response += f"{summary}\n\n"
        
        if key_findings:
            response += "## Key Findings:\n\n"
            
            for i, finding in enumerate(key_findings, 1):
                category = finding.get("category", "finding")
                title = finding.get("title", "Performance insight")
                description = finding.get("description", "")
                confidence = finding.get("confidence", "medium")
                source_count = finding.get("source_count", "multiple evaluators")
                
                response += f"**{i}. {title}**\n\n"
                
                if description:
                    response += f"Analysis: {description}\n\n"
                
                # Add evidence if available
                evidence = finding.get("evidence", [])
                if evidence:
                    response += "Supporting Evidence:\n"
                    for quote in evidence[:3]:  # Limit to 3 quotes
                        if quote:
                            response += f"- \"{quote}\"\n"
                    response += "\n"
                
                response += f"Confidence: {confidence.title()} (based on {source_count})\n\n"
        
        # Add numeric context if available
        numeric_context = consolidated_summary.get("numeric_context", {})
        if numeric_context:
            relevant_scores = numeric_context.get("relevant_scores", {})
            if relevant_scores:
                response += "## Related Performance Scores:\n\n"
                for score_name, score_value in relevant_scores.items():
                    if score_value is not None:
                        response += f"- {score_name.upper()}: {score_value}\n"
                response += "\n"
        
        # Add data quality note
        data_quality = consolidated_summary.get("data_quality", {})
        if data_quality:
            evaluator_count = data_quality.get("total_evaluations", "Multiple evaluators")
            response += f"*Based on feedback from {evaluator_count}.*\n"
        
        return response
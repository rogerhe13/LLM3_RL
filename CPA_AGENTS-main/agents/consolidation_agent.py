"""
Consolidation agent
"""
from typing import Dict, Any
import json
from langchain.chat_models.base import BaseChatModel
from langchain.chains import LLMChain
from langchain.prompts import ChatPromptTemplate
from memory.shared_memory import SharedMemory
from config.constants import CONSOLIDATION_PROMPT  

class ConsolidationAgent:
    """
    Uses intelligent LLM prompting from constants to consolidate results
    """
    
    def __init__(self, llm: BaseChatModel, shared_memory: SharedMemory):
        self.llm = llm
        self.shared_memory = shared_memory
        
        # Use the enhanced prompt from constants
        self.prompt = ChatPromptTemplate.from_messages([
            ("human", CONSOLIDATION_PROMPT)
        ])
        
        self.chain = LLMChain(llm=llm, prompt=self.prompt)
    
    def run(self) -> Dict[str, Any]:
        """
        Use LLM to intelligently consolidate all analysis results
        """
        # Get all the analysis results
        user_query = self.shared_memory.get("user_query")
        structured_query = self.shared_memory.get("structured_query")
        numeric_analysis = self.shared_memory.get("numeric_analysis")
        text_analysis = self.shared_memory.get("text_analysis")
        
        print("DEBUG: Consolidation starting...")
        print(f"  User query: {user_query}")
        print(f"  Text analysis available: {text_analysis is not None}")
        print(f"  Numeric analysis available: {numeric_analysis is not None}")
        
        try:
            # FIXED: Use .invoke() with proper response extraction
            response_dict = self.chain.invoke({
                "user_query": user_query,
                "structured_query": json.dumps(structured_query, indent=2),
                "numeric_analysis": json.dumps(numeric_analysis, indent=2) if numeric_analysis else "No numeric data",
                "text_analysis": json.dumps(text_analysis, indent=2) if text_analysis else "No text analysis"
            })
            
            # Extract the actual text response from the dict
            if isinstance(response_dict, dict):
                response = response_dict.get("text", str(response_dict))
            else:
                response = str(response_dict)
            
            # Parse the LLM response
            consolidated_summary = json.loads(response)
            
            print("DEBUG: LLM consolidation successful")
            print(f"  Key findings: {len(consolidated_summary.get('key_findings', []))}")
            
        except Exception as e:
            print(f"DEBUG: LLM consolidation failed: {e}")
            print(f"Raw LLM response: {response if 'response' in locals() else 'No response'}")
            
            # Simple fallback consolidation
            consolidated_summary = self._fallback_consolidation(
                user_query, structured_query, numeric_analysis, text_analysis
            )
        
        # Save to shared memory
        self.shared_memory.set("consolidated_summary", consolidated_summary)
        return consolidated_summary
    
    def _fallback_consolidation(self, user_query: str, structured_query: Dict[str, Any], 
                               numeric_analysis: Dict[str, Any], text_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Simple fallback when LLM consolidation fails
        """
        print("DEBUG: Using fallback consolidation")
        
        key_findings = []
        
        # Extract findings from text analysis if available
        if text_analysis and "competency_analysis" in text_analysis:
            comp_analysis = text_analysis["competency_analysis"]
            
            # Add strengths
            for strength in comp_analysis.get("strengths", []):
                key_findings.append({
                    "category": "strength",
                    "title": strength.get("pattern_text", "Unknown strength")[:50],
                    "description": f"Consistent pattern noted by evaluators",
                    "evidence": [ev.get("text", "")[:100] for ev in strength.get("supporting_evidence", [])[:2]],
                    "confidence": strength.get("confidence", "low"),
                    "source_count": len(strength.get("supporting_evidence", []))
                })
            
            # Add improvements
            for improvement in comp_analysis.get("improvements", []):
                key_findings.append({
                    "category": "improvement",
                    "title": improvement.get("pattern_text", "Unknown improvement")[:50],
                    "description": f"Area identified for development",
                    "evidence": [ev.get("text", "")[:100] for ev in improvement.get("supporting_evidence", [])[:2]],
                    "confidence": improvement.get("confidence", "low"),
                    "source_count": len(improvement.get("supporting_evidence", []))
                })
        
        return {
            "summary": f"Analysis for query: {user_query}",
            "key_findings": key_findings[:5],  # Limit to top 5
            "numeric_context": {
                "relevant_scores": numeric_analysis.get("by_epa", {}) if numeric_analysis else {},
                "trends": "Limited trend data available"
            },
            "data_quality": {
                "total_evaluations": "Multiple evaluators",
                "evaluator_types": ["Attending", "Resident"],
                "rotations_covered": ["Multiple rotations"], 
                "confidence_assessment": "Fallback analysis - limited detail available"
            }
        }
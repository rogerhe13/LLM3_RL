"""
Text analysis agent
"""

import json
import re
from typing import Dict, Any, List
from datetime import datetime
from langchain.chat_models.base import BaseChatModel
from langchain.chains import LLMChain
from langchain.prompts import ChatPromptTemplate
from memory.shared_memory import SharedMemory
from config.constants import TEXT_ANALYSIS_PROMPT  

class TextAnalysisAgent:
    """
    Text analysis agent with enhanced prompting and sophisticated pattern confidence calculation
    """
    
    def __init__(self, llm: BaseChatModel, shared_memory: SharedMemory):
        self.llm = llm
        self.shared_memory = shared_memory
        
        # Use the enhanced prompt from constants
        self.prompt = ChatPromptTemplate.from_messages([
            ("human", TEXT_ANALYSIS_PROMPT)
        ])
        self.chain = LLMChain(llm=llm, prompt=self.prompt)
    
    def run(self) -> Dict[str, Any]:
        """
        Analyze text comments with enhanced query-specific relevance filtering
        """
        # Get query context from shared memory
        user_query = self.shared_memory.get("user_query")
        structured_query = self.shared_memory.get("structured_query")
        parsed_data = self.shared_memory.get("parsed_data")
        
        if not structured_query:
            print("ERROR: No structured query found. Query understanding must run first.")
            return {}
        
        # Apply rotation filtering if specified
        filtered_data = self._apply_filters(parsed_data, structured_query)
        
        # Extract query context for the prompt
        query_type = structured_query.get("query_type", "general_performance")
        competency_focus = structured_query.get("competency_focus")
        temporal_dimension = structured_query.get("temporal_dimension", False)
        rotation_filters = structured_query.get("rotation_filters", [])
        epa_filters = structured_query.get("epa_filters", [])
        specific_numbers = structured_query.get("specific_numbers", {})
        evidence_criteria = structured_query.get("evidence_criteria", "Any relevant feedback")
        
        print(f"DEBUG Text Analysis - Query Context:")
        print(f"  Original query: {user_query}")
        print(f"  Query type: {query_type}")
        print(f"  Competency focus: {competency_focus}")
        print(f"  Filtered data: {len(filtered_data)} records")
        
        try:
            # Use .invoke() with proper response extraction
            response_dict = self.chain.invoke({
                "user_query": user_query,
                "query_type": query_type,
                "competency_focus": competency_focus or "general",
                "temporal_dimension": temporal_dimension,
                "rotation_filters": rotation_filters,
                "epa_filters": epa_filters,
                "specific_numbers": specific_numbers,
                "evidence_criteria": evidence_criteria,
                "parsed_data": json.dumps(filtered_data[:10], indent=2)  # Limit to prevent token overflow
            })
            
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
                    # If still no JSON found, try the whole response
                    json_str = response
            
            # Parse LLM response
            text_analysis = json.loads(json_str)
            
            # ENHANCED: Apply pattern confidence calculation to LLM results
            text_analysis = self._enhance_with_pattern_confidence(text_analysis)
            
            print(f"DEBUG Text Analysis Results:")
            print(f"  Relevant feedback found: {text_analysis.get('relevant_feedback_found')}")
            if text_analysis.get('competency_analysis'):
                strengths = text_analysis['competency_analysis'].get('strengths', [])
                improvements = text_analysis['competency_analysis'].get('improvements', [])
                print(f"  Strengths found: {len(strengths)}")
                print(f"  Improvements found: {len(improvements)}")
            
            # Apply number limitations if requested
            text_analysis = self._apply_number_limits(text_analysis, specific_numbers)
            
        except Exception as e:
            print(f"LLM text analysis failed: {e}")
            print(f"Raw LLM response: {response if 'response' in locals() else 'No response'}")
            
            # ENHANCED FALLBACK: With proper pattern confidence calculation
            text_analysis = self._enhanced_fallback_analysis(filtered_data, structured_query)
        
        # Save to shared memory
        self.shared_memory.set("text_analysis", text_analysis)
        
        return text_analysis
    
    def _calculate_pattern_confidence(self, supporting_evidence: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calculate pattern confidence based on evaluator consensus and validation multipliers
        
        Args:
            supporting_evidence: List of evidence items with evaluator_role, rotation, date
            
        Returns:
            Dict with confidence level, score, and description
        """
        if not supporting_evidence:
            return {
                "confidence": "low",
                "score": 0.0,
                "description": "No supporting evidence"
            }
        
        # Base Score (Evaluator Count Priority)
        evaluator_count = len(supporting_evidence)
        
        if evaluator_count >= 4:
            base_score = 1.0
        elif evaluator_count == 3:
            base_score = 0.7
        elif evaluator_count == 2:
            base_score = 0.4
        else:  # 1 evaluator
            base_score = 0.1
        
        # Context Validation Multipliers
        multiplier = 1.0
        
        # Cross-rotation consistency check
        unique_rotations = set()
        for evidence in supporting_evidence:
            rotation = evidence.get("rotation", "Unknown")
            if rotation != "Unknown":
                # Clean rotation name for comparison
                clean_rotation = rotation.lower().replace("clinical performance assessment", "").strip()
                if clean_rotation:
                    unique_rotations.add(clean_rotation)
        
        rotation_count = len(unique_rotations)
        if rotation_count >= 3:
            multiplier *= 1.3  # Strong cross-rotation consistency
        elif rotation_count == 2:
            multiplier *= 1.15  # Moderate cross-rotation consistency
        # rotation_count == 1 gets no boost
        
        # Temporal consistency check (if dates available)
        unique_dates = set()
        for evidence in supporting_evidence:
            date = evidence.get("date", "")
            if date and date != "Unknown date":
                unique_dates.add(date)
        
        if len(unique_dates) >= 2:
            # Check if dates span multiple months (basic temporal consistency)
            try:
                parsed_dates = []
                for date_str in unique_dates:
                    for fmt in ["%Y-%m-%d", "%m/%d/%Y", "%m/%d/%y"]:
                        try:
                            parsed_dates.append(datetime.strptime(date_str, fmt))
                            break
                        except ValueError:
                            continue
                
                if len(parsed_dates) >= 2:
                    # Check if dates span more than 30 days
                    date_range = max(parsed_dates) - min(parsed_dates)
                    if date_range.days > 30:
                        multiplier *= 1.2  # Strong temporal consistency
                    elif date_range.days > 7:
                        multiplier *= 1.1  # Moderate temporal consistency
            except:
                # If date parsing fails, give small boost for having multiple dates
                multiplier *= 1.1
        
        # Role diversity check
        unique_roles = set()
        for evidence in supporting_evidence:
            role = evidence.get("evaluator_role", "Unknown")
            if role != "Unknown":
                unique_roles.add(role.lower())
        
        # Check if both residents and attendings are represented
        has_resident = any("resident" in role for role in unique_roles)
        has_attending = any("attending" in role for role in unique_roles)
        
        if has_resident and has_attending:
            multiplier *= 1.1  # Role diversity boost
        
        # Calculate final weighted score
        final_score = base_score * multiplier
        final_score = min(1.0, final_score)  # Cap at 1.0
        
        # Determine confidence level
        if final_score >= 0.8:
            confidence_level = "high"
        elif final_score >= 0.6:
            confidence_level = "medium-high"
        elif final_score >= 0.35:
            confidence_level = "medium"
        elif final_score >= 0.15:
            confidence_level = "low-medium"
        else:
            confidence_level = "low"
        
        # Create description
        rotation_desc = f"{rotation_count} rotation{'s' if rotation_count != 1 else ''}" if rotation_count > 0 else "unknown rotations"
        
        description = f"{evaluator_count} evaluator{'s' if evaluator_count != 1 else ''} across {rotation_desc}"
        
        if multiplier > 1.0:
            boosts = []
            if rotation_count >= 2:
                boosts.append("cross-rotation consistency")
            if len(unique_dates) >= 2:
                boosts.append("temporal consistency")
            if has_resident and has_attending:
                boosts.append("role diversity")
            
            if boosts:
                description += f" (boosted by {', '.join(boosts)})"
        
        return {
            "confidence": confidence_level,
            "score": round(final_score, 3),
            "description": description
        }
    
    def _filter_by_confidence_threshold(self, patterns: List[Dict[str, Any]], min_confidence_score: float = 0.15) -> List[Dict[str, Any]]:
        """
        Filter out low-confidence patterns below threshold
        
        Args:
            patterns: List of pattern dictionaries with confidence scores
            min_confidence_score: Minimum confidence score to include
            
        Returns:
            Filtered list of patterns
        """
        filtered_patterns = []
        
        for pattern in patterns:
            confidence_info = pattern.get("confidence_info", {})
            score = confidence_info.get("score", 0.0)
            
            if score >= min_confidence_score:
                filtered_patterns.append(pattern)
        
        return filtered_patterns
    
    def _enhance_with_pattern_confidence(self, text_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enhance LLM results with pattern confidence calculations
        """
        if not text_analysis.get("competency_analysis"):
            return text_analysis
        
        comp_analysis = text_analysis["competency_analysis"]
        
        # Calculate pattern confidence for strengths
        if "strengths" in comp_analysis:
            enhanced_strengths = []
            for strength in comp_analysis["strengths"]:
                # Handle case where strength is just a string
                if isinstance(strength, str):
                    strength_dict = {
                        "pattern_text": strength,
                        "supporting_evidence": []
                    }
                else:
                    strength_dict = strength
                
                supporting_evidence = strength_dict.get("supporting_evidence", [])
                confidence_info = self._calculate_pattern_confidence(supporting_evidence)
                
                strength_dict["confidence"] = confidence_info["confidence"]
                strength_dict["confidence_info"] = confidence_info
                strength_dict["confidence_description"] = confidence_info["description"]
                enhanced_strengths.append(strength_dict)
            
            comp_analysis["strengths"] = enhanced_strengths
        
        # Calculate pattern confidence for improvements
        if "improvements" in comp_analysis:
            enhanced_improvements = []
            for improvement in comp_analysis["improvements"]:
                # Handle case where improvement is just a string
                if isinstance(improvement, str):
                    improvement_dict = {
                        "pattern_text": improvement,
                        "supporting_evidence": []
                    }
                else:
                    improvement_dict = improvement
                
                supporting_evidence = improvement_dict.get("supporting_evidence", [])
                confidence_info = self._calculate_pattern_confidence(supporting_evidence)
                
                improvement_dict["confidence"] = confidence_info["confidence"]
                improvement_dict["confidence_info"] = confidence_info
                improvement_dict["confidence_description"] = confidence_info["description"]
                enhanced_improvements.append(improvement_dict)
            
            comp_analysis["improvements"] = enhanced_improvements
        
        # Filter by confidence threshold
        if "strengths" in comp_analysis:
            comp_analysis["strengths"] = self._filter_by_confidence_threshold(comp_analysis["strengths"])
        if "improvements" in comp_analysis:
            comp_analysis["improvements"] = self._filter_by_confidence_threshold(comp_analysis["improvements"])
        
        return text_analysis
    
    def _enhanced_fallback_analysis(self, parsed_data: List[Dict[str, Any]], structured_query: Dict[str, Any]) -> Dict[str, Any]:
        """
        ENHANCED fallback analysis with proper pattern confidence calculation
        """
        print("Using enhanced fallback text analysis with pattern confidence...")
        
        competency_focus = structured_query.get("competency_focus")
        temporal_dimension = structured_query.get("temporal_dimension", False)
        
        # Enhanced keyword-based relevance for fallback
        relevant_keywords = {
            "clinical_reasoning": ["reasoning", "diagnostic", "differential", "decision", "analysis", "thinking", "succinct", "decisive", "assessment", "plan"],
            "communication": ["communication", "listening", "patient interaction", "bedside manner", "empathy", "compassion"],
            "professionalism": ["professional", "reliability", "ethics", "responsibility", "punctual", "integrity", "feedback"],
            "patient_care": ["patient care", "bedside manner", "empathy", "compassion", "advocacy"],
            "presentation_skills": ["presentation", "presenting", "oral", "rounds"],
            "teamwork": ["team", "collaboration", "teamwork", "interaction"]
        }
        
        # Group similar feedback by pattern
        strength_patterns = {}
        improvement_patterns = {}
        
        for row in parsed_data:
            # Process strengths
            strength_text = row.get("strengths_comment", "")
            if strength_text and self._is_relevant_text(strength_text, competency_focus, relevant_keywords):
                # Simple pattern grouping by first few words
                pattern_key = " ".join(strength_text.split()[:5]).lower()
                
                if pattern_key not in strength_patterns:
                    strength_patterns[pattern_key] = {
                        "pattern_text": strength_text[:150] + "..." if len(strength_text) > 150 else strength_text,
                        "supporting_evidence": []
                    }
                
                strength_patterns[pattern_key]["supporting_evidence"].append({
                    "text": strength_text,
                    "evaluator_role": row.get("evaluator_role", "Unknown"),
                    "rotation": row.get("form_name", "Unknown rotation"),
                    "date": row.get("release_date_str", "Unknown date"),
                    "relevance_score": "medium"
                })
            
            # Process improvements
            improvement_text = row.get("improvements_comment", "")
            if improvement_text and self._is_relevant_text(improvement_text, competency_focus, relevant_keywords):
                # Simple pattern grouping by first few words
                pattern_key = " ".join(improvement_text.split()[:5]).lower()
                
                if pattern_key not in improvement_patterns:
                    improvement_patterns[pattern_key] = {
                        "pattern_text": improvement_text[:150] + "..." if len(improvement_text) > 150 else improvement_text,
                        "supporting_evidence": []
                    }
                
                improvement_patterns[pattern_key]["supporting_evidence"].append({
                    "text": improvement_text,
                    "evaluator_role": row.get("evaluator_role", "Unknown"),
                    "rotation": row.get("form_name", "Unknown rotation"),
                    "date": row.get("release_date_str", "Unknown date"),
                    "relevance_score": "medium"
                })
        
        # Convert to lists and calculate pattern confidence
        strengths = []
        for pattern in strength_patterns.values():
            confidence_info = self._calculate_pattern_confidence(pattern["supporting_evidence"])
            pattern["confidence"] = confidence_info["confidence"]
            pattern["confidence_info"] = confidence_info
            pattern["confidence_description"] = confidence_info["description"]
            strengths.append(pattern)
        
        improvements = []
        for pattern in improvement_patterns.values():
            confidence_info = self._calculate_pattern_confidence(pattern["supporting_evidence"])
            pattern["confidence"] = confidence_info["confidence"]
            pattern["confidence_info"] = confidence_info
            pattern["confidence_description"] = confidence_info["description"]
            improvements.append(pattern)
        
        # Filter by confidence threshold
        strengths = self._filter_by_confidence_threshold(strengths)
        improvements = self._filter_by_confidence_threshold(improvements)
        
        # Sort by confidence score (highest first)
        strengths.sort(key=lambda x: x.get("confidence_info", {}).get("score", 0), reverse=True)
        improvements.sort(key=lambda x: x.get("confidence_info", {}).get("score", 0), reverse=True)
        
        # Limit results
        strengths = strengths[:5]
        improvements = improvements[:5]
        
        return {
            "relevant_feedback_found": len(strengths) > 0 or len(improvements) > 0,
            "competency_analysis": {
                "strengths": strengths,
                "improvements": improvements,
                "temporal_progression": {
                    "trend_direction": "insufficient_data", 
                    "evidence": "Enhanced fallback analysis - temporal analysis limited",
                    "early_feedback": [],
                    "recent_feedback": []
                }
            },
            "alternative_suggestions": "Consider asking about general performance or specific areas where more feedback is available."
        }
    
    def _apply_filters(self, parsed_data: List[Dict[str, Any]], structured_query: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Apply rotation and other filters to the data
        """
        filtered_data = parsed_data.copy()
        
        # Apply rotation filters
        rotation_filters = structured_query.get("rotation_filters", [])
        if rotation_filters:
            filtered_data = [
                row for row in filtered_data
                if any(rotation.lower() in row.get("form_name", "").lower() for rotation in rotation_filters)
            ]
            print(f"  Applied rotation filter {rotation_filters}: {len(filtered_data)} records remain")
        
        return filtered_data
    
    def _apply_number_limits(self, text_analysis: Dict[str, Any], specific_numbers: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply number limitations (top 3 strengths, etc.)
        """
        if not text_analysis.get("competency_analysis"):
            return text_analysis
        
        comp_analysis = text_analysis["competency_analysis"]
        
        # Apply strength limits
        strengths_requested = specific_numbers.get("strengths_requested")
        if strengths_requested and "strengths" in comp_analysis:
            comp_analysis["strengths"] = comp_analysis["strengths"][:strengths_requested]
        
        # Apply improvement limits  
        improvements_requested = specific_numbers.get("improvements_requested")
        if improvements_requested and "improvements" in comp_analysis:
            comp_analysis["improvements"] = comp_analysis["improvements"][:improvements_requested]
        
        # Apply general top limit
        top_requested = specific_numbers.get("top_requested")
        if top_requested:
            if "strengths" in comp_analysis:
                comp_analysis["strengths"] = comp_analysis["strengths"][:top_requested]
            if "improvements" in comp_analysis:
                comp_analysis["improvements"] = comp_analysis["improvements"][:top_requested]
        
        return text_analysis
    
    def _is_relevant_text(self, text: str, competency_focus: str, relevant_keywords: Dict[str, List[str]]) -> bool:
        """
        Check if text is relevant to the competency focus
        """
        # Handle non-string inputs
        if not isinstance(text, str):
            return False
            
        if not competency_focus or not text:
            return True  
        
        text_lower = text.lower()
        keywords = relevant_keywords.get(competency_focus, [])
        
        return any(keyword in text_lower for keyword in keywords)
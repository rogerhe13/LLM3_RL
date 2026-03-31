"""
Numeric analysis agent
"""

from typing import Dict, Any, List
import json
import re
import statistics
from collections import defaultdict
from datetime import datetime

from langchain.chat_models.base import BaseChatModel
from langchain.chains import LLMChain
from langchain.prompts import ChatPromptTemplate
from memory.shared_memory import SharedMemory
from config.constants import NUMERIC_ANALYSIS_PROMPT  # Import from constants

class NumericAnalysisAgent:
    """
    Numeric analysis agent with enhanced temporal analysis using centralized prompt
    """
    
    def __init__(self, llm: BaseChatModel, shared_memory: SharedMemory):
        self.llm = llm
        self.shared_memory = shared_memory
        
        # Use the enhanced prompt from constants
        self.prompt = ChatPromptTemplate.from_messages([
            ("human", NUMERIC_ANALYSIS_PROMPT)
        ])
        self.chain = LLMChain(llm=llm, prompt=self.prompt)
    
    def run(self) -> Dict[str, Any]:
        """
        Perform numeric analysis using enhanced prompt from constants
        """
        user_query = self.shared_memory.get("user_query")
        structured_query = self.shared_memory.get("structured_query", {})
        parsed_data = self.shared_memory.get("parsed_data")
        
        # Extract query context for the enhanced prompt
        query_type = structured_query.get("query_type", "general_performance")
        competency_focus = structured_query.get("competency_focus")
        temporal_dimension = structured_query.get("temporal_dimension", False)
        rotation_filters = structured_query.get("rotation_filters", [])
        epa_filters = structured_query.get("epa_filters", [])
        
        print(f"DEBUG Numeric Analysis:")
        print(f"  Query type: {query_type}")
        print(f"  Competency focus: {competency_focus}")
        print(f"  Temporal analysis: {temporal_dimension}")
        print(f"  Rotation filters: {rotation_filters}")
        
        try:
            # FIXED: Use .invoke() with proper response extraction
            response_dict = self.chain.invoke({
                "user_query": user_query,
                "query_type": query_type,
                "competency_focus": competency_focus or "general",
                "temporal_dimension": temporal_dimension,
                "rotation_filters": rotation_filters,
                "epa_filters": epa_filters,
                "parsed_data": json.dumps(parsed_data[:10], indent=2)  # Sample for LLM
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
            numeric_analysis = json.loads(json_str)
            
            print("DEBUG: LLM numeric analysis successful")
            
        except Exception as e:
            print(f"LLM numeric analysis failed: {e}")
            print(f"Raw LLM response: {response if 'response' in locals() else 'No response'}")
            
            # ENHANCED FALLBACK: Use the sophisticated logic when LLM fails
            numeric_analysis = self._enhanced_fallback_analysis(parsed_data, structured_query)
        
        self.shared_memory.set("numeric_analysis", numeric_analysis)
        return numeric_analysis
    
    def _enhanced_fallback_analysis(self, parsed_data: List[Dict[str, Any]], 
                                   structured_query: Dict[str, Any]) -> Dict[str, Any]:
        """
        FALLBACK: Use the sophisticated analysis logic when LLM fails
        This includes all the temporal analysis logic from the previous version
        """
        print("DEBUG: Using enhanced fallback numeric analysis")
        
        professionalism_fields = self.shared_memory.get_static_mapping("professionalism_fields")
        
        result = {
            "by_epa": {},
            "by_communication": {},
            "by_professionalism": {},
            "query_specific_analysis": {},
            "temporal_analysis": self._analyze_temporal_progression(parsed_data, structured_query)
        }
        
        # Get all numeric fields dynamically
        all_numeric_fields = set()
        for row in parsed_data:
            for field, value in row.items():
                if (field.startswith("epa") or field.startswith("comm_") or field.startswith("prof_")) and \
                   value is not None and isinstance(value, (int, float)):
                    all_numeric_fields.add(field)
        
        # Process individual field scores with temporal trend analysis
        for field in all_numeric_fields:
            field_scores = []
            field_weights = []
            field_scores_with_dates = []
            
            for row in parsed_data:
                if field in row and row[field] is not None and isinstance(row[field], (int, float)):
                    field_scores.append(row[field])
                    weight = row.get("recency_weight", 1.0)
                    field_weights.append(weight)
                    
                    date_str = row.get("release_date_str") or row.get("release_date")
                    if date_str:
                        field_scores_with_dates.append({
                            "date": date_str,
                            "score": row[field],
                            "weight": weight
                        })
            
            if field_scores:
                weighted_avg = round(self._weighted_mean(field_scores, field_weights), 2)
                
                field_stats = {
                    "avg": weighted_avg,
                    "weighted_avg": weighted_avg,
                    "raw_avg": round(statistics.mean(field_scores), 2),
                    "min": min(field_scores),
                    "max": max(field_scores),
                    "count": len(field_scores),
                    "recent_trend": self._calculate_trend_fixed(field_scores_with_dates)
                }
                
                # Categorize by field type
                if field.startswith("epa"):
                    result["by_epa"][field] = field_stats
                elif field.startswith("comm_"):
                    result["by_communication"][field] = field_stats
                elif field.startswith("prof_"):
                    result["by_professionalism"][field] = field_stats
        
        return result
    
    def _weighted_mean(self, values: List[float], weights: List[float]) -> float:
        """Calculate weighted mean of a list of values"""
        if not values or not weights or len(values) != len(weights):
            return 0
            
        numerator = sum(value * weight for value, weight in zip(values, weights))
        denominator = sum(weights)
        
        return numerator / denominator if denominator > 0 else 0
    
    def _parse_date(self, date_str: str) -> datetime:
        """Parse dates with all possible formats from your data"""
        if not date_str:
            return None
            
        # Include the actual format used in your data
        date_formats = [
            "%Y-%m-%d",    # 2023-03-02 (YOUR ACTUAL FORMAT)
            "%m/%d/%Y",    # 3/2/2023
            "%m/%d/%y",    # 3/2/23
            "%d/%m/%Y",    # 2/3/2023
            "%Y/%m/%d"     # 2023/03/02
        ]
        
        for fmt in date_formats:
            try:
                return datetime.strptime(date_str, fmt)
            except ValueError:
                continue
        
        print(f"WARNING: Could not parse date: {date_str}")
        return None
    
    def _analyze_temporal_progression(self, parsed_data: List[Dict[str, Any]], 
                                    structured_query: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze temporal progression for clinical reasoning and other competencies"""
        is_temporal_query = structured_query.get("temporal_dimension", False)
        competency_focus = structured_query.get("competency_focus")
        
        if not is_temporal_query:
            return {"temporal_analysis_performed": False}
        
        print(f"DEBUG: Performing temporal analysis for competency: {competency_focus}")
        
        # Group data by time periods
        temporal_data = []
        
        for row in parsed_data:
            date_str = row.get("release_date_str") or row.get("release_date")
            if not date_str:
                continue
                
            parsed_date = self._parse_date(date_str)
            if parsed_date is None:
                continue
            
            temporal_data.append({
                "date": parsed_date,
                "date_str": date_str,
                "rotation": row.get("form_name", "Unknown"),
                "evaluator_role": row.get("evaluator_role", "Unknown"),
                "epa_scores": {k: v for k, v in row.items() if k.startswith("epa") and isinstance(v, (int, float))},
                "comm_scores": {k: v for k, v in row.items() if k.startswith("comm_") and isinstance(v, (int, float))},
                "recency_weight": row.get("recency_weight", 1.0)
            })
        
        # Sort by date
        temporal_data.sort(key=lambda x: x["date"])
        
        if len(temporal_data) < 2:
            return {
                "temporal_analysis_performed": True,
                "insufficient_data": True,
                "message": "Need at least 2 time points for temporal analysis"
            }
        
        # Analyze progression by time periods
        early_period = temporal_data[:len(temporal_data)//2]  # First half
        recent_period = temporal_data[len(temporal_data)//2:]  # Second half
        
        # Analyze EPA trends (EPAs 1-3 are most related to clinical reasoning)
        reasoning_epas = ["epa1", "epa2", "epa3"]
        early_epa_scores = []
        recent_epa_scores = []
        
        for period_data, score_list in [(early_period, early_epa_scores), (recent_period, recent_epa_scores)]:
            for eval_data in period_data:
                period_scores = []
                for epa in reasoning_epas:
                    if epa in eval_data["epa_scores"] and eval_data["epa_scores"][epa] is not None:
                        period_scores.append(eval_data["epa_scores"][epa])
                if period_scores:
                    score_list.append(sum(period_scores) / len(period_scores))
        
        # Calculate trends
        epa_trend = "stable"
        epa_change = 0
        if early_epa_scores and recent_epa_scores:
            early_avg = sum(early_epa_scores) / len(early_epa_scores)
            recent_avg = sum(recent_epa_scores) / len(recent_epa_scores)
            epa_change = recent_avg - early_avg
            
            if epa_change > 0.3:
                epa_trend = "improving"
            elif epa_change < -0.3:
                epa_trend = "declining"
        
        return {
            "temporal_analysis_performed": True,
            "total_evaluations": len(temporal_data),
            "time_span": {
                "earliest": temporal_data[0]["date_str"],
                "most_recent": temporal_data[-1]["date_str"],
                "rotations": list(set([d["rotation"] for d in temporal_data]))
            },
            "epa_progression": {
                "direction": epa_trend,
                "change": round(epa_change, 2),
                "early_avg": round(sum(early_epa_scores) / len(early_epa_scores), 2) if early_epa_scores else None,
                "recent_avg": round(sum(recent_epa_scores) / len(recent_epa_scores), 2) if recent_epa_scores else None
            }
        }
    
    def _calculate_trend_fixed(self, scores_with_dates: List[Dict]) -> Dict[str, Any]:
        """Calculate trend information with proper date parsing"""
        if not scores_with_dates or len(scores_with_dates) < 2:
            return {"direction": "stable", "magnitude": 0}
        
        # Parse dates properly
        parsed_scores = []
        for item in scores_with_dates:
            parsed_date = self._parse_date(item["date"])
            if parsed_date:
                parsed_scores.append({
                    "date": parsed_date,
                    "score": item["score"],
                    "weight": item.get("weight", 1.0)
                })
        
        if len(parsed_scores) < 2:
            return {"direction": "stable", "magnitude": 0}
        
        # Sort by date
        parsed_scores.sort(key=lambda x: x["date"])
        
        # Get earliest and most recent scores
        earliest = parsed_scores[0]["score"]
        most_recent = parsed_scores[-1]["score"]
        
        # Calculate change
        change = most_recent - earliest
        
        # Determine trend direction and magnitude
        if abs(change) < 0.3:  # Less than 0.3 point change
            direction = "stable"
        else:
            direction = "improving" if change > 0 else "declining"
        
        # Normalize magnitude to a 0-1 scale (assuming scores are on a 4-point scale)
        magnitude = min(1.0, abs(change) / 4.0)
        
        return {
            "direction": direction,
            "magnitude": round(magnitude, 2),
            "change": round(change, 2),
            "earliest_score": earliest,
            "most_recent_score": most_recent,
            "time_span": f"{parsed_scores[0]['date'].strftime('%Y-%m')} to {parsed_scores[-1]['date'].strftime('%Y-%m')}"
        }
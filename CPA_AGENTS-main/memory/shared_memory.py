"""
Shared memory implementation - Central data storage for all agents in the system
"""

from typing import Dict, Any, List, Optional


class SharedMemory:
    """
    Shared memory implementation, serving as the central "blackboard" for all agents.
    Stores parsed data, structured queries, and various analysis results.
    """
    
    def __init__(self):
        self._memory: Dict[str, Any] = {
            "parsed_data": None,  # Parsed data
            "structured_query": None,  # Structured query
            "numeric_analysis": None,  # Numeric analysis
            "text_analysis": None,  # Text analysis
            "consolidated_summary": None,  # Consolidated summary
            "raw_table": None,  # Raw CSV data
            "columns": None,  # Column names
            "user_query": None,  # User query
        }
        
       
        self._static_mappings = {
            
            "column_type_map": {
                "student_id": "text",
                "form_name": "text",
                "phase_name": "text",
                "academic_year": "date",
                "release_date": "date",
                "evaluator_role": "text",
                "frequency": "text",
                "strengths_comment": "text",
                "improvements_comment": "text",
                # Professionalism EPA fields (numeric scale)
                "prof_shows_dependability_truthfulness_and_integrity": "int",
                "prof_acknowledges_and_demonstrates_awareness_of_limitations": "int",
                "prof_takes_initiative_for_own_learning_and_patient_care": "int",
                "prof_remains_open_to_feedback_and_attempts_to_implement_it": "int",
                "prof_treats_all_patients_with_respect_and_compassion_protects_patient_confidentiality": "int",
                # Communication fields
                "comm_listening": "int",
                "comm_decision_making": "int",
                "comm_advocacy": "int",
                "comm_other": "text",
                # EPA fields (numeric scale)
                "epa1": "int",  # History Taking and Physical Exam
                "epa2": "int",  # Clinical Reasoning, Differential Diagnosis
                "epa3": "int",  # Recommend & Interpret Tests
                "epa4": "int",  # Enter & Discuss Orders and Prescriptions
                "epa5": "int",  # Written Notes
                "epa6": "int",  # Oral Presentation of Patient
                "epa7": "int",  # Medical Decision Making
                "epa8": "int",  # Providing Appropriate Patient Transitions
                "epa9": "int",  # Contributes as a Member of the Team
                "epa10": "int", # Recognition of Patients Needing Urgent Care
                "epa14": "int", # Teaching of Students
            },
            
            # Keyword to field mapping (no domains, direct field mapping)
            "keyword_field_map": {
                # EPA mappings
                "history taking": "epa1",
                "history": "epa1", 
                "H&P": "epa1",
                "HandP": "epa1",
                "physical exam": "epa1",
                "PE": "epa1",
                "clinical reasoning": "epa2",
                "differential diagnosis": "epa2",
                "ddx": "epa2",
                "diagnostic tests": "epa3",
                "screening tests": "epa3",
                "interpret tests": "epa3",
                "recommend tests": "epa3",
                "orders": "epa4",
                "prescriptions": "epa4",
                "documentation": "epa5",
                "written notes": "epa5",
                "oral presentation": "epa6",
                "presentation": "epa6",
                "medical decision": "epa7",
                "literature": "epa7",
                "transitions": "epa8",
                "handoff": "epa8",
                "team member": "epa9",
                "teamwork": "epa9",
                "interaction": "epa9",
                "urgent care": "epa10",
                "emergent care": "epa10",
                "teaching": "epa14",
                "mentoring": "epa14",
                
                # Communication mappings
                "listening": "comm_listening",
                "decision making": "comm_decision_making",
                "shared decision": "comm_decision_making",
                "social determinants": "comm_advocacy",
                "advocacy": "comm_advocacy",
                "advocates": "comm_advocacy",
                "patient advocacy": "comm_advocacy",
                
                # Professionalism mappings
                "professionalism": "professionalism_category",
                "integrity": "prof_shows_dependability_truthfulness_and_integrity",
                "dependability": "prof_shows_dependability_truthfulness_and_integrity",
                "limitations": "prof_acknowledges_and_demonstrates_awareness_of_limitations",
                "initiative": "prof_takes_initiative_for_own_learning_and_patient_care",
                "open to feedback": "prof_remains_open_to_feedback_and_attempts_to_implement_it",
                "respect": "prof_treats_all_patients_with_respect_and_compassion_protects_patient_confidentiality",
                "compassion": "prof_treats_all_patients_with_respect_and_compassion_protects_patient_confidentiality",
                "confidentiality": "prof_treats_all_patients_with_respect_and_compassion_protects_patient_confidentiality",
                
                # Text fields
                "feedback": "strengths_comment,improvements_comment",
                "strengths": "strengths_comment",
                "improvements": "improvements_comment",
            },
            
            # Professionalism fields grouped for easy reference
            "professionalism_fields": [
                "prof_shows_dependability_truthfulness_and_integrity",
                "prof_acknowledges_and_demonstrates_awareness_of_limitations",
                "prof_takes_initiative_for_own_learning_and_patient_care",
                "prof_remains_open_to_feedback_and_attempts_to_implement_it",
                "prof_treats_all_patients_with_respect_and_compassion_protects_patient_confidentiality"
            ]
        }
        
        
        self._session_memory: Dict[str, Any] = {
            "last_response": None  # Last response, for multi-turn follow-up
        }
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        FIXED: Get a value from shared memory with optional default
        
        Args:
            key: The key to retrieve
            default: Default value to return if key not found (optional)
            
        Returns:
            The value associated with the key, or default if not found
        """
        if key in self._memory:
            value = self._memory.get(key)
            return value if value is not None else default
        return default
    
    def set(self, key: str, value: Any) -> None:
        """Set a value in shared memory"""
        if key in self._memory:
            self._memory[key] = value
    
    def get_static_mapping(self, mapping_name: str) -> Dict[str, Any]:
        """Get a static mapping"""
        return self._static_mappings.get(mapping_name, {})
    
    def get_session_data(self, key: str) -> Any:
        """Get a value from session memory"""
        return self._session_memory.get(key)
    
    def set_session_data(self, key: str, value: Any) -> None:
        """Set a value in session memory"""
        self._session_memory[key] = value
    
    def clear_session(self) -> None:
        """Clear session data"""
        self._session_memory = {"last_response": None}
    
    def clear_all(self) -> None:
        """Clear all data, including main memory"""
        for key in self._memory:
            self._memory[key] = None
        self.clear_session()
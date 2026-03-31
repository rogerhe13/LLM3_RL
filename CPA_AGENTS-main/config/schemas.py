
"""
Data schemas and structures
"""
from typing import List, Dict, Any, Optional, Union
from pydantic import BaseModel, Field

class NumericStats(BaseModel):
    #Numeric statistics model
    avg: float = Field(..., description="Average value")
    min: int = Field(..., description="Minimum value")
    max: int = Field(..., description="Maximum value")

class DomainFeedback(BaseModel):
    #Domain feedback model
    strengths: List[str] = Field(default_factory=list, description="List of strengths")
    improvements: List[str] = Field(default_factory=list, description="List of improvement points")

class DomainSummary(BaseModel):
    #Domain summary model
    numeric: Optional[NumericStats] = Field(None, description="Numeric statistics")
    strengths: List[str] = Field(default_factory=list, description="List of strengths")
    improvements: List[str] = Field(default_factory=list, description="List of improvement points")

class StructuredQuery(BaseModel):
    #Structured query model
    ask_strengths: Optional[int] = Field(None, description="Number of strengths requested")
    ask_improvements: Optional[int] = Field(None, description="Number of improvement points requested")
    ask_category_performance: List[str] = Field(default_factory=list, description="List of requested categories")
    ask_epa_performance: List[str] = Field(default_factory=list, description="List of requested EPAs")

class NumericAnalysis(BaseModel):
    #Numeric analysis results model
    by_domain: Dict[str, NumericStats] = Field(default_factory=dict, description="Statistics organized by domain")
    by_epa: Dict[str, NumericStats] = Field(default_factory=dict, description="Statistics organized by EPA")

class TextAnalysis(BaseModel):
    #Text analysis results model
    __root__: Dict[str, DomainFeedback] = Field(..., description="Text feedback organized by domain")

class ConsolidatedSummary(BaseModel):
    #Consolidated summary model
    __root__: Dict[str, DomainSummary] = Field(..., description="Comprehensive summary organized by domain")
    
    class Config:
        arbitrary_types_allowed = True
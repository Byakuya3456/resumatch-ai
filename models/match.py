# models/match.py

from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum

class MatchLevel(str, Enum):
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"

class MatchStatus(str, Enum):
    PENDING = "pending"
    REVIEWED = "reviewed"
    CONTACTED = "contacted"
    INTERVIEWING = "interviewing"
    HIRED = "hired"
    REJECTED = "rejected"

class AdvancedMatch(BaseModel):
    id: Optional[str] = Field(default=None, alias="_id")
    candidate_id: str
    job_id: str
    
    # Scores
    overall_score: float = Field(..., ge=0.0, le=1.0)
    skill_score: float = Field(..., ge=0.0, le=1.0)
    experience_score: float = Field(..., ge=0.0, le=1.0)
    culture_score: float = Field(..., ge=0.0, le=1.0)
    
    match_level: MatchLevel
    status: MatchStatus = MatchStatus.PENDING
    
    # Detailed analysis
    strengths: List[str] = Field(default_factory=list)
    weaknesses: List[str] = Field(default_factory=list)
    skill_gaps: List[str] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)
    
    # LLM analysis
    llm_analysis: Optional[Dict[str, Any]] = None
    ml_prediction: Optional[Dict[str, Any]] = None
    
    # Metadata
    explanation: Optional[str] = None
    viewed_by_recruiter: bool = False
    contacted_candidate: bool = False
    candidate_response: Optional[bool] = None
    
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    last_viewed: Optional[datetime] = None
    
    class Config:
        arbitrary_types_allowed = True
        from_attributes = True
        validate_by_name = True
        use_enum_values = True
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
    
    @validator('overall_score')
    def validate_overall_score(cls, v, values):
        """Ensure overall score is consistent with component scores."""
        if 'skill_score' in values and 'experience_score' in values and 'culture_score' in values:
            calculated = (values['skill_score'] * 0.4 + 
                         values['experience_score'] * 0.3 + 
                         values['culture_score'] * 0.3)
            if abs(v - calculated) > 0.1:  # Allow some tolerance
                raise ValueError('Overall score does not match component scores')
        return v
    
    @validator('match_level')
    def determine_match_level(cls, v, values):
        """Determine match level based on score."""
        if 'overall_score' in values:
            score = values['overall_score']
            if score >= 0.8:
                return MatchLevel.EXCELLENT
            elif score >= 0.6:
                return MatchLevel.GOOD
            elif score >= 0.4:
                return MatchLevel.FAIR
            else:
                return MatchLevel.POOR
        return v

# For backward compatibility
class Match(AdvancedMatch):
    """Legacy match model for compatibility."""
    score: float = Field(..., ge=0.0, le=1.0)
    
    @validator('score')
    def map_score_to_overall(cls, v):
        """Map legacy score to overall_score."""
        return v
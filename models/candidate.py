# models/candidate.py

from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum
import re

# Try to import email validator, fallback to simple validation
try:
    from pydantic import EmailStr
    HAS_EMAIL_VALIDATOR = True
except ImportError:
    HAS_EMAIL_VALIDATOR = False
    # Simple email pattern for fallback
    EmailStr = str

class ExperienceLevel(str, Enum):
    JUNIOR = "junior"
    MID = "mid"
    SENIOR = "senior"
    UNKNOWN = "unknown"

class Education(BaseModel):
    degree: str
    institution: str
    field_of_study: str
    graduation_year: Optional[int] = None
    gpa: Optional[float] = Field(None, ge=0.0, le=4.0)

class Experience(BaseModel):
    title: str
    company: str
    start_date: str
    end_date: Optional[str] = None
    description: str
    skills_used: List[str] = []

class AdvancedCandidate(BaseModel):
    id: Optional[str] = Field(default=None, alias="_id")
    full_name: str = Field(..., min_length=2, max_length=100)
    
    # Use conditional email validation
    if HAS_EMAIL_VALIDATOR:
        email: Optional[EmailStr] = None
    else:
        email: Optional[str] = None
        
    phone: Optional[str] = None
    location: Optional[str] = None
    summary: Optional[str] = Field(None, max_length=1000)
    
    skills: List[str] = Field(default_factory=list)
    primary_skills: List[str] = Field(default_factory=list)
    secondary_skills: List[str] = Field(default_factory=list)
    
    experience_level: ExperienceLevel = ExperienceLevel.UNKNOWN
    total_experience_years: float = Field(0.0, ge=0.0)
    
    education: List[Education] = Field(default_factory=list)
    experiences: List[Experience] = Field(default_factory=list)
    
    certifications: List[str] = Field(default_factory=list)
    languages: List[str] = Field(default_factory=list)
    
    current_salary: Optional[float] = Field(None, ge=0.0)
    expected_salary: Optional[float] = Field(None, ge=0.0)
    
    notice_period_days: int = Field(0, ge=0, le=90)
    relocation_willingness: bool = False
    
    embedding: Optional[List[float]] = None
    resume_filename: Optional[str] = None
    
    # Metadata
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    last_matched: Optional[datetime] = None
    match_count: int = 0
    profile_completeness: float = Field(0.0, ge=0.0, le=100.0)
    
    class Config:
        arbitrary_types_allowed = True
        from_attributes = True
        validate_by_name = True
        use_enum_values = True
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
    
    @validator('phone')
    def validate_phone(cls, v):
        if v is None:
            return v
        # Simple phone validation
        phone_pattern = r'^[\d\s\-\+\(\)]{10,20}$'
        if not re.match(phone_pattern, v):
            raise ValueError('Invalid phone number format')
        return v
    
    @validator('email', pre=True, always=True)
    def validate_email_fallback(cls, v):
        if not HAS_EMAIL_VALIDATOR and v is not None:
            # Simple email validation fallback
            email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
            if not re.match(email_pattern, v):
                raise ValueError('Invalid email format')
        return v
    
    @validator('skills', 'primary_skills', 'secondary_skills')
    def validate_skills(cls, v):
        # Remove duplicates and empty strings
        return list(set(filter(None, [skill.strip() for skill in v])))
    
    @validator('total_experience_years')
    def validate_experience(cls, v):
        if v > 50:  # Reasonable upper limit
            raise ValueError('Experience years cannot exceed 50')
        return v
    
    @validator('profile_completeness', always=True)
    def calculate_completeness(cls, v, values):
        """Calculate profile completeness percentage."""
        required_fields = ['full_name', 'skills', 'experience_level', 'education']
        filled_fields = sum(1 for field in required_fields if values.get(field))
        return (filled_fields / len(required_fields)) * 100

# For backward compatibility
class Candidate(AdvancedCandidate):
    """Legacy candidate model for compatibility."""
    experience: Optional[str] = None
    education: Optional[str] = None
    
    @validator('experience', pre=True, always=True)
    def format_experience(cls, v, values):
        """Format experience for backward compatibility."""
        if v is None and values.get('total_experience_years'):
            return f"{values['total_experience_years']} years experience"
        return v
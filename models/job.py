# models/job.py

from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict
from datetime import datetime
from enum import Enum

class EmploymentType(str, Enum):
    FULL_TIME = "full_time"
    PART_TIME = "part_time"
    CONTRACT = "contract"
    INTERNSHIP = "internship"
    FREELANCE = "freelance"

class ExperienceLevel(str, Enum):
    ENTRY = "entry"
    JUNIOR = "junior"
    MID = "mid"
    SENIOR = "senior"
    LEAD = "lead"
    PRINCIPAL = "principal"

class SalaryCurrency(str, Enum):
    USD = "USD"
    EUR = "EUR"
    GBP = "GBP"
    INR = "INR"
    CAD = "CAD"
    AUD = "AUD"

class SalaryRange(BaseModel):
    min: float = Field(..., ge=0.0)
    max: float = Field(..., ge=0.0)
    currency: SalaryCurrency = SalaryCurrency.USD
    period: str = "year"  # year, month, hour

class Location(BaseModel):
    city: str
    state: Optional[str] = None
    country: str
    remote: bool = False
    hybrid: bool = False

class AdvancedJob(BaseModel):
    id: Optional[str] = Field(default=None, alias="_id")
    title: str = Field(..., min_length=3, max_length=100)
    company: str = Field(..., min_length=2, max_length=100)
    department: Optional[str] = None
    
    employment_type: EmploymentType = EmploymentType.FULL_TIME
    experience_level: ExperienceLevel = ExperienceLevel.MID
    
    location: Location
    salary_range: Optional[SalaryRange] = None
    
    required_skills: List[str] = Field(default_factory=list)
    preferred_skills: List[str] = Field(default_factory=list)
    nice_to_have_skills: List[str] = Field(default_factory=list)
    
    description: str = Field(..., min_length=50, max_length=5000)
    responsibilities: List[str] = Field(default_factory=list)
    requirements: List[str] = Field(default_factory=list)
    benefits: List[str] = Field(default_factory=list)
    
    application_url: Optional[str] = None
    contact_email: Optional[str] = None
    
    # Metadata
    embedding: Optional[List[float]] = None
    is_active: bool = True
    is_featured: bool = False
    views_count: int = 0
    applications_count: int = 0
    match_count: int = 0
    
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    expires_at: Optional[datetime] = None
    last_matched: Optional[datetime] = None
    
    class Config:
        arbitrary_types_allowed = True
        from_attributes = True
        validate_by_name = True
        use_enum_values = True
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
    
    @validator('required_skills', 'preferred_skills', 'nice_to_have_skills')
    def validate_skills(cls, v):
        return list(set(filter(None, [skill.strip() for skill in v])))
    
    @validator('application_url')
    def validate_url(cls, v):
        if v and not v.startswith(('http://', 'https://')):
            raise ValueError('Application URL must start with http:// or https://')
        return v
    
    @validator('expires_at')
    def validate_expiry(cls, v, values):
        if v and values.get('created_at') and v <= values['created_at']:
            raise ValueError('Expiry date must be after creation date')
        return v

# For backward compatibility
class Job(AdvancedJob):
    """Legacy job model for compatibility."""
    location: Optional[str] = None
    salary_range: Optional[str] = None
    
    @validator('location', pre=True, always=True)
    def format_location(cls, v, values):
        """Format location for backward compatibility."""
        if v is None and values.get('location'):
            loc = values['location']
            if isinstance(loc, Location):
                parts = [loc.city]
                if loc.state:
                    parts.append(loc.state)
                parts.append(loc.country)
                return ", ".join(parts)
        return v
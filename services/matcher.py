# services/matcher.py

import numpy as np
import logging
from typing import List, Dict, Any, Tuple
from datetime import datetime
from enum import Enum
from services.llm_wrapper import llm_wrapper
from config import MatchThreshold, TOP_N_RESULTS, MIN_MATCH_SCORE, ENABLE_SKILL_GAP_ANALYSIS
import re
from collections import Counter

logger = logging.getLogger(__name__)

class MatchLevel(str, Enum):
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"

class AdvancedMatcher:
    """Advanced matching engine with multiple scoring strategies."""
    
    def __init__(self):
        self.skill_synonyms = self._load_skill_synonyms()
        self.industry_keywords = self._load_industry_keywords()
        self.common_skills = [
            "Python", "Java", "JavaScript", "C++", "C#", "Ruby", "Go", "Rust",
            "HTML", "CSS", "React", "Angular", "Vue", "Node.js", "Express",
            "SQL", "MySQL", "PostgreSQL", "MongoDB", "Redis",
            "AWS", "Azure", "GCP", "Docker", "Kubernetes", "Jenkins",
            "Machine Learning", "Deep Learning", "NLP", "Computer Vision",
            "Data Analysis", "Pandas", "NumPy", "TensorFlow", "PyTorch",
            "Agile", "Scrum", "DevOps", "CI/CD", "REST API", "GraphQL"
        ]
    def _load_skill_synonyms(self) -> Dict[str, List[str]]:
        """Load skill synonyms for better matching."""
        return {
            "python": ["python", "py", "python3", "python programming"],
            "javascript": ["javascript", "js", "ecmascript"],
            "java": ["java", "j2ee", "java ee"],
            "react": ["react", "reactjs", "react.js"],
            "aws": ["aws", "amazon web services"],
            # Add more synonyms as needed
        }
    
    def _load_industry_keywords(self) -> Dict[str, List[str]]:
        """Load industry-specific keywords."""
        return {
            "tech": ["software", "developer", "engineer", "programmer"],
            "finance": ["banking", "investment", "financial", "accounting"],
            "healthcare": ["medical", "health", "patient", "clinical"],
            # Add more industries
        }
    def extract_skills(self, text: str) -> List[str]:
        """Extract skills from text (compatibility method)."""
        if not text:
            return []
            
        text_lower = text.lower()
        found_skills = []
        
        for skill in self.common_skills:
            skill_lower = skill.lower()
            # Check for exact matches and variations
            if (skill_lower in text_lower or 
                skill_lower.replace(" ", "") in text_lower.replace(" ", "") or
                any(word in text_lower for word in skill_lower.split())):
                found_skills.append(skill)
                
        return list(set(found_skills))
    
    def calculate_comprehensive_match(self, candidate: Dict, job: Dict) -> Dict:
        """Calculate comprehensive match score with multiple factors."""
        # Multiple scoring components
        embedding_score = self._calculate_embedding_similarity(candidate, job)
        skill_score = self._calculate_skill_match(candidate, job)
        experience_score = self._calculate_experience_match(candidate, job)
        industry_score = self._calculate_industry_match(candidate, job)
        
        # Weighted final score
        weights = {
            'embedding': 0.4,
            'skills': 0.3,
            'experience': 0.2,
            'industry': 0.1
        }
        
        final_score = (
            embedding_score * weights['embedding'] +
            skill_score * weights['skills'] +
            experience_score * weights['experience'] +
            industry_score * weights['industry']
        )
        
        # Get detailed analysis
        analysis = llm_wrapper.analyze_match(candidate, job)
        
        # Skill gap analysis
        skill_gaps = self._analyze_skill_gaps(candidate, job) if ENABLE_SKILL_GAP_ANALYSIS else []
        
        return {
            "final_score": round(final_score, 3),
            "component_scores": {
                "embedding": round(embedding_score, 3),
                "skills": round(skill_score, 3),
                "experience": round(experience_score, 3),
                "industry": round(industry_score, 3)
            },
            "match_level": self._get_match_level(final_score),
            "analysis": analysis,
            "skill_gaps": skill_gaps,
            "timestamp": datetime.now().isoformat()
        }
    
    def _calculate_embedding_similarity(self, candidate: Dict, job: Dict) -> float:
        """Calculate embedding-based similarity."""
        candidate_emb = candidate.get('embedding', [])
        job_emb = job.get('embedding', [])
        
        if not candidate_emb or not job_emb:
            return 0.0
            
        return llm_wrapper._cosine_similarity(candidate_emb, job_emb)
    
    def _calculate_skill_match(self, candidate: Dict, job: Dict) -> float:
        """Calculate skill matching score."""
        candidate_skills = set(self._normalize_skills(candidate.get('skills', [])))
        job_skills = set(self._normalize_skills(job.get('required_skills', [])))
        
        if not job_skills:
            return 0.0
            
        # Exact matches
        exact_matches = candidate_skills.intersection(job_skills)
        
        # Synonym matches
        synonym_matches = set()
        for job_skill in job_skills:
            synonyms = self.skill_synonyms.get(job_skill, [])
            synonym_matches.update(candidate_skills.intersection(synonyms))
        
        all_matches = exact_matches.union(synonym_matches)
        match_ratio = len(all_matches) / len(job_skills)
        
        return min(1.0, match_ratio)
    
    def _calculate_experience_match(self, candidate: Dict, job: Dict) -> float:
        """Calculate experience matching score."""
        candidate_exp = self._extract_experience_years(candidate.get('experience', ''))
        job_exp = job.get('required_experience', 0)
        
        if job_exp == 0:
            return 1.0  # No experience requirement
            
        if candidate_exp >= job_exp:
            return 1.0
        else:
            return candidate_exp / job_exp
    
    def _calculate_industry_match(self, candidate: Dict, job: Dict) -> float:
        """Calculate industry relevance score."""
        candidate_text = f"{candidate.get('experience', '')} {candidate.get('skills', '')}".lower()
        job_text = f"{job.get('description', '')} {job.get('title', '')}".lower()
        
        candidate_industries = self._detect_industries(candidate_text)
        job_industries = self._detect_industries(job_text)
        
        if not job_industries:
            return 1.0
            
        intersection = candidate_industries.intersection(job_industries)
        return len(intersection) / len(job_industries)
    
    def _normalize_skills(self, skills: List[str]) -> List[str]:
        """Normalize skills to lowercase and handle synonyms."""
        normalized = []
        for skill in skills:
            if isinstance(skill, str):
                skill_lower = skill.strip().lower()
                # Check if this skill is a synonym
                for base_skill, synonyms in self.skill_synonyms.items():
                    if skill_lower in synonyms:
                        normalized.append(base_skill)
                        break
                else:
                    normalized.append(skill_lower)
        return normalized
    
    def _extract_experience_years(self, experience_text: str) -> int:
        """Extract years of experience from text."""
        if not experience_text:
            return 0
            
        patterns = [
            r'(\d+)\s* years? experience',
            r'experience.*(\d+)\s* years?',
            r'(\d+)\+ years',
            r'(\d+)\s*yr',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, experience_text.lower())
            if match:
                return int(match.group(1))
                
        return 0
    
    def _detect_industries(self, text: str) -> set:
        """Detect industries mentioned in text."""
        detected = set()
        for industry, keywords in self.industry_keywords.items():
            if any(keyword in text for keyword in keywords):
                detected.add(industry)
        return detected
    
    def _analyze_skill_gaps(self, candidate: Dict, job: Dict) -> List[Dict]:
        """Analyze skill gaps between candidate and job."""
        candidate_skills = set(self._normalize_skills(candidate.get('skills', [])))
        job_skills = set(self._normalize_skills(job.get('required_skills', [])))
        
        missing_skills = job_skills - candidate_skills
        
        return [
            {
                "skill": skill,
                "importance": "high",  # Could be customized
                "suggested_learning": self._get_learning_suggestions(skill)
            }
            for skill in missing_skills
        ]
    
    def _get_learning_suggestions(self, skill: str) -> List[str]:
        """Get learning suggestions for missing skills."""
        suggestions = {
            "python": ["Python Crash Course", "Real Python tutorials", "Codecademy Python"],
            "javascript": ["MDN JavaScript Guide", "JavaScript30 course", "Eloquent JavaScript"],
            "react": ["React Official Tutorial", "Fullstack React", "React documentation"],
            # Add more suggestions
        }
        return suggestions.get(skill, [f"Online courses for {skill}", f"{skill} documentation"])
    
    def _get_match_level(self, score: float) -> MatchLevel:
        """Convert score to match level."""
        if score >= MatchThreshold.EXCELLENT:
            return MatchLevel.EXCELLENT
        elif score >= MatchThreshold.GOOD:
            return MatchLevel.GOOD
        elif score >= MatchThreshold.FAIR:
            return MatchLevel.FAIR
        else:
            return MatchLevel.POOR
        
    def rank_candidate_to_jobs(self, candidate: Dict, jobs: List[Dict], top_n: int = None) -> List[Dict]:
        """Rank jobs for a candidate."""
        results = self.rank_matches(candidate, jobs, "candidate_to_jobs")
        if top_n is not None:
            return results[:top_n]
        return results
    
    def rank_job_to_candidates(self, job: Dict, candidates: List[Dict], top_n: int = None) -> List[Dict]:
        """Rank candidates for a job."""
        results = self.rank_matches(job, candidates, "job_to_candidates")
        if top_n is not None:
            return results[:top_n]
        return results
    
    def rank_matches(self, source_profile: Dict, target_profiles: List[Dict], profile_type: str="candidate_to_jobs") -> List[Dict]:
        """Rank matches between profiles."""
        ranked_matches = []
        
        for target_profile in target_profiles:
            if profile_type == "candidate_to_jobs":
                match_result = self.calculate_comprehensive_match(source_profile, target_profile)
            else:  # job_to_candidates
                match_result = self.calculate_comprehensive_match(target_profile, source_profile)
            
            if match_result['final_score'] >= MIN_MATCH_SCORE:
                ranked_matches.append({
                    "profile_id": target_profile.get('id'),
                    "title": target_profile.get('title', target_profile.get('full_name')),
                    "company": target_profile.get('company', target_profile.get('department', 'N/A')),
                    "score": match_result['final_score'],
                    "match_level": match_result['match_level'],
                    "analysis": match_result['analysis'],
                    "skill_gaps": match_result['skill_gaps']
                })
        
        # Sort by score descending
        ranked_matches.sort(key=lambda x: x['score'], reverse=True)
        return ranked_matches[:TOP_N_RESULTS]

# Singleton instance
matcher = AdvancedMatcher()
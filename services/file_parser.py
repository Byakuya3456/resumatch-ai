# services/file_parser.py

import io
import pdfplumber
import docx2txt
import logging
import re
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import spacy
from spacy import Language
from config import ENABLE_ADVANCED_NER, MAX_RESUME_PAGES, MAX_FILE_SIZE_MB
import requests
from services.llm_wrapper import llm_wrapper

logger = logging.getLogger(__name__)

class AdvancedFileParser:
    """Advanced file parser with AI-powered extraction and validation."""
    
    def __init__(self):
        self.nlp = None
        self._initialize_nlp()
    
    def _initialize_nlp(self):
        """Initialize NLP models for advanced parsing."""
        try:
            if ENABLE_ADVANCED_NER:
                try:
                    self.nlp = spacy.load("en_core_web_sm")
                    logger.info("spaCy model loaded successfully")
                except OSError:
                    logger.warning("spaCy model not found. Installing...")
                    import subprocess
                    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
                    self.nlp = spacy.load("en_core_web_sm")
        except Exception as e:
            logger.warning(f"Failed to initialize spaCy: {e}")
            self.nlp = None
    
    def extract_text(self, file_bytes: bytes, file_ext: str) -> str:
        """Extract text from file with validation and optimization."""
        if len(file_bytes) > MAX_FILE_SIZE_MB * 1024 * 1024:
            raise ValueError(f"File size exceeds {MAX_FILE_SIZE_MB}MB limit")
        
        file_ext = file_ext.lower()
        
        if file_ext == ".pdf":
            return self._extract_text_from_pdf_optimized(file_bytes)
        elif file_ext == ".docx":
            return self._extract_text_from_docx(file_bytes)
        elif file_ext == ".txt":
            return file_bytes.decode('utf-8', errors='ignore')
        else:
            raise ValueError(f"Unsupported file extension: {file_ext}")
    
    def _extract_text_from_pdf_optimized(self, file_bytes: bytes) -> str:
        """Optimized PDF text extraction with page limits."""
        try:
            text_parts = []
            with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
                for i, page in enumerate(pdf.pages[:MAX_RESUME_PAGES]):
                    try:
                        page_text = page.extract_text() or ""
                        if page_text.strip():
                            text_parts.append(page_text)
                    except Exception as e:
                        logger.warning(f"Error extracting page {i+1}: {e}")
                        continue
            
            full_text = "\n".join(text_parts)
            if not full_text.strip():
                raise ValueError("No text could be extracted from PDF")
                
            return full_text
        except Exception as e:
            logger.error(f"PDF extraction failed: {e}")
            raise
    
    def _extract_text_from_docx(self, file_bytes: bytes) -> str:
        """Extract text from DOCX files with error handling."""
        try:
            with io.BytesIO(file_bytes) as bio:
                # Use temp file for better reliability
                import tempfile
                with tempfile.NamedTemporaryFile(delete=True, suffix=".docx") as tmp:
                    tmp.write(bio.read())
                    tmp.flush()
                    text = docx2txt.process(tmp.name)
            
            if not text.strip():
                raise ValueError("No text could be extracted from DOCX")
                
            return text
        except Exception as e:
            logger.error(f"DOCX extraction failed: {e}")
            raise
    
    def extract_structured_data(self, text: str) -> Dict:
        """Extract structured data from resume text using hybrid approach."""
        try:
            # Basic text cleaning
            cleaned_text = self._clean_resume_text(text)
            
            # Extract using multiple methods
            data = {
                "personal_info": self._extract_personal_info(cleaned_text),
                "skills": self._extract_skills_advanced(cleaned_text),
                "experience": self._extract_experience_advanced(cleaned_text),
                "education": self._extract_education_advanced(cleaned_text),
                "certifications": self._extract_certifications(cleaned_text),
                "languages": self._extract_languages(cleaned_text),
            }
            
            # Enhanced extraction using LLM if available
            try:
                llm_data = self._extract_with_llm(cleaned_text)
                data.update(llm_data)
            except Exception as e:
                logger.warning(f"LLM extraction failed: {e}")
            
            return data
        except Exception as e:
            logger.error(f"Structured data extraction failed: {e}")
            return {}
    
    def _clean_resume_text(self, text: str) -> str:
        """Clean and normalize resume text."""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters but keep relevant ones
        text = re.sub(r'[^\w\s@\.\-\+\(\)]', ' ', text)
        return text.strip()
    
    def _extract_personal_info(self, text: str) -> Dict:
        """Extract personal information from text."""
        info = {}
        
        # Email extraction
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        emails = re.findall(email_pattern, text)
        if emails:
            info['email'] = emails[0]
        
        # Phone extraction
        phone_pattern = r'(\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}'
        phones = re.findall(phone_pattern, text)
        if phones:
            info['phone'] = phones[0]
        
        # Name extraction (simple first line)
        lines = text.split('\n')
        if lines and len(lines[0].split()) >= 2:
            info['name'] = lines[0].strip()
        
        return info
    
    def _extract_skills_advanced(self, text: str) -> List[str]:
        """Advanced skill extraction using multiple techniques."""
        skills = set()
        
        # Common skills pattern matching
        common_skills = [
            "python", "java", "javascript", "sql", "aws", "docker", "kubernetes",
            "react", "angular", "vue", "node.js", "express", "django", "flask",
            "machine learning", "deep learning", "nlp", "computer vision",
            "tensorflow", "pytorch", "scikit-learn", "pandas", "numpy",
            "git", "jenkins", "ci/cd", "devops", "agile", "scrum"
        ]
        
        text_lower = text.lower()
        for skill in common_skills:
            if skill in text_lower:
                skills.add(skill.title())
        
        # NLP-based extraction if available
        if self.nlp:
            try:
                doc = self.nlp(text)
                for ent in doc.ents:
                    if ent.label_ in ["SKILL", "TECH"]:
                        skills.add(ent.text.title())
            except Exception as e:
                logger.warning(f"NLP skill extraction failed: {e}")
        
        return sorted(list(skills))
    
    def _extract_experience_advanced(self, text: str) -> List[Dict]:
        """Extract experience information."""
        experience = []
        
        # Simple pattern matching for experience
        experience_patterns = [
            r'(\d+)\s*years?.*experience',
            r'experience.*(\d+)\s*years?',
            r'(\d+)\+ years',
            r'(\d+)\s*yr',
        ]
        
        years = 0
        for pattern in experience_patterns:
            match = re.search(pattern, text.lower())
            if match:
                years = max(years, int(match.group(1)))
                break
        
        # Extract job titles
        job_titles = []
        title_patterns = [
            r'\b(?:senior|junior|lead|principal)?\s*(?:software|data|devops|ml|ai)'
            r'\s*(?:engineer|developer|scientist|analyst|architect)\b',
            r'\b(?:product|project|technical)\s*(?:manager|lead|director)\b',
            r'\b(?:frontend|backend|fullstack|web)\s*(?:developer|engineer)\b'
        ]
        
        for pattern in title_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            job_titles.extend(matches)
        
        return {
            "total_years": years,
            "job_titles": list(set(job_titles))[:5],
            "raw_experience": f"{years} years experience" if years > 0 else "Experience not specified"
        }
    
    def _extract_education_advanced(self, text: str) -> List[Dict]:
        """Extract education information."""
        education = []
        
        # Degree patterns
        degree_patterns = {
            'bachelor': r'\b(?:bachelor|b\.?s\.?|b\.?a\.?|b\.?tech|b\.?e\.?)\b',
            'master': r'\b(?:master|m\.?s\.?|m\.?a\.?|m\.?tech|m\.?e\.?|mba)\b',
            'phd': r'\b(?:ph\.?d|doctorate|d\.?phil)\b'
        }
        
        # University patterns
        university_pattern = r'\b(?:university|college|institute|school)\b'
        
        found_degrees = []
        for degree, pattern in degree_patterns.items():
            if re.search(pattern, text, re.IGNORECASE):
                found_degrees.append(degree.title())
        
        return {
            "degrees": found_degrees,
            "raw_education": ", ".join(found_degrees) if found_degrees else "Education not specified"
        }
    
    def _extract_certifications(self, text: str) -> List[str]:
        """Extract certifications."""
        cert_patterns = [
            r'\b(?:aws|azure|gcp)\s*certified',
            r'\b(?:pmp|scrum|agile)\s*certified',
            r'\b(?:cisco|ccna|ccnp)\b',
            r'\b(?:java|python|javascript)\s*certified'
        ]
        
        certifications = []
        for pattern in cert_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            certifications.extend(matches)
        
        return list(set(certifications))
    
    def _extract_languages(self, text: str) -> List[str]:
        """Extract programming languages."""
        lang_pattern = r'\b(?:python|java|javascript|c\+\+|c#|ruby|go|rust|php|swift|kotlin|typescript)\b'
        return list(set(re.findall(lang_pattern, text, re.IGNORECASE)))
    
    def _extract_with_llm(self, text: str) -> Dict:
        """Use LLM for advanced information extraction."""
        prompt = f"""
        Extract structured information from this resume text:
        
        {text[:2000]}  # Truncate for efficiency
        
        Return JSON with these fields:
        - full_name (string)
        - email (string or null)
        - phone (string or null)
        - skills (list of strings)
        - experience_years (number)
        - education (list of degrees)
        - certifications (list of strings)
        - summary (brief professional summary)
        
        Format the response as valid JSON only.
        """
        
        try:
            response = llm_wrapper.get_enhanced_completion(
                [{"role": "user", "content": prompt}],
                max_tokens=500,
                temperature=0.1
            )
            
            # Extract JSON from response
            if '```json' in response:
                json_str = response.split('```json')[1].split('```')[0].strip()
            else:
                json_str = response.strip()
            
            import json
            return json.loads(json_str)
        except Exception as e:
            logger.warning(f"LLM extraction failed: {e}")
            return {}
    
    def validate_resume_completeness(self, text: str) -> Tuple[bool, List[str]]:
        """Validate if resume contains essential information."""
        issues = []
        
        # Check for essential sections
        essential_keywords = [
            'experience', 'education', 'skills', 'project', 
            'work', 'employment', 'certification'
        ]
        
        text_lower = text.lower()
        missing_sections = []
        for keyword in essential_keywords:
            if keyword not in text_lower:
                missing_sections.append(keyword)
        
        if missing_sections:
            issues.append(f"Missing sections: {', '.join(missing_sections[:3])}")
        
        # Check for contact information
        if not re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text):
            issues.append("Missing email address")
        
        # Check minimum length
        if len(text.split()) < 50:
            issues.append("Resume appears too short")
        
        return len(issues) == 0, issues

# Singleton instance
file_parser = AdvancedFileParser()

# Backward compatibility
class FileParser:
    """Legacy file parser for compatibility."""
    
    @staticmethod
    def extract_text_from_pdf(file_bytes: bytes) -> str:
        return file_parser._extract_text_from_pdf_optimized(file_bytes)
    
    @staticmethod
    def extract_text_from_docx(file_bytes: bytes) -> str:
        return file_parser._extract_text_from_docx(file_bytes)
    
    @staticmethod
    def extract_text(file_bytes: bytes, file_ext: str) -> str:
        return file_parser.extract_text(file_bytes, file_ext)
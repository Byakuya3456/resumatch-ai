# services/llm_wrapper.py

import requests
import logging
import numpy as np
import json
import redis
from typing import List, Dict, Optional, Tuple
from sentence_transformers import SentenceTransformer
from config import (
    LLM_API_URL, LLM_MODEL, EMBEDDING_MODEL, EMBEDDING_DIMENSION,
    MAX_EMBEDDING_TEXT_LENGTH, REDIS_URL, CACHE_TTL, LLM_TIMEOUT,
    LLM_MAX_TOKENS, USE_ML_MODEL, ML_MODEL_PATH
)
import xgboost as xgb
from datetime import datetime

logger = logging.getLogger(__name__)

class AdvancedLMWrapper:
    """Advanced wrapper with hybrid ML+LLM approach and caching."""
    
    def __init__(self):
        self.api_url = LLM_API_URL
        self.model_name = LLM_MODEL
        self.embedding_model = None
        self.ml_model = None
        self.redis_client = None
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize all model components."""
        try:
            # Initialize embedding model
            self.embedding_model = SentenceTransformer(EMBEDDING_MODEL)
            logger.info(f"Loaded embedding model: {EMBEDDING_MODEL}")
            
            # Initialize ML model if enabled
            if USE_ML_MODEL:
                try:
                    self.ml_model = xgb.Booster()
                    self.ml_model.load_model(ML_MODEL_PATH)
                    logger.info(f"Loaded ML model from {ML_MODEL_PATH}")
                except Exception as e:
                    logger.warning(f"Could not load ML model: {e}. Using fallback.")
                    self.ml_model = None
            
            # Initialize Redis cache
            try:
                self.redis_client = redis.from_url(REDIS_URL)
                self.redis_client.ping()
                logger.info("Redis cache connected successfully")
            except Exception as e:
                logger.warning(f"Redis not available: {e}. Using in-memory cache.")
                self.redis_client = None
                
        except Exception as e:
            logger.error(f"Failed to initialize components: {e}")
            raise

    def get_embeddings(self, text: str, use_cache: bool = True) -> List[float]:
        """Get embeddings with caching support."""
        if not text.strip():
            return []
        
        # Truncate text for efficiency
        truncated_text = text[:MAX_EMBEDDING_TEXT_LENGTH]
        
        # Check cache first
        cache_key = f"embedding:{hash(truncated_text)}"
        if use_cache and self.redis_client:
            try:
                cached = self.redis_client.get(cache_key)
                if cached:
                    return json.loads(cached)
            except Exception as e:
                logger.warning(f"Cache read failed: {e}")
        
        # Generate embedding
        try:
            embedding = self.embedding_model.encode(truncated_text).tolist()
            
            # Cache the result
            if use_cache and self.redis_client:
                try:
                    self.redis_client.setex(
                        cache_key, 
                        CACHE_TTL, 
                        json.dumps(embedding)
                    )
                except Exception as e:
                    logger.warning(f"Cache write failed: {e}")
            
            return embedding
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            return []

    def get_enhanced_completion(self, messages: List[Dict], **kwargs) -> str:
        """Enhanced completion with fallback strategies."""
        payload = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": kwargs.get('max_tokens', LLM_MAX_TOKENS),
            "temperature": kwargs.get('temperature', 0.1),
            "top_p": kwargs.get('top_p', 0.9),
            "stream": False
        }
        
        try:
            response = requests.post(
                f"{self.api_url}/chat/completions",
                json=payload,
                timeout=LLM_TIMEOUT
            )
            response.raise_for_status()
            data = response.json()
            return data['choices'][0]['message']['content']
        except Exception as e:
            logger.error(f"LLM API call failed: {e}")
            return self._generate_fallback_response(messages)

    def analyze_match(self, candidate: Dict, job: Dict) -> Dict:
        """Comprehensive match analysis using hybrid approach."""
        # Get embeddings for both profiles
        candidate_text = self._prepare_text_for_embedding(candidate)
        job_text = self._prepare_text_for_embedding(job)
        
        candidate_embedding = self.get_embeddings(candidate_text)
        job_embedding = self.get_embeddings(job_text)
        
        # Calculate similarity
        similarity_score = self._cosine_similarity(candidate_embedding, job_embedding)
        
        # Get detailed analysis from LLM
        llm_analysis = self._get_llm_match_analysis(candidate, job, similarity_score)
        
        # Get ML prediction if available
        ml_prediction = self._get_ml_prediction(candidate, job, similarity_score)
        
        return {
            "similarity_score": round(similarity_score, 3),
            "llm_analysis": llm_analysis,
            "ml_prediction": ml_prediction,
            "timestamp": datetime.now().isoformat(),
            "model_used": self.model_name
        }

    def _prepare_text_for_embedding(self, profile: Dict) -> str:
        """Prepare text for embedding generation."""
        if 'skills' in profile:
            skills_text = ', '.join(profile['skills']) if isinstance(profile['skills'], list) else profile['skills']
        else:
            skills_text = ""
            
        return f"""
        {profile.get('full_name', '')} 
        Skills: {skills_text}
        Experience: {profile.get('experience', '')}
        Education: {profile.get('education', '')}
        Description: {profile.get('description', '')}
        """.strip()

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        if not vec1 or not vec2 or len(vec1) != len(vec2):
            return 0.0
        
        try:
            v1 = np.array(vec1)
            v2 = np.array(vec2)
            return float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
        except Exception as e:
            logger.error(f"Cosine similarity calculation failed: {e}")
            return 0.0

    def _get_llm_match_analysis(self, candidate: Dict, job: Dict, similarity: float) -> Dict:
        """Get detailed analysis from LLM."""
        prompt = self._build_analysis_prompt(candidate, job, similarity)
        
        messages = [
            {
                "role": "system",
                "content": """You are an expert HR analyst. Provide detailed, objective analysis of job-candidate matches.
                Include: strengths, weaknesses, skill gaps, cultural fit, and recommendations."""
            },
            {"role": "user", "content": prompt}
        ]
        
        response = self.get_enhanced_completion(messages)
        return self._parse_llm_response(response)

    def _build_analysis_prompt(self, candidate: Dict, job: Dict, similarity: float) -> str:
        """Build comprehensive analysis prompt."""
        return f"""
        Analyze the match between candidate and job position:

        CANDIDATE PROFILE:
        Name: {candidate.get('full_name', 'N/A')}
        Skills: {', '.join(candidate.get('skills', []))}
        Experience: {candidate.get('experience', 'N/A')}
        Education: {candidate.get('education', 'N/A')}

        JOB PROFILE:
        Title: {job.get('title', 'N/A')}
        Company: {job.get('company', 'N/A')}
        Required Skills: {', '.join(job.get('required_skills', []))}
        Description: {job.get('description', 'N/A')}

        Embedding Similarity Score: {similarity:.3f}

        Provide analysis in this JSON format:
        {{
            "strengths": ["list", "of", "strengths"],
            "weaknesses": ["list", "of", "weaknesses"],
            "skill_gaps": ["missing", "skills"],
            "cultural_fit": "analysis",
            "overall_recommendation": "recommendation with confidence level",
            "suggested_interview_questions": ["question1", "question2"]
        }}
        """

    # In services/llm_wrapper.py - Improve JSON parsing

    def _parse_llm_response(self, response: str) -> Dict:
        """Parse LLM response into structured format with better error handling."""
        if not response:
            return {"raw_analysis": "No analysis provided"}
        
        try:
            # Try to extract JSON from various formats
            json_str = response.strip()
            
            # Remove markdown code blocks if present
            if '```json' in json_str:
                json_str = json_str.split('```json')[1].split('```')[0].strip()
            elif '```' in json_str:
                json_str = json_str.split('```')[1].strip()
            
            # Remove any non-JSON text before or after
            if '{' in json_str and '}' in json_str:
                # Extract the first complete JSON object
                start_idx = json_str.find('{')
                end_idx = json_str.rfind('}') + 1
                json_str = json_str[start_idx:end_idx]
            
            # Parse the JSON
            return json.loads(json_str)
            
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse LLM response as JSON: {e}")
            logger.warning(f"Raw response: {response[:500]}...")
            
            # Fallback: return the raw response in a structured format
            return {
                "raw_analysis": response,
                "strengths": ["Analysis available in raw_analysis field"],
                "weaknesses": [],
                "skill_gaps": [],
                "recommendations": ["View raw_analysis for complete details"]
            }
        except Exception as e:
            logger.error(f"Unexpected error parsing LLM response: {e}")
            return {"raw_analysis": response}

    def _get_ml_prediction(self, candidate: Dict, job: Dict, similarity: float) -> Optional[Dict]:
        """Get ML model prediction if available."""
        if not self.ml_model:
            return None
        
        try:
            # Prepare features for ML model
            features = self._prepare_ml_features(candidate, job, similarity)
            prediction = self.ml_model.predict(xgb.DMatrix([features]))[0]
            
            return {
                "match_probability": float(prediction),
                "recommendation": "Recommended" if prediction > 0.5 else "Not Recommended",
                "confidence": abs(prediction - 0.5) * 2  # Convert to 0-1 confidence
            }
        except Exception as e:
            logger.error(f"ML prediction failed: {e}")
            return None

    def _prepare_ml_features(self, candidate: Dict, job: Dict, similarity: float) -> List[float]:
        """Prepare features for ML model."""
        # This would be customized based on your trained model
        return [
            similarity,
            len(candidate.get('skills', [])),
            len(job.get('required_skills', [])),
            # Add more features as needed
        ]

    def _generate_fallback_response(self, messages: List[Dict]) -> str:
        """Generate fallback response when LLM is unavailable."""
        user_content = messages[-1]['content'] if messages else ""
        
        if "analysis" in user_content.lower():
            return json.dumps({
                "strengths": ["Fallback analysis - LLM unavailable"],
                "weaknesses": ["Cannot analyze without LLM"],
                "recommendation": "Please try again later"
            })
        
        return "I'm currently unavailable. Please try again later."

# Singleton instance
llm_wrapper = AdvancedLMWrapper()
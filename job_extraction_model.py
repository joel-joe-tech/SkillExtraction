import json
import re
import sqlite3
import logging
from typing import Dict, List, Any, Optional, Tuple
import spacy
from spacy.tokens import Doc
from spacy.matcher import PhraseMatcher, Matcher
from collections import defaultdict
import os
from spacy.language import Language

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("job_extraction.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Load spaCy model - using the medium model for better accuracy
try:
    nlp = spacy.load("en_core_web_md")
    logger.info("Loaded spaCy model: en_core_web_md")
except OSError:
    logger.warning("Downloading spaCy model en_core_web_md...")
    try:
        # Try downloading using spacy CLI
        spacy.cli.download("en_core_web_md")
        nlp = spacy.load("en_core_web_md")
        logger.info("Loaded spaCy model: en_core_web_md")
    except Exception as e:
        logger.error(f"Error downloading spaCy model: {str(e)}")
        logger.warning("Falling back to small spaCy model...")
        try:
            # Try loading the small model as fallback
            nlp = spacy.load("en_core_web_sm")
            logger.info("Loaded fallback spaCy model: en_core_web_sm")
        except OSError:
            # Try downloading the small model
            logger.warning("Downloading fallback spaCy model en_core_web_sm...")
            spacy.cli.download("en_core_web_sm")
            nlp = spacy.load("en_core_web_sm")
            logger.info("Loaded fallback spaCy model: en_core_web_sm")

# Custom component to add to the spaCy pipeline
class SkillMatcher:
    def __init__(self, nlp, skill_patterns=None):
        self.matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
        self.skill_categories = defaultdict(str)
        
        if skill_patterns:
            self.add_patterns(nlp, skill_patterns)
            
    def add_patterns(self, nlp, patterns):
        for category, skills in patterns.items():
            patterns = [nlp.make_doc(text) for text in skills]
            self.matcher.add(category, patterns)
            for skill in skills:
                self.skill_categories[skill.lower()] = category
                
    def __call__(self, doc):
        matches = self.matcher(doc)
        
        # Create a spans object with matched skills
        skill_spans = []
        matched_skills = set()  # Keep track of already matched skills to avoid duplicates
        
        for match_id, start, end in matches:
            # Get the matched skill and its category
            match_span = doc[start:end]
            skill_text = match_span.text.lower()
            
            # Skip duplicates
            if skill_text in matched_skills:
                continue
                
            matched_skills.add(skill_text)
            category = nlp.vocab.strings[match_id]
            
            # Add metadata: category
            span = spacy.tokens.Span(doc, start, end, label=category)
            skill_spans.append(span)
            
        # Also check for common skill signals in text
        skill_signal_patterns = [
            r'(?:knowledge|experience|proficiency|expertise)\s+(?:in|with|of)\s+([A-Za-z0-9\+\.\#\-\s]+?)(?:,|\.|and|;|$)',
            r'(?:familiar|fluent)\s+(?:in|with)\s+([A-Za-z0-9\+\.\#\-\s]+?)(?:,|\.|and|;|$)',
            r'skills?:\s*([A-Za-z0-9\+\.\#\-\s,]+?)(?:\.|;|$)',
            r'(?:strong|excellent)\s+([A-Za-z0-9\+\.\#\-\s]+?)\s+skills'
        ]
        
        text = doc.text.lower()
        for pattern in skill_signal_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                if len(match.groups()) > 0:
                    skill_text = match.group(1).strip()
                    # Simple heuristic to avoid sentences
                    if len(skill_text.split()) <= 5 and skill_text not in matched_skills:
                        # Find the position in the original doc
                        start_pos = text.find(skill_text.lower())
                        if start_pos >= 0:
                            end_pos = start_pos + len(skill_text)
                            # Find closest token indices
                            start_token = 0
                            end_token = 0
                            for i, token in enumerate(doc):
                                if token.idx <= start_pos and (i == len(doc) - 1 or doc[i+1].idx > start_pos):
                                    start_token = i
                                if token.idx + len(token) >= end_pos:
                                    end_token = i + 1
                                    break
                            
                            if start_token < end_token:
                                # Assign the best matching category
                                best_category = "PROGRAMMING"  # Default category
                                span = spacy.tokens.Span(doc, start_token, end_token, label=best_category)
                                skill_spans.append(span)
                                matched_skills.add(skill_text.lower())
            
        # Add the matched skills to the document as a SpanGroup
        doc.spans["skills"] = skill_spans
        return doc

class RelationshipExtractor:
    def __init__(self, nlp):
        # Pattern for company-job relationships
        self.company_patterns = [
            [{"LOWER": {"IN": ["at", "with", "for"]}}, 
             {"OP": "?", "POS": "DET"}, 
             {"OP": "*", "POS": "ADJ"}, 
             {"OP": "+", "ENT_TYPE": "ORG"}]
        ]
        
        # Pattern for job-location relationships
        self.location_patterns = [
            [{"LOWER": {"IN": ["in", "at", "near"]}}, 
             {"OP": "?", "POS": "DET"}, 
             {"OP": "+", "ENT_TYPE": "GPE"}],
            [{"LOWER": "remote"}, 
             {"OP": "?", "LOWER": "work"}]
        ]
        
        # Initialize matchers
        self.company_matcher = Matcher(nlp.vocab)
        self.company_matcher.add("COMPANY_RELATION", self.company_patterns)
        
        self.location_matcher = Matcher(nlp.vocab)
        self.location_matcher.add("LOCATION_RELATION", self.location_patterns)
    
    def __call__(self, doc):
        # Extract company relationships
        company_matches = self.company_matcher(doc)
        location_matches = self.location_matcher(doc)
        
        relationships = []
        
        # Process company relationships
        for match_id, start, end in company_matches:
            span = doc[start:end]
            for ent in span.ents:
                if ent.label_ == "ORG":
                    relationships.append({
                        "type": "WORKS_AT",
                        "entity1": "JOB",
                        "entity2": ent.text,
                        "confidence": 0.85
                    })
        
        # Process location relationships
        for match_id, start, end in location_matches:
            span = doc[start:end]
            # Check if this is a "remote" pattern
            if "remote" in span.text.lower():
                relationships.append({
                    "type": "LOCATED_IN",
                    "entity1": "JOB",
                    "entity2": "REMOTE",
                    "confidence": 0.9
                })
            else:
                for ent in span.ents:
                    if ent.label_ == "GPE":
                        relationships.append({
                            "type": "LOCATED_IN",
                            "entity1": "JOB",
                            "entity2": ent.text,
                            "confidence": 0.8
                        })
        
        # Add relationships as document extension
        doc._.relationships = relationships
        return doc

# Define categories and keywords for skills
SKILL_PATTERNS = {
    "PROGRAMMING": [
        "python", "java", "javascript", "typescript", "c++", "c#", "ruby", "php", 
        "swift", "kotlin", "rust", "go", "scala", "perl", "r", "matlab", "shell",
        "bash", "powershell", "objective-c", "groovy", "vba", "lua", "fortran",
        "cobol", "julia", "erlang", "haskell", "clojure", "lisp", "prolog",
        "code", "coding", "programming", "development", "software engineering"
    ],
    "WEB_TECH": [
        "html", "css", "react", "angular", "vue", "node.js", "nodejs", "django", 
        "flask", "spring", ".net", "express", "rails", "laravel", "wordpress",
        "frontend", "backend", "full stack", "web development", "rest api",
        "aspnet", "bootstrap", "jquery", "sass", "less", "webpack", "babel",
        "responsive design", "web services", "spa", "pwa", "web application",
        "microservices", "web api", "http", "front-end", "back-end"
    ],
    "DATA_SCIENCE": [
        "machine learning", "ml", "deep learning", "dl", "nlp", 
        "natural language processing", "computer vision", "data science", 
        "tensorflow", "pytorch", "keras", "scikit-learn", "pandas", "numpy",
        "data analysis", "data mining", "statistics", "a/b testing", 
        "predictive modeling", "regression", "classification", "clustering",
        "neural networks", "analytics", "big data", "data processing",
        "data cleansing", "etl", "data pipeline", "data visualization",
        "matplotlib", "tableau", "power bi", "data modeling", "hadoop"
    ],
    "DATABASE": [
        "sql", "mysql", "postgresql", "mongodb", "elasticsearch", "redis", 
        "graphql", "neo4j", "snowflake", "redshift", "bigquery", "oracle",
        "database", "data warehouse", "rdbms", "nosql", "sql server", "sqlite",
        "mariadb", "dynamodb", "cassandra", "couchdb", "firebase", "cosmosdb",
        "dax", "rds", "acid", "crud", "query", "stored procedure", "trigger"
    ],
    "CLOUD": [
        "aws", "azure", "gcp", "docker", "kubernetes", "jenkins", "ci/cd", 
        "terraform", "ansible", "puppet", "chef", "serverless", "cloud",
        "devops", "infrastructure", "iaas", "paas", "saas", "deployment",
        "containers", "virtual machine", "vm", "ec2", "s3", "lambda",
        "microservices", "orchestration", "configuration management",
        "load balancer", "autoscaling", "high availability", "fault tolerance"
    ],
    "AI_LLM": [
        "llm", "rag", "generative ai", "langchain", "openai", 
        "prompt engineering", "vector databases", "transformers", "bert", "gpt",
        "chatbot", "gpt-3", "gpt-4", "llama", "mistral", "claude", "embedding",
        "fine-tuning", "ai", "artificial intelligence", "gemini", "text generation",
        "semantic search", "language model", "tokenization", "llm application"
    ],
    "LANGUAGE": [
        "english", "spanish", "italian", "german", "french", "portuguese", "russian",
        "chinese", "japanese", "korean", "arabic", "hindi", "bengali", "hebrew",
        "multilingual", "bilingual", "language skills", "native speaker", "fluent",
        "proficient", "translation", "localization", "interpreter", "linguistics"
    ],
    "SOFT_SKILLS": [
        "communication", "teamwork", "leadership", "problem solving", 
        "critical thinking", "time management", "adaptability", "creativity",
        "collaboration", "presentation", "interpersonal", "negotiation",
        "organizational", "detail-oriented", "project management", "agile",
        "scrum", "multitasking", "self-motivated", "analytical thinking",
        "decision making", "conflict resolution", "emotional intelligence"
    ]
}

# Add custom components to pipeline
if not Doc.has_extension("relationships"):
    Doc.set_extension("relationships", default=[])

# Create skill matcher and relationship extractor objects
# These will be used directly rather than as pipeline components
skill_matcher = SkillMatcher(nlp, SKILL_PATTERNS)
relationship_extractor = RelationshipExtractor(nlp)

# Remove attempts to add to pipeline which is causing compatibility issues

class JobDatabase:
    """Manages the relational database for job data"""
    
    def __init__(self, db_path="jooble_jobs.db"):
        self.db_path = db_path
        self.conn = None
        self.cursor = None
        
        # Delete existing database if it exists to ensure schema is up to date
        if os.path.exists(db_path):
            try:
                os.remove(db_path)
                logger.info(f"Removed existing database to ensure schema compatibility: {db_path}")
            except OSError as e:
                logger.warning(f"Could not remove existing database: {str(e)}")
        
        self.initialize_db()
        
    def initialize_db(self):
        """Initialize the database and create tables if they don't exist"""
        try:
            self.conn = sqlite3.connect(self.db_path)
            self.cursor = self.conn.cursor()
            
            # Create tables
            self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS jobs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                jooble_id TEXT UNIQUE,
                title TEXT NOT NULL,
                company TEXT,
                location TEXT,
                job_type TEXT,
                snippet TEXT,
                salary_min REAL,
                salary_max REAL,
                salary_currency TEXT,
                source TEXT,
                link TEXT,
                updated TEXT
            )
            ''')
            
            self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS skills (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE,
                category TEXT
            )
            ''')
            
            self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS job_skills (
                job_id INTEGER,
                skill_id INTEGER,
                confidence REAL,
                PRIMARY KEY (job_id, skill_id),
                FOREIGN KEY (job_id) REFERENCES jobs (id),
                FOREIGN KEY (skill_id) REFERENCES skills (id)
            )
            ''')
            
            self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS relationships (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                job_id INTEGER,
                relationship_type TEXT,
                entity_text TEXT,
                confidence REAL,
                FOREIGN KEY (job_id) REFERENCES jobs (id)
            )
            ''')
            
            self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS job_qualities (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                job_id INTEGER,
                quality_type TEXT,
                description TEXT,
                confidence REAL,
                FOREIGN KEY (job_id) REFERENCES jobs (id)
            )
            ''')
            
            self.conn.commit()
            logger.info(f"Database initialized: {self.db_path}")
            
        except Exception as e:
            logger.error(f"Database initialization error: {str(e)}")
            if self.conn:
                self.conn.close()
            raise
            
    def close(self):
        """Close the database connection"""
        if self.conn:
            self.conn.close()
            
    def add_job(self, job_data):
        """Add a job to the database and return its ID"""
        try:
            # Check if job already exists
            self.cursor.execute("SELECT id FROM jobs WHERE jooble_id = ?", (job_data["jooble_id"],))
            existing_job = self.cursor.fetchone()
            
            if existing_job:
                logger.info(f"Job ID {job_data['jooble_id']} already exists in database.")
                return existing_job[0]
            
            # Insert job data
            self.cursor.execute('''
            INSERT INTO jobs (jooble_id, title, company, location, job_type, snippet, 
                             salary_min, salary_max, salary_currency, source, link, updated)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                job_data["jooble_id"],
                job_data["title"],
                job_data["company"],
                job_data["location"],
                job_data["job_type"],
                job_data["snippet"],
                job_data["salary_min"],
                job_data["salary_max"],
                job_data["salary_currency"],
                job_data["source"],
                job_data["link"],
                job_data["updated"]
            ))
            
            self.conn.commit()
            job_id = self.cursor.lastrowid
            logger.info(f"Added job to database with ID: {job_id}")
            return job_id
            
        except Exception as e:
            logger.error(f"Error adding job to database: {str(e)}")
            self.conn.rollback()
            return None
            
    def add_skill(self, skill_name, category):
        """Add a skill to the database if it doesn't exist and return its ID"""
        try:
            # Check if skill already exists
            self.cursor.execute("SELECT id FROM skills WHERE name = ?", (skill_name,))
            existing_skill = self.cursor.fetchone()
            
            if existing_skill:
                return existing_skill[0]
            
            # Insert skill
            self.cursor.execute('''
            INSERT INTO skills (name, category) VALUES (?, ?)
            ''', (skill_name, category))
            
            self.conn.commit()
            return self.cursor.lastrowid
            
        except Exception as e:
            logger.error(f"Error adding skill to database: {str(e)}")
            self.conn.rollback()
            return None
            
    def link_job_skill(self, job_id, skill_id, confidence=0.8):
        """Create a link between a job and skill with confidence score"""
        try:
            self.cursor.execute('''
            INSERT OR IGNORE INTO job_skills (job_id, skill_id, confidence)
            VALUES (?, ?, ?)
            ''', (job_id, skill_id, confidence))
            
            self.conn.commit()
            
        except Exception as e:
            logger.error(f"Error linking job and skill: {str(e)}")
            self.conn.rollback()
            
    def add_relationship(self, job_id, relationship_type, entity_text, confidence=0.8):
        """Add a relationship for a job"""
        try:
            self.cursor.execute('''
            INSERT INTO relationships (job_id, relationship_type, entity_text, confidence)
            VALUES (?, ?, ?, ?)
            ''', (job_id, relationship_type, entity_text, confidence))
            
            self.conn.commit()
            
        except Exception as e:
            logger.error(f"Error adding relationship: {str(e)}")
            self.conn.rollback()
            
    def add_job_quality(self, job_id, quality_type, description, confidence=0.8):
        """Add a job quality (requirement, responsibility, benefit, etc.)"""
        try:
            self.cursor.execute('''
            INSERT INTO job_qualities (job_id, quality_type, description, confidence)
            VALUES (?, ?, ?, ?)
            ''', (job_id, quality_type, description, confidence))
            
            self.conn.commit()
            
        except Exception as e:
            logger.error(f"Error adding job quality: {str(e)}")
            self.conn.rollback()

class JobPostExtractor:
    """Extracts structured information from job posts using ML techniques"""
    
    def __init__(self):
        self.db = JobDatabase()
        
    def close(self):
        """Clean up resources"""
        self.db.close()
        
    def extract_salary(self, text: str) -> Dict[str, Any]:
        """Extract salary information using regex patterns"""
        if not text:
            return {'min': None, 'max': None, 'currency': 'USD'}
        
        # Common patterns for salary information
        patterns = [
            r'\$(\d+[.,]?\d*)[kK]?\s*-\s*\$(\d+[.,]?\d*)[kK]?',
            r'(\d+[.,]?\d*)[kK]?\s*-\s*(\d+[.,]?\d*)[kK]?\s*USD',
            r'salary\s+range.*?\$(\d+[.,]?\d*)[kK]?(?:\s*-\s*|\s+to\s+)\$?(\d+[.,]?\d*)[kK]?',
            r'up to \$?(\d+[.,]?\d*)[kK]?',
            r'from \$?(\d+[.,]?\d*)[kK]?'
        ]
        
        for pattern in patterns:
            matches = re.search(pattern, text, re.IGNORECASE)
            if matches:
                # Handle "up to" pattern
                if 'up to' in pattern:
                    max_salary = self._parse_salary(matches.group(1))
                    min_salary = max_salary * 0.7  # Estimate
                    return {'min': min_salary, 'max': max_salary, 'currency': 'USD'}
                # Handle "from" pattern
                elif 'from' in pattern and len(matches.groups()) == 1:
                    min_salary = self._parse_salary(matches.group(1))
                    max_salary = min_salary * 1.3  # Estimate
                    return {'min': min_salary, 'max': max_salary, 'currency': 'USD'}
                else:
                    min_salary = self._parse_salary(matches.group(1))
                    max_salary = self._parse_salary(matches.group(2))
                    return {'min': min_salary, 'max': max_salary, 'currency': 'USD'}
        
        return {'min': None, 'max': None, 'currency': 'USD'}
        
    def _parse_salary(self, salary_str: str) -> float:
        """Parse salary string to float, handling K multiplier"""
        num = float(salary_str.replace(',', ''))
        if 'k' in salary_str.lower():
            num *= 1000
        return num
        
    def extract_job_qualities(self, text: str) -> Dict[str, List[Dict[str, Any]]]:
        """Extract job qualities like requirements and responsibilities"""
        if not text:
            return {
                'requirements': [],
                'responsibilities': [],
                'benefits': []
            }
        
        # Detect sections in the job description
        requirements_pattern = r'(?:requirements|qualifications|what you\'ll need)(?::|.*?:)?(.+?)(?:(?:responsibilities|what you\'ll do|benefits|about us|apply now)(?::|.*?:)?|$)'
        responsibilities_pattern = r'(?:responsibilities|what you\'ll do|role|duties)(?::|.*?:)?(.+?)(?:(?:requirements|qualifications|benefits|about us|apply now)(?::|.*?:)?|$)'
        benefits_pattern = r'(?:benefits|what we offer|perks|package|advantages)(?::|.*?:)?(.+?)(?:(?:requirements|qualifications|responsibilities|about us|apply now)(?::|.*?:)?|$)'
        
        # Extract sections
        requirements_match = re.search(requirements_pattern, text, re.IGNORECASE | re.DOTALL)
        responsibilities_match = re.search(responsibilities_pattern, text, re.IGNORECASE | re.DOTALL)
        benefits_match = re.search(benefits_pattern, text, re.IGNORECASE | re.DOTALL)
        
        # Process sections to extract individual items (using bullet points or numbered lists)
        requirements = self._extract_list_items(requirements_match.group(1) if requirements_match else "")
        responsibilities = self._extract_list_items(responsibilities_match.group(1) if responsibilities_match else "")
        benefits = self._extract_list_items(benefits_match.group(1) if benefits_match else "")
        
        return {
            'requirements': [{'description': item, 'confidence': 0.8} for item in requirements],
            'responsibilities': [{'description': item, 'confidence': 0.8} for item in responsibilities],
            'benefits': [{'description': item, 'confidence': 0.8} for item in benefits]
        }
        
    def _extract_list_items(self, text: str) -> List[str]:
        """Extract items from a bulleted or numbered list"""
        if not text:
            return []
            
        # Common bullet point patterns
        bullet_patterns = [
            r'•\s+(.*?)(?=(?:•|\d+\.\s+|\n\n|$))',
            r'★\s+(.*?)(?=(?:★|\d+\.\s+|\n\n|$))',
            r'✓\s+(.*?)(?=(?:✓|\d+\.\s+|\n\n|$))',
            r'-\s+(.*?)(?=(?:-|\d+\.\s+|\n\n|$))',
            r'\*\s+(.*?)(?=(?:\*|\d+\.\s+|\n\n|$))',
            r'\d+\.\s+(.*?)(?=(?:•|\d+\.\s+|\n\n|$))'
        ]
        
        items = []
        # Try to extract bullet points
        for pattern in bullet_patterns:
            matches = re.findall(pattern, text, re.DOTALL)
            if matches:
                items.extend([item.strip() for item in matches if item.strip()])
                
        # If no bullet points found, try to split by newlines
        if not items:
            lines = [line.strip() for line in text.split('\n') if line.strip()]
            # Only use if lines are relatively short (likely list items)
            items = [line for line in lines if len(line) < 200]
            
        return items
    
    def extract_info_from_job(self, job_post: Dict[str, Any]) -> Dict[str, Any]:
        """Extract structured information from a job post using NLP"""
        # Get the text to analyze - combine title, snippet and description for better extraction
        title = job_post.get('title', '')
        snippet = job_post.get('snippet', '')
        description = job_post.get('description', '')
        
        # Combine texts with appropriate weighting (duplicate important fields)
        analysis_text = f"{title}\n{title}\n{snippet}\n{description}"
        
        # Basic job information
        job_info = {
            "jooble_id": str(job_post.get("id", "")),
            "title": title,
            "company": job_post.get("company", ""),
            "location": job_post.get("location", ""),
            "job_type": job_post.get("type", ""),
            "snippet": snippet,
            "source": job_post.get("source", ""),
            "link": job_post.get("link", ""),
            "updated": job_post.get("updated", ""),
            # Default values for fields that might not be in the API response
            "salary_min": None,
            "salary_max": None,
            "salary_currency": "USD"
        }
        
        # Extract salary information from combined text
        salary = self.extract_salary(analysis_text)
        if salary["min"] is not None:
            job_info["salary_min"] = salary["min"]
        if salary["max"] is not None:
            job_info["salary_max"] = salary["max"]
        if salary["currency"] is not None:
            job_info["salary_currency"] = salary["currency"]
        
        # Process text with spaCy - use combined text for better extraction
        doc = nlp(analysis_text)
        
        # Use skill matcher directly without pipeline
        doc = skill_matcher(doc)
        doc = relationship_extractor(doc)
        
        # Extract skills (using the custom skill matcher)
        skills = []
        for skill_span in doc.spans.get("skills", []):
            skill_name = skill_span.text
            skill_category = skill_span.label_
            skills.append({
                "name": skill_name,
                "category": skill_category,
                "confidence": 0.85
            })
        
        # Extract relationships
        relationships = getattr(doc._, "relationships", [])
        
        # Extract job qualities (requirements, responsibilities, benefits)
        job_qualities = self.extract_job_qualities(analysis_text)
        
        return {
            "job_info": job_info,
            "skills": skills,
            "relationships": relationships,
            "job_qualities": job_qualities
        }
        
    def process_job_post(self, job_post: Dict[str, Any]) -> int:
        """Process a single job post and add to database"""
        try:
            # Extract info using our ML model
            extracted_data = self.extract_info_from_job(job_post)
            
            # Add job to database
            job_id = self.db.add_job(extracted_data["job_info"])
            if not job_id:
                logger.error(f"Failed to add job to database: {job_post.get('title')}")
                return None
                
            # Add skills
            for skill in extracted_data["skills"]:
                skill_id = self.db.add_skill(skill["name"], skill["category"])
                if skill_id:
                    self.db.link_job_skill(job_id, skill_id, skill["confidence"])
                    
            # Add relationships
            for rel in extracted_data["relationships"]:
                self.db.add_relationship(
                    job_id, 
                    rel["type"], 
                    rel["entity2"],
                    rel["confidence"]
                )
                
            # Add job qualities
            for quality_type, items in extracted_data["job_qualities"].items():
                for item in items:
                    self.db.add_job_quality(
                        job_id,
                        quality_type,
                        item["description"],
                        item["confidence"]
                    )
                    
            logger.info(f"Successfully processed job: {job_post.get('title')}")
            return job_id
            
        except Exception as e:
            logger.error(f"Error processing job post: {str(e)}")
            return None
            
    def process_jooble_api_response(self, api_response: str) -> List[int]:
        """Process the full Jooble API response and return added job IDs"""
        try:
            # Parse the API response
            if isinstance(api_response, str):
                response_data = json.loads(api_response)
            else:
                response_data = api_response
                
            # Extract jobs from the response
            jobs = response_data.get("jobs", [])
            logger.info(f"Processing {len(jobs)} jobs from Jooble API response")
            
            job_ids = []
            for job in jobs:
                job_id = self.process_job_post(job)
                if job_id:
                    job_ids.append(job_id)
                    
            logger.info(f"Successfully processed {len(job_ids)} out of {len(jobs)} jobs")
            return job_ids
            
        except Exception as e:
            logger.error(f"Error processing API response: {str(e)}")
            return []

def process_jooble_response(response_text: str) -> Dict[str, Any]:
    """Process a Jooble API response and return stats"""
    extractor = JobPostExtractor()
    try:
        job_ids = extractor.process_jooble_api_response(response_text)
        
        stats = {
            "processed_jobs": len(job_ids),
            "success": len(job_ids) > 0,
            "job_ids": job_ids
        }
        
        return stats
    finally:
        extractor.close()

if __name__ == "__main__":
    # If run directly, try to process from test.py output
    if os.path.exists("test_output.json"):
        with open("test_output.json", "r") as f:
            response_data = f.read()
        stats = process_jooble_response(response_data)
        print(f"Processing complete: {stats}")
    else:
        print("No test output file found. Please run test.py first.") 
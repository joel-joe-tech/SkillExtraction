import requests
import pandas as pd
import json
import re
from sqlalchemy import create_engine, Column, Integer, String, ForeignKey, Table, Text, Boolean, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker
import spacy
import nltk
from nltk.tokenize import sent_tokenize
import time
import logging
from tqdm import tqdm
from sqlalchemy.sql import func, desc

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("job_extraction.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Download necessary NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    logger.info("NLTK punkt tokenizer already downloaded")
except LookupError:
    logger.info("Downloading NLTK punkt tokenizer...")
    nltk.download('punkt', quiet=False)
    logger.info("NLTK punkt tokenizer download complete")

# Load spaCy model (use medium model for better balance of speed and accuracy)
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

# Define the SQLAlchemy Base
Base = declarative_base()

# Association tables for many-to-many relationships
job_skill = Table(
    'job_skill', Base.metadata,
    Column('job_id', Integer, ForeignKey('jobs.id')),
    Column('skill_id', Integer, ForeignKey('skills.id'))
)

job_qualification = Table(
    'job_qualification', Base.metadata,
    Column('job_id', Integer, ForeignKey('jobs.id')),
    Column('qualification_id', Integer, ForeignKey('qualifications.id'))
)

job_responsibility = Table(
    'job_responsibility', Base.metadata,
    Column('job_id', Integer, ForeignKey('jobs.id')),
    Column('responsibility_id', Integer, ForeignKey('responsibilities.id'))
)

# Define database models
class Job(Base):
    __tablename__ = 'jobs'
    
    id = Column(Integer, primary_key=True)
    jooble_id = Column(String(100), unique=True)
    title = Column(String(255), nullable=False)
    company = Column(String(255))
    location = Column(String(255))
    job_type = Column(String(100))
    experience_level = Column(String(100))
    salary_min = Column(Float)
    salary_max = Column(Float)
    salary_currency = Column(String(10))
    description = Column(Text)
    snippet = Column(Text)
    posting_date = Column(String(50))
    updated_date = Column(String(50))
    source_url = Column(String(500))
    
    # Relationships
    skills = relationship("Skill", secondary=job_skill, back_populates="jobs")
    qualifications = relationship("Qualification", secondary=job_qualification, back_populates="jobs")
    responsibilities = relationship("Responsibility", secondary=job_responsibility, back_populates="jobs")

class Skill(Base):
    __tablename__ = 'skills'
    
    id = Column(Integer, primary_key=True)
    name = Column(String(255), nullable=False, unique=True)
    category = Column(String(100))  # technical, soft, tool, etc.
    
    # Relationships
    jobs = relationship("Job", secondary=job_skill, back_populates="skills")

class Qualification(Base):
    __tablename__ = 'qualifications'
    
    id = Column(Integer, primary_key=True)
    description = Column(Text, nullable=False, unique=True)
    degree_required = Column(Boolean, default=False)
    years_experience = Column(Integer, nullable=True)
    
    # Relationships
    jobs = relationship("Job", secondary=job_qualification, back_populates="qualifications")

class Responsibility(Base):
    __tablename__ = 'responsibilities'
    
    id = Column(Integer, primary_key=True)
    description = Column(Text, nullable=False, unique=True)
    
    # Relationships
    jobs = relationship("Job", secondary=job_responsibility, back_populates="responsibilities")

# Create database engine and session
def create_db_session(db_url="sqlite:///jooble_jobs.db"):
    logger.info(f"Creating database connection to: {db_url}")
    engine = create_engine(db_url)
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    return Session()

# Function to fetch job posts from Jooble API
def fetch_jooble_jobs(api_key, keywords="data scientist", location="", page=1, limit=20):
    """
    Fetch job listings from Jooble API
    
    Parameters:
    api_key (str): Jooble API key
    keywords (str): Job keywords to search for
    location (str): Location to search in
    page (int): Page number for pagination
    limit (int): Number of results per page
    
    Returns:
    dict: JSON response from Jooble API
    """
    url = "https://jooble.org/api/"
    
    payload = {
        "keywords": keywords,
        "location": location,
        "page": page,
        "limit": limit
    }
    
    headers = {
        "Content-Type": "application/json"
    }
    
    try:
        logger.info(f"Requesting Jooble API: keywords={keywords}, location={location}, page={page}")
        response = requests.post(url + api_key, json=payload, headers=headers)
        response.raise_for_status()
        data = response.json()
        
        jobs = data.get('jobs', [])
        logger.info(f"Received {len(jobs)} jobs from Jooble API")
        
        # Log a sample job for debugging response format
        if jobs and page == 1:
            sample_job = jobs[0]
            logger.info("Sample job fields available: " + ", ".join(sample_job.keys()))
            logger.info(f"Sample job title: {sample_job.get('title', 'N/A')}")
            logger.info(f"Sample job has description: {bool(sample_job.get('description'))}")
            description = sample_job.get('description', '')
            if description:
                logger.info(f"Description starts with: {description[:100]}...")
            else:
                logger.info("Description is empty")
        
        return data
    except requests.exceptions.RequestException as e:
        logger.error(f"API request failed: {str(e)}")
        if response.status_code == 429:
            logger.warning("Rate limit exceeded. Waiting before retrying...")
            time.sleep(60)  # Wait for a minute before retrying
            return fetch_jooble_jobs(api_key, keywords, location, page, limit)
        else:
            raise Exception(f"API request failed: {str(e)}")

# Extract salary information using regex patterns
def extract_salary_info(text):
    """Extract salary information from job description text"""
    if not text:
        return {'min': None, 'max': None, 'currency': None}
    
    # Common patterns for salary information
    patterns = [
        r'\$(\d+[.,]?\d*)\s*-\s*\$(\d+[.,]?\d*)\s*(?:k|K)?(?:/(?:yr|year|annual))?',
        r'(\d+[.,]?\d*)\s*-\s*(\d+[.,]?\d*)\s*(?:k|K)?(?:\s*USD)?(?:/(?:yr|year|annual))?',
        r'(?:salary|pay)(?:\s+range)?(?:\s+is)?(?:\s+from)?\s+\$?(\d+[.,]?\d*)(?:k|K)?(?:\s*-\s*|(?:\s+to\s+))?\$?(\d+[.,]?\d*)(?:k|K)?',
        r'up to \$?(\d+[.,]?\d*)(?:k|K)?',
        r'from \$?(\d+[.,]?\d*)(?:k|K)?'
    ]
    
    for pattern in patterns:
        matches = re.search(pattern, text, re.IGNORECASE)
        if matches:
            # Handle "up to" pattern
            if 'up to' in pattern:
                max_salary = float(matches.group(1).replace(',', ''))
                min_salary = max_salary * 0.7  # Estimate min as 70% of max
            # Handle "from" pattern
            elif 'from' in pattern and len(matches.groups()) == 1:
                min_salary = float(matches.group(1).replace(',', ''))
                max_salary = min_salary * 1.3  # Estimate max as 30% more than min
            else:
                min_salary = float(matches.group(1).replace(',', ''))
                max_salary = float(matches.group(2).replace(',', ''))
            
            # Check if amounts are in thousands (k)
            if 'k' in text.lower() or 'K' in text:
                min_salary *= 1000
                max_salary *= 1000
                
            # Determine currency
            currency = 'USD'  # Default
            if '£' in text or 'GBP' in text:
                currency = 'GBP'
            elif '€' in text or 'EUR' in text:
                currency = 'EUR'
            
            return {
                'min': min_salary,
                'max': max_salary,
                'currency': currency
            }
    
    return {'min': None, 'max': None, 'currency': None}

# Load the skill keywords list from a comprehensive file or use a default list
def load_skill_keywords():
    """Load technical skill keywords from file or use default list"""
    try:
        with open('skill_keywords.txt', 'r') as file:
            return [line.strip().lower() for line in file if line.strip()]
    except FileNotFoundError:
        # Default tech skills list
        return [
            'python', 'java', 'javascript', 'typescript', 'c++', 'c#', 'ruby', 'php', 'swift',
            'kotlin', 'rust', 'go', 'scala', 'perl', 'html', 'css', 'react', 'angular', 'vue', 
            'node.js', 'nodejs', 'django', 'flask', 'spring', '.net', 'express', 'rails', 'laravel',
            'tensorflow', 'pytorch', 'keras', 'scikit-learn', 'pandas', 'numpy', 'scipy',
            'machine learning', 'ml', 'deep learning', 'dl', 'nlp', 'natural language processing',
            'computer vision', 'data science', 'data analysis', 'data mining', 'big data',
            'hadoop', 'spark', 'aws', 'azure', 'gcp', 'docker', 'kubernetes', 'jenkins',
            'ci/cd', 'git', 'github', 'devops', 'agile', 'scrum', 'jira', 'sql', 'mysql',
            'postgresql', 'mongodb', 'elasticsearch', 'redis', 'graphql', 'rest api',
            'microservices', 'system design', 'security', 'networking', 'linux', 'unix',
            'algorithms', 'data structures', 'design patterns', 'testing', 'junit', 'selenium',
            'figma', 'sketch', 'adobe xd', 'ui/ux', 'responsive design', 'mobile development',
            'web development', 'frontend', 'backend', 'full stack', 'cloud computing',
            'serverless', 'blockchain', 'artificial intelligence', 'ai', 'ar/vr', 'game development',
            'embedded systems', 'iot', 'shell scripting', 'powerbi', 'tableau', 'excel',
            'power automate', 'power apps', 'sharepoint', 'product management', 'project management',
            'neo4j', 'graph databases', 'llm', 'rag', 'generative ai', 'langchain', 'openai',
            'prompt engineering', 'vector databases', 'transformers', 'bert', 'gpt',
            'java', 'r', 'sas', 'spss', 'matlab', 'statistics', 'a/b testing', 'etl',
            'data warehousing', 'data modeling', 'data visualization', 'power bi', 'looker',
            'dbt', 'airflow', 'luigi', 'prefect', 'snowflake', 'redshift', 'bigquery', 'databricks',
            'pyspark', 'kafka', 'rabbitmq', 'nginx', 'apache', 'rest', 'soap', 'webrtc',
            'websockets', 'oauth', 'jwt', 'sso', 'ldap', 'active directory', 'cybersecurity',
            'penetration testing', 'ethical hacking', 'cryptography', 'blockchain', 'smart contracts',
            'solidity', 'ethereum', 'swift', 'objective-c', 'flutter', 'react native', 'kotlin',
            'android', 'ios', 'xamarin', 'unity', 'unreal engine', 'godot', 'blender',
            'photoshop', 'illustrator', 'indesign', 'after effects', 'premiere pro',
            'user research', 'user testing', 'wireframing', 'prototyping', 'interaction design',
            'accessibility', 'wcag', 'section 508', 'aria', 'seo', 'sem', 'google analytics',
            'gtm', 'marketing automation', 'hubspot', 'salesforce', 'marketo', 'mailchimp',
            'crm', 'erp', 'cms', 'wordpress', 'drupal', 'joomla', 'shopify', 'magento',
            'woocommerce', 'e-commerce', 'payment processing', 'stripe', 'paypal', 'quickbooks',
            'saas', 'paas', 'iaas', 'virtualization', 'vmware', 'hypervisor', 'terraform',
            'ansible', 'puppet', 'chef', 'cloudformation', 'load balancing', 'cdn',
            'dns', 'domain', 'ssl/tls', 'https', 'vpn', 'firewall', 'ids/ips', 'waf',
            'cism', 'cissp', 'cisa', 'ceh', 'comptia security+', 'comptia network+',
            'comptia a+', 'itil', 'cobit', 'iso 27001', 'gdpr', 'hipaa', 'sox', 'pci dss',
            'rhce', 'rhcsa', 'mcse', 'mcsa', 'ccna', 'ccnp', 'ccie', 'aws certified',
            'azure certified', 'gcp certified', 'pmp', 'prince2', 'six sigma', 'lean',
            'togaf', 'zachman', 'uml', 'bpmn', 'cmmn', 'dmn', 'archiMate', 'aris'
        ]

# Extract skills using a combined approach of spaCy and regex with a skills database
def extract_skills(text, skill_keywords=None):
    """Extract skills from job description text"""
    if not text:
        return []
    
    if skill_keywords is None:
        skill_keywords = load_skill_keywords()
    
    extracted_skills = set()
    
    # Convert text to lowercase for case-insensitive matching
    text_lower = text.lower()
    
    # Simple keyword matching with word boundaries
    for skill in skill_keywords:
        # Create a pattern that matches the skill as a whole word
        # Account for skills that might have special characters
        skill_pattern = r'\b' + re.escape(skill) + r'\b'
        if re.search(skill_pattern, text_lower):
            extracted_skills.add(skill)
    
    # Use spaCy for additional entity recognition (for multi-word terms)
    doc = nlp(text)
    
    # Extract programming languages, frameworks, tools from named entities
    for ent in doc.ents:
        if ent.label_ in ["PRODUCT", "ORG", "GPE"]:
            candidate = ent.text.lower()
            # Check if this entity or any word within it is in our skills list
            if candidate in skill_keywords:
                extracted_skills.add(candidate)
            else:
                # Check if any word in multi-word entities matches our skills
                for word in candidate.split():
                    if word in skill_keywords:
                        extracted_skills.add(word)
    
    # Return as a list of dictionaries with name and category
    return [{"name": skill, "category": categorize_skill(skill)} for skill in extracted_skills]

# Function to categorize skills
def categorize_skill(skill):
    """Categorize a skill into predefined categories"""
    # Define categories and their associated skills
    categories = {
        "programming_language": [
            'python', 'java', 'javascript', 'typescript', 'c++', 'c#', 'ruby', 'php', 
            'swift', 'kotlin', 'rust', 'go', 'scala', 'perl', 'r', 'matlab'
        ],
        "web_development": [
            'html', 'css', 'react', 'angular', 'vue', 'node.js', 'nodejs', 'django', 
            'flask', 'spring', '.net', 'express', 'rails', 'laravel', 'wordpress',
            'frontend', 'backend', 'full stack', 'web development', 'rest api'
        ],
        "data_science": [
            'machine learning', 'ml', 'deep learning', 'dl', 'nlp', 'natural language processing',
            'computer vision', 'data science', 'data analysis', 'data mining', 
            'tensorflow', 'pytorch', 'keras', 'scikit-learn', 'pandas', 'numpy', 'scipy',
            'statistics', 'a/b testing', 'r', 'sas', 'spss'
        ],
        "database": [
            'sql', 'mysql', 'postgresql', 'mongodb', 'elasticsearch', 'redis', 'graphql',
            'neo4j', 'graph databases', 'snowflake', 'redshift', 'bigquery', 'oracle',
            'data warehousing', 'data modeling'
        ],
        "cloud_devops": [
            'aws', 'azure', 'gcp', 'docker', 'kubernetes', 'jenkins', 'ci/cd', 'devops',
            'terraform', 'ansible', 'puppet', 'chef', 'cloudformation', 'serverless',
            'cloud computing', 'iaas', 'paas', 'saas'
        ],
        "ai_llm": [
            'llm', 'rag', 'generative ai', 'langchain', 'openai', 'prompt engineering',
            'vector databases', 'transformers', 'bert', 'gpt', 'ai', 'artificial intelligence'
        ],
        "project_management": [
            'agile', 'scrum', 'jira', 'product management', 'project management',
            'pmp', 'prince2', 'six sigma', 'lean', 'kanban'
        ],
        "soft_skill": [
            'communication', 'teamwork', 'leadership', 'problem solving', 'critical thinking',
            'time management', 'adaptability', 'creativity', 'collaboration', 'presentation'
        ]
    }
    
    # Check which category the skill belongs to
    for category, skills in categories.items():
        if skill in skills:
            return category
    
    # Default category if not found
    return "other"

# Extract qualifications using NLP and pattern matching
def extract_qualifications(text):
    """Extract qualifications from job description text"""
    if not text:
        return []
    
    qualifications = []
    
    # Extract sentences that might contain qualifications
    sentences = sent_tokenize(text.lower())
    qualification_sentences = []
    
    # Keywords indicating qualification requirements
    qualification_indicators = [
        'degree', 'bachelor', 'master', 'phd', 'education', 'qualification', 'certif',
        'diploma', 'years of experience', 'years experience', 'minimum of', 'at least',
        'required', 'qualification', 'must have', 'should have', 'need to have'
    ]
    
    for sentence in sentences:
        if any(indicator in sentence for indicator in qualification_indicators):
            qualification_sentences.append(sentence)
    
    # Process each qualification sentence
    for sentence in qualification_sentences:
        # Check for degree requirements
        degree_required = any(term in sentence for term in ['degree', 'bachelor', 'master', 'phd', 'diploma'])
        
        # Extract years of experience
        years_exp = None
        exp_pattern = r'(\d+)[\+]?\s*(?:\+\s*)?(?:years|year|yrs)(?:\s+of\s+|\s+)(?:experience|exp)'
        exp_match = re.search(exp_pattern, sentence)
        if exp_match:
            years_exp = int(exp_match.group(1))
        
        qualification = {
            "description": sentence.strip(),
            "degree_required": degree_required,
            "years_experience": years_exp
        }
        
        qualifications.append(qualification)
    
    return qualifications

# Extract responsibilities from job description
def extract_responsibilities(text):
    """Extract job responsibilities from job description text"""
    if not text:
        return []
    
    responsibilities = []
    
    # Look for sections typically containing responsibilities
    responsibility_section = None
    
    # Common section headers for responsibilities
    headers = [
        r'responsibilities:',
        r'key responsibilities:',
        r'job responsibilities:',
        r'duties:',
        r'what you\'ll do:',
        r'what you will do:',
        r'role responsibilities:'
    ]
    
    # Try to find a responsibilities section
    text_lower = text.lower()
    for header in headers:
        match = re.search(header, text_lower)
        if match:
            start_idx = match.end()
            # Find the next section header or end of text
            next_section_match = re.search(r'^\s*[a-z\s]+:', text_lower[start_idx:], re.MULTILINE)
            if next_section_match:
                end_idx = start_idx + next_section_match.start()
                responsibility_section = text[start_idx:end_idx].strip()
            else:
                responsibility_section = text[start_idx:].strip()
            break
    
    # If we found a responsibility section, extract bullet points or sentences
    if responsibility_section:
        # Try to find bullet points
        bullet_pattern = r'(?:^|\n)(?:\s*[\•\-\*\✓\✔\■\○\●]\s*)(.*?)(?=\n|$)'
        bullet_matches = re.findall(bullet_pattern, responsibility_section)
        
        if bullet_matches:
            responsibilities = [{"description": item.strip()} for item in bullet_matches if item.strip()]
        else:
            # If no bullet points, split by sentences
            sentences = sent_tokenize(responsibility_section)
            responsibilities = [{"description": sentence.strip()} for sentence in sentences if sentence.strip()]
    else:
        # If no dedicated responsibility section found, try to identify responsibility sentences
        # Common phrases in responsibility statements
        responsibility_phrases = [
            r'responsible for',
            r'be responsible',
            r'your responsibilities include',
            r'duties include',
            r'you will',
            r'you\'ll',
            r'lead',
            r'develop',
            r'manage',
            r'coordinate',
            r'design',
            r'implement',
            r'maintain',
            r'create',
            r'ensure',
            r'drive'
        ]
        
        for sentence in sent_tokenize(text):
            sentence_lower = sentence.lower()
            if any(re.search(phrase, sentence_lower) for phrase in responsibility_phrases):
                responsibilities.append({"description": sentence.strip()})
    
    return responsibilities

# Process a single job post
def process_job_post(job_post, session):
    """Process a single job post and add to database"""
    # Check if job already exists
    existing_job = session.query(Job).filter_by(jooble_id=job_post.get("id", "")).first()
    if existing_job:
        logger.info(f"Job ID {job_post.get('id')} already exists in database. Skipping.")
        return existing_job
    
    # Extract basic job information
    job_data = {
        "jooble_id": job_post.get("id", ""),
        "title": job_post.get("title", ""),
        "company": job_post.get("company", ""),
        "location": job_post.get("location", ""),
        "job_type": job_post.get("type", ""),
        "posting_date": job_post.get("updated", ""),  # Use 'updated' as Jooble posting date
        "updated_date": job_post.get("updated", ""),
        "source_url": job_post.get("link", ""),
        "description": job_post.get("description", ""),
        "snippet": job_post.get("snippet", "")
    }
    
    description = job_data["description"]
    
    # Log description length for debugging
    logger.info(f"Job ID {job_post.get('id')} description length: {len(description) if description else 0} chars")
    
    # Extract more complex information
    salary_info = extract_salary_info(description)
    job_data["salary_min"] = salary_info["min"]
    job_data["salary_max"] = salary_info["max"]
    job_data["salary_currency"] = salary_info["currency"]
    
    # Create job entry
    job = Job(**job_data)
    
    try:
        # Process skills
        skills_data = extract_skills(description)
        logger.info(f"Extracted {len(skills_data)} skills for job ID {job_post.get('id')}")
        
        for skill_data in skills_data:
            try:
                # Check if skill already exists
                skill = session.query(Skill).filter_by(name=skill_data["name"]).first()
                if not skill:
                    skill = Skill(name=skill_data["name"], category=skill_data["category"])
                    session.add(skill)
                    # Ensure the session has the skill ID before adding to relationship
                    session.flush()
                job.skills.append(skill)
            except Exception as e:
                logger.error(f"Error adding skill {skill_data['name']} to job {job_post.get('id')}: {str(e)}")
    except Exception as e:
        logger.error(f"Error extracting skills for job {job_post.get('id')}: {str(e)}")
    
    try:
        # Process qualifications
        qualifications_data = extract_qualifications(description)
        logger.info(f"Extracted {len(qualifications_data)} qualifications for job ID {job_post.get('id')}")
        
        for qual_data in qualifications_data:
            try:
                # Check if qualification already exists
                qual = session.query(Qualification).filter_by(description=qual_data["description"]).first()
                if not qual:
                    qual = Qualification(**qual_data)
                    session.add(qual)
                    # Ensure the session has the qualification ID before adding to relationship
                    session.flush()
                job.qualifications.append(qual)
            except Exception as e:
                logger.error(f"Error adding qualification to job {job_post.get('id')}: {str(e)}")
    except Exception as e:
        logger.error(f"Error extracting qualifications for job {job_post.get('id')}: {str(e)}")
    
    try:
        # Process responsibilities
        responsibilities_data = extract_responsibilities(description)
        logger.info(f"Extracted {len(responsibilities_data)} responsibilities for job ID {job_post.get('id')}")
        
        for resp_data in responsibilities_data:
            try:
                # Check if responsibility already exists
                resp = session.query(Responsibility).filter_by(description=resp_data["description"]).first()
                if not resp:
                    resp = Responsibility(**resp_data)
                    session.add(resp)
                    # Ensure the session has the responsibility ID before adding to relationship
                    session.flush()
                job.responsibilities.append(resp)
            except Exception as e:
                logger.error(f"Error adding responsibility to job {job_post.get('id')}: {str(e)}")
    except Exception as e:
        logger.error(f"Error extracting responsibilities for job {job_post.get('id')}: {str(e)}")
    
    session.add(job)
    return job

# Main function to fetch and process job posts
def fetch_and_process_jobs(api_key, keywords=None, locations=None, total_jobs=100, batch_size=20):
    """
    Fetch and process jobs from Jooble API
    
    Parameters:
    api_key (str): Jooble API key
    keywords (list): List of job keywords to search for
    locations (list): List of locations to search in
    total_jobs (int): Total number of jobs to fetch
    batch_size (int): Number of jobs to fetch per API call
    
    Returns:
    list: List of processed job objects
    """
    if keywords is None:
        keywords = ["data scientist", "machine learning engineer", "data engineer", "software engineer"]
    
    if locations is None:
        locations = ["", "USA", "UK", "Remote"]  # Empty string for no location filter
    
    # Create database session
    session = create_db_session()
    processed_jobs = []
    
    try:
        # Fetch jobs for each keyword and location combination
        for keyword in keywords:
            for location in locations:
                jobs_to_fetch = min(total_jobs // (len(keywords) * len(locations)), 100)  # Limit per combination
                
                # Calculate how many pages to fetch
                pages_to_fetch = (jobs_to_fetch + batch_size - 1) // batch_size
                
                for page in range(1, pages_to_fetch + 1):
                    try:
                        # Fetch job posts
                        logger.info(f"Fetching jobs for keyword='{keyword}', location='{location}', page={page}")
                        
                        # Add delay to avoid rate limiting
                        if page > 1:
                            time.sleep(2)  # 2 second delay between requests
                            
                        response = fetch_jooble_jobs(
                            api_key=api_key,
                            keywords=keyword,
                            location=location,
                            page=page,
                            limit=batch_size
                        )
                        
                        # Process each job post
                        jobs = response.get('jobs', [])
                        
                        if not jobs:
                            logger.warning(f"No jobs returned for keyword='{keyword}', location='{location}', page={page}")
                            break
                            
                        logger.info(f"Processing {len(jobs)} jobs from page {page}")
                        
                        for job in tqdm(jobs, desc=f"Processing {keyword}/{location} - Page {page}"):
                            processed_job = process_job_post(job, session)
                            if processed_job:
                                processed_jobs.append(processed_job)
                        
                        # Commit after each batch
                        session.commit()
                        
                        # If we didn't get a full page of results, no need to fetch more
                        if len(jobs) < batch_size:
                            logger.info(f"Received partial page ({len(jobs)}/{batch_size}). No more results available.")
                            break
                            
                    except Exception as e:
                        logger.error(f"Error processing page {page} for '{keyword}'/'{location}': {str(e)}")
                        session.rollback()
                        # Continue with next page
    
    except Exception as e:
        logger.error(f"Error in fetch_and_process_jobs: {str(e)}")
        session.rollback()
    finally:
        # Final commit and close session
        session.commit()
        session.close()
    
    return processed_jobs

# Generate statistics about the processed data
def generate_statistics(db_url="sqlite:///jooble_jobs.db"):
    """Generate statistics from the processed job data"""
    engine = create_engine(db_url)
    Session = sessionmaker(bind=engine)
    session = Session()
    
    stats = {}
    
    # Total number of jobs
    stats['total_jobs'] = session.query(Job).count()
    
    # Total number of unique skills
    stats['total_skills'] = session.query(Skill).count()
    
    # Top 20 skills
    top_skills = []
    skill_counts = session.query(
        Skill.name, 
        Skill.category,
        func.count(job_skill.c.job_id).label('job_count')
    ).join(job_skill).group_by(Skill.id).order_by(desc('job_count')).limit(20).all()
    
    for skill in skill_counts:
        top_skills.append({
            'name': skill.name,
            'category': skill.category,
            'count': skill.job_count
        })
    stats['top_skills'] = top_skills
    
    # Average number of skills per job
    subquery = session.query(
        job_skill.c.job_id, 
        func.count(job_skill.c.skill_id).label('skill_count')
    ).group_by(job_skill.c.job_id).subquery()
    
    avg_skills = session.query(func.avg(subquery.c.skill_count)).scalar() or 0
    stats['avg_skills_per_job'] = round(avg_skills, 2)
    
    # Skills by category
    skills_by_category = session.query(
        Skill.category,
        func.count(Skill.id).label('count')
    ).group_by(Skill.category).all()
    
    categories = {}
    for category in skills_by_category:
        categories[category.category] = category.count
    stats['skills_by_category'] = categories
    
    # Salary statistics (excluding null values)
    salary_stats = {}
    salary_query = session.query(
        func.avg(Job.salary_min).label('avg_min'),
        func.avg(Job.salary_max).label('avg_max'),
        func.min(Job.salary_min).label('min'),
        func.max(Job.salary_max).label('max')
    ).filter(Job.salary_min != None, Job.salary_max != None).first()
    
    if salary_query:
        salary_stats['avg_min'] = round(salary_query.avg_min, 2) if salary_query.avg_min else None
        salary_stats['avg_max'] = round(salary_query.avg_max, 2) if salary_query.avg_max else None
        salary_stats['min'] = salary_query.min
        salary_stats['max'] = salary_query.max
    
    stats['salary_stats'] = salary_stats
    
    # Most common job types
    job_types = session.query(
        Job.job_type,
        func.count(Job.id).label('count')
    ).group_by(Job.job_type).order_by(desc('count')).limit(5).all()
    
    top_job_types = []
    for job_type in job_types:
        if job_type.job_type:  # Skip empty job types
            top_job_types.append({
                'type': job_type.job_type,
                'count': job_type.count
            })
    stats['top_job_types'] = top_job_types
    
    # Most common qualifications
    qual_counts = session.query(
        Qualification.description,
        func.count(job_qualification.c.job_id).label('job_count')
    ).join(job_qualification).group_by(Qualification.id).order_by(desc('job_count')).limit(10).all()
    
    top_qualifications = []
    for qual in qual_counts:
        top_qualifications.append({
            'description': qual.description,
            'count': qual.job_count
        })
    stats['top_qualifications'] = top_qualifications
    
    session.close()
    return stats

# Function to export statistics to JSON
def export_statistics_to_json(stats, filename="job_stats.json"):
    """Export job statistics to a JSON file"""
    with open(filename, 'w') as f:
        json.dump(stats, f, indent=4)
    logger.info(f"Statistics exported to {filename}")

# Function to test extraction with a sample job description
def test_extraction():
    """Test the extraction functions with a sample job description"""
    sample_description = """
    Data Scientist
    
    About the Role:
    We are seeking a talented Data Scientist to join our analytics team. The ideal candidate will have strong programming skills and experience with machine learning algorithms.
    
    Responsibilities:
    • Build and optimize machine learning models for prediction and classification tasks
    • Collaborate with engineering teams to implement models into production
    • Analyze large datasets to extract actionable insights
    • Present findings to stakeholders in a clear and compelling manner
    
    Requirements:
    • Bachelor's degree in Computer Science, Statistics, or related field
    • 3+ years of experience in data science or similar role
    • Proficiency in Python, SQL, and statistical analysis
    • Experience with machine learning frameworks such as TensorFlow or PyTorch
    • Knowledge of cloud platforms (AWS, GCP, or Azure)
    • Excellent communication and problem-solving skills
    
    Salary range: $90,000 - $120,000 per year depending on experience
    
    Job Type: Full-time
    """
    
    logger.info("Testing extraction functions with sample job description")
    
    # Test skill extraction
    try:
        skills = extract_skills(sample_description)
        logger.info(f"Extracted {len(skills)} skills from sample:")
        for skill in skills:
            logger.info(f"  - {skill['name']} (Category: {skill['category']})")
    except Exception as e:
        logger.error(f"Error in skill extraction test: {str(e)}")
        skills = []
    
    # Test qualification extraction
    try:
        qualifications = extract_qualifications(sample_description)
        logger.info(f"Extracted {len(qualifications)} qualifications from sample:")
        for qual in qualifications:
            logger.info(f"  - {qual['description']}")
            logger.info(f"    Degree Required: {qual['degree_required']}, Years Experience: {qual['years_experience']}")
    except Exception as e:
        logger.error(f"Error in qualification extraction test: {str(e)}")
        qualifications = []
    
    # Test responsibility extraction
    try:
        responsibilities = extract_responsibilities(sample_description)
        logger.info(f"Extracted {len(responsibilities)} responsibilities from sample:")
        for resp in responsibilities:
            logger.info(f"  - {resp['description']}")
    except Exception as e:
        logger.error(f"Error in responsibility extraction test: {str(e)}")
        responsibilities = []
    
    # Test salary extraction
    try:
        salary = extract_salary_info(sample_description)
        logger.info(f"Extracted salary: ${salary['min']} - ${salary['max']} {salary['currency']}")
    except Exception as e:
        logger.error(f"Error in salary extraction test: {str(e)}")
        salary = {'min': None, 'max': None, 'currency': None}
    
    return {
        "skills": skills,
        "qualifications": qualifications,
        "responsibilities": responsibilities,
        "salary": salary
    }

# Function to rebuild statistics and fix relationships in the database
def rebuild_statistics():
    """Repair and rebuild the statistics database by reprocessing job descriptions"""
    logger.info("Starting database repair and statistics rebuild")
    
    # Create database session
    engine = create_engine("sqlite:///jooble_jobs.db")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    
    try:
        # Count existing records
        job_count = session.query(Job).count()
        logger.info(f"Found {job_count} existing jobs in database")
        
        if job_count == 0:
            logger.error("No jobs found in database. Nothing to rebuild.")
            return False
        
        # Verify NLTK and spaCy are working
        logger.info("Verifying NLTK and spaCy are working properly...")
        try:
            # Test with a simple sentence
            test_sentence = "Testing NLTK and spaCy. This should split into sentences."
            sentences = sent_tokenize(test_sentence)
            if len(sentences) != 2:
                logger.warning(f"NLTK sent_tokenize may not be working correctly. Got {len(sentences)} sentences, expected 2.")
            
            doc = nlp("Testing spaCy with entities like Google, Microsoft, and Python.")
            if len(list(doc.ents)) == 0:
                logger.warning("spaCy may not be identifying entities correctly.")
            else:
                logger.info(f"spaCy identified {len(list(doc.ents))} entities in test sentence.")
                
            logger.info("NLTK and spaCy verification complete.")
        except Exception as e:
            logger.error(f"Error during NLTK/spaCy verification: {str(e)}")
            logger.warning("Continuing with rebuild despite verification issues...")
        
        # Clear existing relationships but keep the jobs
        logger.info("Clearing existing relationships")
        session.execute(job_skill.delete())
        session.execute(job_qualification.delete())
        session.execute(job_responsibility.delete())
        session.commit()
        
        # Delete all skills, qualifications, and responsibilities
        session.query(Skill).delete()
        session.query(Qualification).delete()
        session.query(Responsibility).delete()
        session.commit()
        
        # Check if any job has empty description
        empty_desc_count = session.query(Job).filter(
            (Job.description == None) | (Job.description == '')
        ).count()
        
        if empty_desc_count > 0:
            logger.warning(f"{empty_desc_count} jobs have empty descriptions. These jobs will not yield any skills/qualifications.")
        
        # Get a sample job to verify
        sample_job = session.query(Job).filter(Job.description != '').first()
        if sample_job:
            logger.info(f"Sample job title: {sample_job.title}")
            logger.info(f"Sample job description length: {len(sample_job.description)} chars")
            logger.info(f"Description starts with: {sample_job.description[:100] if sample_job.description else 'N/A'}...")
        
        # Reprocess each job
        logger.info("Reprocessing job descriptions to extract data")
        jobs = session.query(Job).all()
        
        success_count = 0
        error_count = 0
        
        for idx, job in enumerate(tqdm(jobs, desc="Reprocessing jobs")):
            try:
                if not job.description:
                    logger.warning(f"Job ID {job.id} has empty description. Skipping extraction.")
                    continue
                
                # Extract skills
                skills_data = extract_skills(job.description)
                for skill_data in skills_data:
                    try:
                        # Check if skill already exists
                        skill = session.query(Skill).filter_by(name=skill_data["name"]).first()
                        if not skill:
                            skill = Skill(name=skill_data["name"], category=skill_data["category"])
                            session.add(skill)
                            session.flush()
                        job.skills.append(skill)
                    except Exception as e:
                        logger.error(f"Error adding skill {skill_data['name']} for job {job.id}: {str(e)}")
                
                # Extract qualifications
                qualifications_data = extract_qualifications(job.description)
                for qual_data in qualifications_data:
                    try:
                        # Check if qualification already exists
                        qual = session.query(Qualification).filter_by(description=qual_data["description"]).first()
                        if not qual:
                            qual = Qualification(**qual_data)
                            session.add(qual)
                            session.flush()
                        job.qualifications.append(qual)
                    except Exception as e:
                        logger.error(f"Error adding qualification for job {job.id}: {str(e)}")
                
                # Extract responsibilities
                responsibilities_data = extract_responsibilities(job.description)
                for resp_data in responsibilities_data:
                    try:
                        # Check if responsibility already exists
                        resp = session.query(Responsibility).filter_by(description=resp_data["description"]).first()
                        if not resp:
                            resp = Responsibility(**resp_data)
                            session.add(resp)
                            session.flush()
                        job.responsibilities.append(resp)
                    except Exception as e:
                        logger.error(f"Error adding responsibility for job {job.id}: {str(e)}")
                
                # Commit every 50 jobs
                if idx % 50 == 0:
                    session.commit()
                    logger.info(f"Processed {idx+1}/{job_count} jobs")
                
                success_count += 1
                
            except Exception as e:
                logger.error(f"Error reprocessing job ID {job.id}: {str(e)}")
                error_count += 1
        
        # Final commit
        session.commit()
        logger.info(f"Database relationships rebuilt successfully. {success_count} jobs processed successfully, {error_count} errors.")
        
        # Check if we got any skills
        skill_count = session.query(Skill).count()
        if skill_count == 0:
            logger.warning("No skills were extracted during the rebuild process. This could indicate an issue with the extraction code or the job descriptions.")
        else:
            logger.info(f"Successfully extracted {skill_count} unique skills.")
        
        # Generate and export statistics
        logger.info("Generating statistics from repaired database")
        stats = generate_statistics()
        export_statistics_to_json(stats, "repaired_job_stats.json")
        
        logger.info("Database repair complete!")
        return True
    
    except Exception as e:
        logger.error(f"Error during database repair: {str(e)}")
        session.rollback()
        return False
    finally:
        session.close()

# Main execution function
def main():
    """Main execution function"""
    # Test extraction functions first to verify they work
    logger.info("Running extraction test before starting API calls")
    test_results = test_extraction()
    
    if not test_results["skills"]:
        logger.warning("Test extraction didn't find any skills in the sample description. This could indicate an issue with the extraction code.")
    
    # Menu for user options
    print("\nSelect an option:")
    print("1. Fetch new jobs from API")
    print("2. Repair and rebuild existing database")
    print("3. Exit")
    
    choice = input("Enter choice (1-3): ")
    
    if choice == "1":
        # Replace with your Jooble API key
        api_key = "6aca5242-9a2e-4b00-9a81-420bc53f3888"
        
        # Define search parameters
        keywords = [
            "data scientist", 
            "machine learning engineer", 
            "data engineer", 
            "data analyst", 
            "AI engineer",
            "big data engineer"
        ]
        
        locations = ["", "USA", "UK", "Remote"]
        
        # Fetch and process jobs
        logger.info("Starting job data extraction process")
        jobs = fetch_and_process_jobs(
            api_key=api_key,
            keywords=keywords,
            locations=locations,
            total_jobs=200,
            batch_size=20
        )
        logger.info(f"Processed {len(jobs)} jobs")
        
        # Generate and export statistics
        logger.info("Generating statistics from processed jobs")
        stats = generate_statistics()
        export_statistics_to_json(stats)
        
        logger.info("Job processing complete!")
    
    elif choice == "2":
        # Call the rebuild_statistics function
        rebuild_statistics()
    
    elif choice == "3":
        logger.info("Exiting per user request")
    
    else:
        logger.error("Invalid choice. Exiting.")

if __name__ == "__main__":
    main()
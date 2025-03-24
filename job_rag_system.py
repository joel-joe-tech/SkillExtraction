from neo4j import GraphDatabase
import logging
from typing import List, Dict, Any
from langchain_huggingface import HuggingFaceEndpoint
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import spacy
from langchain_community.vectorstores import FAISS
from langchain.embeddings.base import Embeddings
import json
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SpacyEmbeddings(Embeddings):
    def __init__(self, nlp):
        self.nlp = nlp

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents using spaCy."""
        return [self.nlp(text).vector.tolist() for text in texts]

    def embed_query(self, text: str) -> List[float]:
        """Embed a query using spaCy."""
        return self.nlp(text).vector.tolist()

class JobRAGSystem:
    def __init__(self, 
                 neo4j_uri: str = "bolt://localhost:7687",
                 neo4j_user: str = "neo4j",
                 neo4j_password: str = "abcde12345"):
        """Initialize the RAG system with Neo4j connection and LLM."""
        self.driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
        
        # Initialize Hugging Face model (free)
        self.llm = HuggingFaceEndpoint(
            repo_id="google/flan-t5-base",
            temperature=0.7,
            huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN", "")
        )
        
        # Initialize spaCy model for embeddings
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            logger.info("Downloading spaCy model...")
            os.system("python -m spacy download en_core_web_sm")
            self.nlp = spacy.load("en_core_web_sm")
        
        # Initialize vector store with custom embeddings
        self.embeddings = SpacyEmbeddings(self.nlp)
        self.vector_store = self._initialize_vector_store()
        
    def close(self):
        """Close the Neo4j connection."""
        self.driver.close()
    
    def _get_embedding(self, text: str) -> List[float]:
        """Get embedding for text using spaCy."""
        doc = self.nlp(text)
        return doc.vector.tolist()
    
    def _initialize_vector_store(self):
        """Initialize FAISS vector store with job data."""
        # Get all jobs and their requirements from Neo4j
        with self.driver.session() as session:
            query = """
            MATCH (j:Job)
            OPTIONAL MATCH (j)-[:REQUIRES_SKILL]->(s:Skill)
            RETURN j.title as title,
                   j.company as company,
                   j.location as location,
                   j.description as description,
                   collect(DISTINCT s.name) as skills
            """
            result = session.run(query)
            jobs = [dict(record) for record in result]
        
        # Create documents for vector store
        documents = []
        for job in jobs:
            doc = f"""
            Job Title: {job['title']}
            Company: {job['company']}
            Location: {job['location']}
            Description: {job['description']}
            Required Skills: {', '.join(job['skills'])}
            """
            documents.append(doc)
        
        # Create and return vector store using the custom embeddings
        return FAISS.from_texts(documents, self.embeddings)
    
    def get_relevant_jobs(self, query: str, k: int = 3) -> List[Dict[str, Any]]:
        """Get relevant jobs based on the query using vector similarity search."""
        docs = self.vector_store.similarity_search(query, k=k)
        return [json.loads(doc.page_content) for doc in docs]
    
    def get_skill_network(self, skill_name: str) -> Dict[str, Any]:
        """Get the network of related skills and jobs for a specific skill."""
        with self.driver.session() as session:
            query = """
            MATCH (s:Skill {name: $name})
            OPTIONAL MATCH (j:Job)-[:REQUIRES_SKILL]->(s)
            OPTIONAL MATCH (j)-[:REQUIRES_SKILL]->(related:Skill)
            WHERE related <> s
            RETURN s.name as skill,
                   collect(DISTINCT j.title) as jobs,
                   collect(DISTINCT related.name) as related_skills
            """
            result = session.run(query, name=skill_name)
            return dict(result.single())
    
    def get_career_path(self, current_skills: List[str], target_role: str) -> Dict[str, Any]:
        """Get a career path recommendation based on current skills and target role."""
        with self.driver.session() as session:
            # Get required skills for target role
            query = """
            MATCH (j:Job)
            WHERE j.title CONTAINS $target_role
            OPTIONAL MATCH (j)-[:REQUIRES_SKILL]->(s:Skill)
            RETURN collect(DISTINCT s.name) as required_skills
            """
            result = session.run(query, target_role=target_role)
            required_skills = result.single()["required_skills"]
            
            # Find missing skills
            missing_skills = [skill for skill in required_skills if skill not in current_skills]
            
            # Get jobs that can help acquire missing skills
            skill_development_path = []
            for skill in missing_skills:
                skill_query = """
                MATCH (j:Job)-[:REQUIRES_SKILL]->(s:Skill {name: $skill})
                RETURN j.title as title,
                       j.company as company,
                       collect(s.name) as skills
                LIMIT 3
                """
                skill_result = session.run(skill_query, skill=skill)
                skill_development_path.extend([dict(record) for record in skill_result])
            
            return {
                "target_role": target_role,
                "current_skills": current_skills,
                "required_skills": required_skills,
                "missing_skills": missing_skills,
                "skill_development_path": skill_development_path
            }
    
    def answer_question(self, question: str) -> str:
        """Answer a question about jobs and careers using RAG."""
        # Get relevant context from vector store
        relevant_jobs = self.get_relevant_jobs(question)
        
        # Create prompt template
        template = """
        Based on the following job market information, please answer the question.
        If you cannot answer the question with the provided information, say so.
        
        Context:
        {context}
        
        Question: {question}
        
        Answer: """
        
        prompt = PromptTemplate(
            input_variables=["context", "question"],
            template=template
        )
        
        # Create and run chain
        chain = LLMChain(llm=self.llm, prompt=prompt)
        
        # Format context
        context = "\n\n".join([
            f"Job: {job['title']}\n"
            f"Company: {job['company']}\n"
            f"Skills: {', '.join(job['skills'])}\n"
            for job in relevant_jobs
        ])
        
        # Get answer
        response = chain.run(context=context, question=question)
        return response

def main():
    """Main function to demonstrate the RAG system."""
    rag = JobRAGSystem()
    
    try:
        # Example questions
        questions = [
            "What skills are required for a software engineer position?",
            "What career path should I take to become a data scientist?",
            "Which companies are hiring for Python developers?"
        ]
        
        for question in questions:
            print(f"\nQuestion: {question}")
            answer = rag.answer_question(question)
            print(f"Answer: {answer}")
            
        # Example career path
        current_skills = ["Python", "SQL"]
        target_role = "Data Scientist"
        print(f"\nCareer path from {', '.join(current_skills)} to {target_role}:")
        path = rag.get_career_path(current_skills, target_role)
        print(json.dumps(path, indent=2))
        
    except Exception as e:
        logger.error(f"Error in RAG system: {str(e)}")
        raise
    finally:
        rag.close()

if __name__ == "__main__":
    main() 
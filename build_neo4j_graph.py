import sqlite3
from neo4j import GraphDatabase
import logging
from typing import List, Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Neo4jGraphBuilder:
    def __init__(self, uri: str = "bolt://localhost:7687", user: str = "neo4j", password: str = "abcde12345"):
        """Initialize Neo4j connection."""
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        
    def close(self):
        """Close the Neo4j connection."""
        self.driver.close()
        
    def create_constraints(self, session):
        """Create constraints for the graph database."""
        constraints = [
            "CREATE CONSTRAINT job_id IF NOT EXISTS FOR (j:Job) REQUIRE j.id IS UNIQUE",
            "CREATE CONSTRAINT skill_name IF NOT EXISTS FOR (s:Skill) REQUIRE s.name IS UNIQUE",
            "CREATE CONSTRAINT experience_level IF NOT EXISTS FOR (e:Experience) REQUIRE e.level IS UNIQUE",
            "CREATE CONSTRAINT education_level IF NOT EXISTS FOR (e:Education) REQUIRE e.level IS UNIQUE"
        ]
        
        for constraint in constraints:
            try:
                session.run(constraint)
                logger.info(f"Created constraint: {constraint}")
            except Exception as e:
                logger.warning(f"Could not create constraint {constraint}: {str(e)}")
    
    def create_job_node(self, session, job_data: Dict[str, Any]):
        """Create a Job node with its properties."""
        query = """
        MERGE (j:Job {id: $id})
        SET j.title = $title,
            j.company = $company,
            j.location = $location,
            j.description = $description,
            j.url = $url,
            j.salary = $salary
        """
        session.run(query, job_data)
        logger.info(f"Created job node: {job_data['title']}")
    
    def create_skill_node(self, session, skill_name: str):
        """Create a Skill node if it doesn't exist."""
        query = "MERGE (s:Skill {name: $name})"
        session.run(query, name=skill_name)
        logger.info(f"Created skill node: {skill_name}")
    
    def create_experience_node(self, session, level: str):
        """Create an Experience node if it doesn't exist."""
        query = "MERGE (e:Experience {level: $level})"
        session.run(query, level=level)
        logger.info(f"Created experience node: {level}")
    
    def create_education_node(self, session, level: str):
        """Create an Education node if it doesn't exist."""
        query = "MERGE (e:Education {level: $level})"
        session.run(query, level=level)
        logger.info(f"Created education node: {level}")
    
    def create_relationships(self, session, job_id: str, relationships: Dict[str, List[str]]):
        """Create relationships between Job node and other nodes."""
        for rel_type, items in relationships.items():
            for item in items:
                if rel_type == "REQUIRES_SKILL":
                    query = """
                    MATCH (j:Job {id: $job_id})
                    MATCH (s:Skill {name: $item})
                    MERGE (j)-[:REQUIRES_SKILL]->(s)
                    """
                elif rel_type == "REQUIRES_EXPERIENCE":
                    query = """
                    MATCH (j:Job {id: $job_id})
                    MATCH (e:Experience {level: $item})
                    MERGE (j)-[:REQUIRES_EXPERIENCE]->(e)
                    """
                elif rel_type == "REQUIRES_EDUCATION":
                    query = """
                    MATCH (j:Job {id: $job_id})
                    MATCH (e:Education {level: $item})
                    MERGE (j)-[:REQUIRES_EDUCATION]->(e)
                    """
                
                try:
                    session.run(query, job_id=job_id, item=item)
                    logger.info(f"Created {rel_type} relationship for job {job_id} with {item}")
                except Exception as e:
                    logger.error(f"Error creating {rel_type} relationship: {str(e)}")

def build_graph():
    """Main function to build the Neo4j graph from SQLite database."""
    # Connect to SQLite database
    sqlite_conn = sqlite3.connect('jooble_jobs.db')
    sqlite_cursor = sqlite_conn.cursor()
    
    # Initialize Neo4j graph builder
    graph_builder = Neo4jGraphBuilder()
    
    try:
        with graph_builder.driver.session() as session:
            # Create constraints
            graph_builder.create_constraints(session)
            
            # Check if tables exist
            sqlite_cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            existing_tables = [table[0] for table in sqlite_cursor.fetchall()]
            
            if 'jobs' not in existing_tables:
                logger.error("No jobs table found. Please run job_extraction_model.py first to process the job data.")
                return
            
            # Get all jobs
            sqlite_cursor.execute("SELECT * FROM jobs")
            jobs = sqlite_cursor.fetchall()
            
            # Process each job
            for job in jobs:
                job_id = job[0]
                job_data = {
                    'id': job_id,
                    'title': job[1],
                    'company': job[2],
                    'location': job[3],
                    'description': job[4],
                    'url': job[5],
                    'salary': job[6]
                }
                
                # Create job node
                graph_builder.create_job_node(session, job_data)
                
                # Get skills for this job if table exists
                if 'skills' in existing_tables and 'job_skills' in existing_tables:
                    sqlite_cursor.execute("""
                        SELECT s.name 
                        FROM skills s 
                        JOIN job_skills js ON s.id = js.skill_id 
                        WHERE js.job_id = ?
                    """, (job_id,))
                    skills = [row[0] for row in sqlite_cursor.fetchall()]
                else:
                    skills = []
                
                # Get experience requirements if table exists
                if 'experience' in existing_tables and 'job_experience' in existing_tables:
                    sqlite_cursor.execute("""
                        SELECT e.level 
                        FROM experience e 
                        JOIN job_experience je ON e.id = je.experience_id 
                        WHERE je.job_id = ?
                    """, (job_id,))
                    experience_levels = [row[0] for row in sqlite_cursor.fetchall()]
                else:
                    experience_levels = []
                
                # Get education requirements if table exists
                if 'education' in existing_tables and 'job_education' in existing_tables:
                    sqlite_cursor.execute("""
                        SELECT e.level 
                        FROM education e 
                        JOIN job_education je ON e.id = je.education_id 
                        WHERE je.job_id = ?
                    """, (job_id,))
                    education_levels = [row[0] for row in sqlite_cursor.fetchall()]
                else:
                    education_levels = []
                
                # Create skill nodes and relationships
                for skill in skills:
                    graph_builder.create_skill_node(session, skill)
                
                # Create experience nodes and relationships
                for level in experience_levels:
                    graph_builder.create_experience_node(session, level)
                
                # Create education nodes and relationships
                for level in education_levels:
                    graph_builder.create_education_node(session, level)
                
                # Create relationships
                relationships = {
                    "REQUIRES_SKILL": skills,
                    "REQUIRES_EXPERIENCE": experience_levels,
                    "REQUIRES_EDUCATION": education_levels
                }
                graph_builder.create_relationships(session, job_id, relationships)
                
    except Exception as e:
        logger.error(f"Error building graph: {str(e)}")
        raise
    finally:
        sqlite_conn.close()
        graph_builder.close()

if __name__ == "__main__":
    logger.info("Starting Neo4j graph build process...")
    build_graph()
    logger.info("Neo4j graph build completed successfully!") 
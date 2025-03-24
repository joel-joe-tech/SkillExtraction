from neo4j import GraphDatabase
import logging
from typing import List, Dict, Any
import networkx as nx
import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Neo4jGraphQuerier:
    def __init__(self, uri: str = "bolt://localhost:7687", user: str = "neo4j", password: str = "abcde12345"):
        """Initialize Neo4j connection."""
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        
    def close(self):
        """Close the Neo4j connection."""
        self.driver.close()
    
    def get_skill_network(self, session, limit: int = 10):
        """Get the network of skills and their relationships."""
        query = """
        MATCH (j:Job)-[:REQUIRES_SKILL]->(s:Skill)
        WITH s, count(*) as job_count
        ORDER BY job_count DESC
        LIMIT $limit
        MATCH (j:Job)-[:REQUIRES_SKILL]->(s)
        RETURN s.name as skill, collect(j.title) as jobs
        """
        result = session.run(query, limit=limit)
        return [dict(record) for record in result]
    
    def get_job_requirements(self, session, job_title: str):
        """Get all requirements for a specific job."""
        query = """
        MATCH (j:Job)
        WHERE j.title CONTAINS $title
        OPTIONAL MATCH (j)-[:REQUIRES_SKILL]->(s:Skill)
        OPTIONAL MATCH (j)-[:REQUIRES_EXPERIENCE]->(e:Experience)
        OPTIONAL MATCH (j)-[:REQUIRES_EDUCATION]->(ed:Education)
        RETURN j.title as title,
               collect(DISTINCT s.name) as skills,
               collect(DISTINCT e.level) as experience,
               collect(DISTINCT ed.level) as education
        LIMIT 1
        """
        result = session.run(query, title=job_title)
        record = result.single()
        if record is None:
            logger.warning(f"No job found containing '{job_title}' in the title")
            return {
                'title': job_title,
                'skills': [],
                'experience': [],
                'education': []
            }
        return dict(record)
    
    def get_related_jobs(self, session, skill_name: str):
        """Get jobs that require a specific skill."""
        query = """
        MATCH (j:Job)-[:REQUIRES_SKILL]->(s:Skill {name: $name})
        RETURN j.title as title, j.company as company, j.location as location
        """
        result = session.run(query, name=skill_name)
        jobs = [dict(record) for record in result]
        if not jobs:
            logger.warning(f"No jobs found requiring the skill '{skill_name}'")
        return jobs
    
    def visualize_skill_network(self, network_data: List[Dict[str, Any]], output_file: str = "skill_network.png"):
        """Visualize the skill network using NetworkX and Matplotlib."""
        G = nx.Graph()
        
        # Add nodes and edges
        for data in network_data:
            skill = data['skill']
            G.add_node(skill, node_type='skill')
            
            for job in data['jobs']:
                G.add_node(job, node_type='job')
                G.add_edge(skill, job)
        
        # Set up the plot
        plt.figure(figsize=(15, 10))
        pos = nx.spring_layout(G)
        
        # Draw nodes
        nx.draw_networkx_nodes(G, pos,
                             nodelist=[n for n, d in G.nodes(data=True) if d['node_type'] == 'skill'],
                             node_color='lightblue',
                             node_size=2000)
        nx.draw_networkx_nodes(G, pos,
                             nodelist=[n for n, d in G.nodes(data=True) if d['node_type'] == 'job'],
                             node_color='lightgreen',
                             node_size=1000)
        
        # Draw edges
        nx.draw_networkx_edges(G, pos, alpha=0.5)
        
        # Draw labels
        nx.draw_networkx_labels(G, pos, font_size=8)
        
        plt.title("Job-Skill Network Visualization")
        plt.axis('off')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Network visualization saved to {output_file}")

def main():
    """Main function to demonstrate Neo4j graph queries."""
    querier = Neo4jGraphQuerier()
    
    try:
        with querier.driver.session() as session:
            # Get and visualize skill network
            logger.info("Getting skill network...")
            network_data = querier.get_skill_network(session)
            if network_data:
                querier.visualize_skill_network(network_data)
            else:
                logger.warning("No skill network data found. Please run build_neo4j_graph.py first.")
            
            # Get requirements for a specific job
            logger.info("\nGetting requirements for 'Software Engineer'...")
            requirements = querier.get_job_requirements(session, "Software Engineer")
            print("\nJob Requirements:")
            print(f"Title: {requirements['title']}")
            print(f"Skills: {', '.join(requirements['skills']) if requirements['skills'] else 'None'}")
            print(f"Experience: {', '.join(requirements['experience']) if requirements['experience'] else 'None'}")
            print(f"Education: {', '.join(requirements['education']) if requirements['education'] else 'None'}")
            
            # Get jobs requiring a specific skill
            logger.info("\nGetting jobs requiring 'Python'...")
            related_jobs = querier.get_related_jobs(session, "Python")
            if related_jobs:
                print("\nJobs requiring Python:")
                for job in related_jobs:
                    print(f"- {job['title']} at {job['company']} ({job['location']})")
            else:
                print("\nNo jobs found requiring Python.")
                
    except Exception as e:
        logger.error(f"Error querying graph: {str(e)}")
        raise
    finally:
        querier.close()

if __name__ == "__main__":
    main() 
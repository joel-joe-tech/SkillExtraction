import sqlite3
import json
import sys
from tabulate import tabulate

def connect_to_db(db_path="jooble_jobs.db"):
    """Connect to the SQLite database"""
    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row  # Row factory for dict-like rows
        return conn
    except sqlite3.Error as e:
        print(f"Error connecting to database: {e}")
        sys.exit(1)

def get_all_jobs(conn):
    """Get all jobs from the database"""
    cursor = conn.cursor()
    cursor.execute("""
    SELECT id, jooble_id, title, company, location 
    FROM jobs
    ORDER BY id
    """)
    return cursor.fetchall()

def get_job_details(conn, job_id):
    """Get detailed information about a job"""
    cursor = conn.cursor()
    
    # Get job base info
    cursor.execute("""
    SELECT * FROM jobs WHERE id = ?
    """, (job_id,))
    job = cursor.fetchone()
    
    if not job:
        return None
    
    # Get job skills
    cursor.execute("""
    SELECT s.name, s.category, js.confidence
    FROM skills s
    JOIN job_skills js ON s.id = js.skill_id
    WHERE js.job_id = ?
    ORDER BY js.confidence DESC
    """, (job_id,))
    skills = cursor.fetchall()
    
    # Get job relationships
    cursor.execute("""
    SELECT relationship_type, entity_text, confidence
    FROM relationships
    WHERE job_id = ?
    ORDER BY confidence DESC
    """, (job_id,))
    relationships = cursor.fetchall()
    
    # Get job qualities (requirements, responsibilities, benefits)
    cursor.execute("""
    SELECT quality_type, description, confidence
    FROM job_qualities
    WHERE job_id = ?
    ORDER BY quality_type, confidence DESC
    """, (job_id,))
    qualities = cursor.fetchall()
    
    return {
        "job": dict(job),
        "skills": [dict(skill) for skill in skills],
        "relationships": [dict(rel) for rel in relationships],
        "qualities": [dict(qual) for qual in qualities]
    }

def get_job_skills_summary(conn):
    """Get summary of skills across all jobs"""
    cursor = conn.cursor()
    cursor.execute("""
    SELECT s.name, s.category, COUNT(js.job_id) as job_count
    FROM skills s
    JOIN job_skills js ON s.id = js.skill_id
    GROUP BY s.id
    ORDER BY job_count DESC, s.name
    LIMIT 20
    """)
    return cursor.fetchall()

def display_job_list(jobs):
    """Display a list of jobs in tabular format"""
    if not jobs:
        print("No jobs found in the database.")
        return
    
    # Format data for tabulate
    table_data = [[job['id'], job['title'], job['company'], job['location']] for job in jobs]
    
    # Print table
    print("\n=== Jobs in Database ===")
    print(tabulate(table_data, headers=["ID", "Title", "Company", "Location"], tablefmt="grid"))
    print(f"Total: {len(jobs)} jobs")

def display_job_details(job_details):
    """Display detailed information about a job"""
    if not job_details:
        print("Job not found.")
        return
    
    job = job_details["job"]
    print("\n=== Job Details ===")
    print(f"Title: {job['title']}")
    print(f"Company: {job['company']}")
    print(f"Location: {job['location']}")
    print(f"Job Type: {job['job_type']}")
    print(f"Source: {job['source']}")
    print(f"Link: {job['link']}")
    print(f"Updated: {job['updated']}")
    
    if job['salary_min'] or job['salary_max']:
        print(f"Salary: {job['salary_min']} - {job['salary_max']} {job['salary_currency']}")
    
    # Display skills grouped by category
    skills = job_details["skills"]
    if skills:
        print("\n=== Skills ===")
        
        # Group skills by category
        skills_by_category = {}
        for skill in skills:
            category = skill['category']
            if category not in skills_by_category:
                skills_by_category[category] = []
            skills_by_category[category].append(skill)
        
        # Print skills by category
        for category, category_skills in skills_by_category.items():
            print(f"\n{category}:")
            skill_table = [[s['name'], f"{s['confidence']:.2f}"] for s in category_skills]
            print(tabulate(skill_table, headers=["Skill", "Confidence"], tablefmt="simple"))
    
    # Display relationships
    relationships = job_details["relationships"]
    if relationships:
        print("\n=== Relationships ===")
        rel_table = [[r['relationship_type'], r['entity_text'], f"{r['confidence']:.2f}"] for r in relationships]
        print(tabulate(rel_table, headers=["Type", "Entity", "Confidence"], tablefmt="simple"))
    
    # Display job qualities
    qualities = job_details["qualities"]
    if qualities:
        # Group by quality type
        quality_groups = {}
        for q in qualities:
            if q['quality_type'] not in quality_groups:
                quality_groups[q['quality_type']] = []
            quality_groups[q['quality_type']].append(q)
        
        for quality_type, items in quality_groups.items():
            print(f"\n=== {quality_type.title()} ===")
            for item in items:
                print(f"• {item['description']} ({item['confidence']:.2f})")

def display_skills_summary(skills):
    """Display summary of skills across all jobs"""
    if not skills:
        print("No skills found in the database.")
        return
    
    print("\n=== Top Skills ===")
    skill_table = [[s['name'], s['category'], s['job_count']] for s in skills]
    print(tabulate(skill_table, headers=["Skill", "Category", "Job Count"], tablefmt="grid"))

def export_job_network(conn, output_file="job_network.json"):
    """Export job relationships as a network graph"""
    cursor = conn.cursor()
    
    # Get all jobs
    cursor.execute("SELECT id, title, company FROM jobs")
    jobs = cursor.fetchall()
    
    # Get all relationships
    cursor.execute("""
    SELECT j.id as job_id, j.title, r.relationship_type, r.entity_text
    FROM jobs j
    JOIN relationships r ON j.id = r.job_id
    ORDER BY j.id
    """)
    relationships = cursor.fetchall()
    
    # Get all job-skill relationships
    cursor.execute("""
    SELECT j.id as job_id, j.title, s.name as skill_name, s.category
    FROM jobs j
    JOIN job_skills js ON j.id = js.job_id
    JOIN skills s ON js.skill_id = s.id
    ORDER BY j.id
    """)
    job_skills = cursor.fetchall()
    
    # Create network data
    nodes = []
    links = []
    
    # Add job nodes
    for job in jobs:
        nodes.append({
            "id": f"job_{job['id']}",
            "name": job['title'],
            "type": "job",
            "company": job['company']
        })
    
    # Add relationship entity nodes and links
    entity_nodes = set()
    for rel in relationships:
        entity_id = f"entity_{rel['entity_text'].lower().replace(' ', '_')}"
        
        if entity_id not in entity_nodes:
            nodes.append({
                "id": entity_id,
                "name": rel['entity_text'],
                "type": "entity",
                "entity_type": rel['relationship_type']
            })
            entity_nodes.add(entity_id)
        
        links.append({
            "source": f"job_{rel['job_id']}",
            "target": entity_id,
            "type": rel['relationship_type']
        })
    
    # Add skill nodes and links
    skill_nodes = set()
    for js in job_skills:
        skill_id = f"skill_{js['skill_name'].lower().replace(' ', '_')}"
        
        if skill_id not in skill_nodes:
            nodes.append({
                "id": skill_id,
                "name": js['skill_name'],
                "type": "skill",
                "category": js['category']
            })
            skill_nodes.add(skill_id)
        
        links.append({
            "source": f"job_{js['job_id']}",
            "target": skill_id,
            "type": "HAS_SKILL"
        })
    
    # Create network data
    network_data = {
        "nodes": nodes,
        "links": links
    }
    
    # Save to file
    with open(output_file, 'w') as f:
        json.dump(network_data, f, indent=2)
    
    print(f"\nNetwork data exported to {output_file}")
    print(f"Nodes: {len(nodes)}, Links: {len(links)}")

def main():
    """Main function"""
    conn = connect_to_db()
    
    while True:
        print("\n=== Job Relationship Explorer ===")
        print("1. List all jobs")
        print("2. View job details")
        print("3. View skills summary")
        print("4. Export job network")
        print("5. Exit")
        
        choice = input("\nEnter your choice (1-5): ")
        
        if choice == "1":
            jobs = get_all_jobs(conn)
            display_job_list(jobs)
            
        elif choice == "2":
            job_id = input("Enter job ID: ")
            try:
                job_id = int(job_id)
                job_details = get_job_details(conn, job_id)
                display_job_details(job_details)
            except ValueError:
                print("Invalid job ID. Please enter a number.")
            
        elif choice == "3":
            skills = get_job_skills_summary(conn)
            display_skills_summary(skills)
            
        elif choice == "4":
            output_file = input("Enter output filename [job_network.json]: ") or "job_network.json"
            export_job_network(conn, output_file)
            
        elif choice == "5":
            break
            
        else:
            print("Invalid choice. Please try again.")
    
    conn.close()
    print("Goodbye!")

if __name__ == "__main__":
    try:
        # Check if tabulate is installed without reimporting
        main()
    except ImportError:
        print("This script requires the 'tabulate' package.")
        print("Please install it with: pip install tabulate")
        sys.exit(1) 
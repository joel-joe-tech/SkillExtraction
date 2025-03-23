import http.client
import json
import time
import os

def fetch_jooble_jobs(keywords, location=""):
    host = 'jooble.org'
    key = '6aca5242-9a2e-4b00-9a81-420bc53f3888'
    
    connection = http.client.HTTPConnection(host)
    headers = {"Content-type": "application/json"}
    
    # Create query with keywords and location
    body = json.dumps({
        "keywords": keywords,
        "location": location
    })
    
    print(f"\nFetching jobs for: {keywords} in {location if location else 'any location'}")
    connection.request('POST', '/api/' + key, body, headers)
    response = connection.getresponse()
    print(f"Response: {response.status} {response.reason}")
    
    response_data = response.read().decode('utf-8')
    return response_data

# Define technical keywords to search for
tech_keywords = [
    "software engineer",
    "developer",
    "data scientist",
    "full stack",
    "devops",
    "cloud engineer",
    "machine learning"
]

# Define locations to search in
locations = ["", "Remote", "Switzerland"]  # Empty string means any location

# Delete existing database to start fresh
db_path = "jooble_jobs.db"
if os.path.exists(db_path):
    try:
        os.remove(db_path)
        print(f"Removed existing database: {db_path}")
    except OSError as e:
        print(f"Could not remove existing database: {str(e)}")
        print("Please close any applications that might be using the database file.")
        exit(1)

# Fetch and process jobs for each keyword and location
all_jobs = []
for keyword in tech_keywords:
    for location in locations:
        try:
            # Fetch jobs
            response_data = fetch_jooble_jobs(keyword, location)
            
            # Parse the response
            job_data = json.loads(response_data)
            
            # Get jobs from the response
            jobs = job_data.get("jobs", [])
            print(f"Found {len(jobs)} jobs for '{keyword}' in '{location if location else 'any location'}'")
            
            # Add to our collection
            all_jobs.extend(jobs)
            
            # Avoid rate limiting
            if keyword != tech_keywords[-1] or location != locations[-1]:
                print("Waiting 2 seconds before next request...")
                time.sleep(2)
        except Exception as e:
            print(f"Error fetching jobs for '{keyword}' in '{location}': {str(e)}")

# Remove duplicates (same job might appear in multiple searches)
unique_jobs = {}
for job in all_jobs:
    job_id = job.get("id", "")
    if job_id and job_id not in unique_jobs:
        unique_jobs[job_id] = job

print(f"\nTotal unique jobs fetched: {len(unique_jobs)}")

# Create a combined dataset
combined_data = {"jobs": list(unique_jobs.values())}

# Save the combined data to a file
with open("tech_jobs_data.json", "w") as f:
    json.dump(combined_data, f, indent=2)
    print("Response saved to tech_jobs_data.json")

# Try to process with our job extraction model
try:
    import job_extraction_model
    print("\nProcessing jobs with ML model...")
    stats = job_extraction_model.process_jooble_response(combined_data)
    print(f"Processing complete: {stats}")
    print("\nJob data has been extracted and stored in the database (jooble_jobs.db)")
    print("You can query the database to see the extracted relationships")
    print("\nTo view job details and skills, run: python view_job_relationships.py")
    print("To visualize the job network, run: python visualize_job_network.py")
except ImportError:
    print("\nTo process this data with the ML model, run: python job_extraction_model.py")
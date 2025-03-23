# Skill Extraction from Job Postings

An ML-powered system for extracting structured data, skills, and relationships from job listings using NLP techniques.

## Overview

This project fetches job listings from the Jooble API and processes them using an NLP model to extract:

- Key job details (title, company, location, salary)
- Technical skills and their categories
- Job relationships (requires, prefers, reports to, etc.)
- Job qualities (requirements, responsibilities, benefits)

The extracted data is stored in a SQLite relational database for further analysis and visualization.

## Features

- **API Integration**: Fetches job listings from Jooble API based on technical keywords
- **ML Processing Pipeline**: Utilizes spaCy NLP for named entity recognition and skill extraction
- **Skill Matching**: Custom `SkillMatcher` component identifies technical skills from job descriptions
- **Relationship Extraction**: Maps connections between jobs, skills, and entities
- **Data Storage**: Stores structured data in a relational SQLite database
- **Visualization**: Tools to explore job relationships and skill networks

## Requirements

- Python 3.7+
- spaCy
- SQLite3
- matplotlib (for visualization)
- NetworkX (for network visualization)

## Installation

```bash
# Clone the repository
git clone https://github.com/joel-joe-tech/SkillExtraction.git
cd SkillExtraction

# Install dependencies
pip install -r requirements.txt

# Download spaCy model
python -m spacy download en_core_web_sm
```

## Usage

1. Run the main script to fetch and process job listings:
   ```bash
   python test.py
   ```

2. View extracted job relationships:
   ```bash
   python view_job_relationships.py
   ```

3. Visualize the job network:
   ```bash
   python visualize_job_network.py
   ```

## Project Structure

- `test.py` - Main script for fetching job listings from Jooble API
- `job_extraction_model.py` - NLP model for processing job listings and extracting structured data
- `view_job_relationships.py` - Script for querying and displaying extracted job data
- `visualize_job_network.py` - Script for visualizing job-skill relationships as a network graph
- `restart_connections.py` - Utility script for managing database connections

## License

MIT

## Author

Joel Joe

## Acknowledgements

- Jooble API for providing job data
- spaCy for NLP capabilities 
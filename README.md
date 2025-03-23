# Job Post Extraction and Relationship Analysis

This system extracts structured data from job postings using ML techniques and organizes the information into a relational database. It can identify skills, job requirements, responsibilities, and relationships between entities in job descriptions.

## Features

- Extracts job data from Jooble API
- Uses NLP and ML techniques to identify skills, relationships, and job qualities
- Stores data in a SQLite relational database
- Provides tools to query and visualize the extracted information
- Exports network data for visualization

## Installation

1. Make sure you have Python 3.7+ installed
2. Install the required packages:

```bash
pip install spacy tabulate
python -m spacy download en_core_web_md
```

## Usage

### Fetching and Processing Job Data

1. Run the test.py script to fetch job data from the Jooble API:

```bash
python test.py
```

This will:
- Fetch job postings from Jooble API
- Save the response to test_output.json
- Process the jobs using the ML model
- Store the extracted data in jooble_jobs.db

### Viewing Extracted Data

Use the view_job_relationships.py script to explore the extracted data:

```bash
python view_job_relationships.py
```

This interactive tool allows you to:
- List all jobs in the database
- View detailed information about a specific job
- See summary statistics about skills
- Export job relationship network data for visualization

### Visualizing the Data

The exported network data (job_network.json) can be visualized using various network visualization tools like:
- [D3.js](https://d3js.org/) for web-based visualization
- [Gephi](https://gephi.org/) for desktop visualization
- [NetworkX](https://networkx.org/) + matplotlib for Python-based visualization

## Database Schema

The system uses a relational database with the following structure:

- **jobs**: Basic job information (title, company, location, etc.)
- **skills**: Unique skills extracted from job descriptions
- **job_skills**: Many-to-many relationship between jobs and skills
- **relationships**: Relationships between jobs and other entities (companies, locations)
- **job_qualities**: Job qualities like requirements, responsibilities, and benefits

## How It Works

1. **Data Extraction**: The system fetches job data from the Jooble API
2. **NLP Processing**: Uses spaCy for entity recognition and custom extractors for relationships
3. **Pattern Matching**: Uses regex and pattern matching to identify skills and job qualities
4. **Data Storage**: Stores the structured data in a relational database
5. **Relationship Extraction**: Identifies relationships between jobs and other entities

## Customization

You can customize the skill patterns and extraction rules in the `job_extraction_model.py` file:

- `SKILL_PATTERNS`: Dictionary of skill categories and keywords
- `RelationshipExtractor`: Class for identifying relationships
- `extract_job_qualities`: Function for extracting requirements, responsibilities, etc.

## License

This project is licensed under the MIT License 
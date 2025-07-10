# job_embedder.py - Enhanced version without duplicates

"""
Utilities for loading and embedding job descriptions.
"""

import os
from typing import List, Dict
from pinecone_handler import PineconeHandler
from embedding_utils import embed_texts
import json


def load_job_descriptions(directory: str) -> Dict[str, Dict[str, str]]:
    """Load and parse all job description text files from a directory."""
    job_data = {}
    
    for fname in os.listdir(directory):
        if fname.endswith('.txt'):
            path = os.path.join(directory, fname)
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()
                
                # Parse job description structure
                parsed_job = parse_job_description(content, fname)
                job_data[fname] = parsed_job
    
    return job_data


def parse_job_description(content: str, filename: str) -> Dict[str, str]:
    """Parse a job description text into structured format."""
    lines = content.split('\n')
    
    # Initialize fields
    company = ""
    job_title = filename.replace('.txt', '').replace('_', ' ').title()
    description = ""
    responsibilities = []
    requirements = []
    
    current_section = None
    
    for line in lines:
        line = line.strip()
        
        if line.startswith("Company:"):
            company = line.replace("Company:", "").strip()
        elif line == "Job Description:":
            current_section = "description"
        elif line == "Responsibilities:":
            current_section = "responsibilities"
        elif line == "Requirements:":
            current_section = "requirements"
        elif line and current_section:
            if current_section == "description":
                description += line + " "
            elif current_section == "responsibilities" and line.startswith("-"):
                responsibilities.append(line[1:].strip())
            elif current_section == "requirements" and line.startswith("-"):
                requirements.append(line[1:].strip())
    
    # Create searchable full text
    full_text = f"{job_title} {company} {description} {' '.join(responsibilities)} {' '.join(requirements)}"
    
    return {
        "job_title": job_title,
        "company": company,
        "description": description.strip(),
        "responsibilities": responsibilities,
        "requirements": requirements,
        "responsibilities_text": " • ".join(responsibilities),
        "requirements_text": " • ".join(requirements),
        "full_text": full_text,
        "filename": filename
    }


def batch_embed_and_upload(directory: str = "job_description"):
    """Load, embed, and upload all job descriptions to Pinecone."""
    print(f"Loading job descriptions from {directory}...")
    job_data = load_job_descriptions(directory)
    
    if not job_data:
        print("No job descriptions found!")
        return
    
    print(f"Found {len(job_data)} job descriptions")
    
    # Initialize Pinecone handler
    handler = PineconeHandler()
    
    # Prepare batch upload
    items_to_upload = []
    
    for fname, data in job_data.items():
        item = {
            "id": fname,
            "text": data["full_text"],
            "metadata": {
                "job_title": data["job_title"],
                "company": data["company"],
                "description": data["description"],
                "requirements": data["requirements_text"],
                "responsibilities": data["responsibilities_text"],
                "filename": fname,
                "req_count": len(data["requirements"]),
                "resp_count": len(data["responsibilities"])
            }
        }
        items_to_upload.append(item)
    
    # Upload in batches
    batch_size = 10
    for i in range(0, len(items_to_upload), batch_size):
        batch = items_to_upload[i:i + batch_size]
        handler.upsert_batch(batch)
        print(f"Uploaded batch {i//batch_size + 1}/{(len(items_to_upload) + batch_size - 1)//batch_size}")
    
    print(f"✅ Successfully uploaded {len(job_data)} job descriptions to Pinecone")
    
    # Save metadata locally for reference
    metadata_file = os.path.join(directory, "job_metadata.json")
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(job_data, f, indent=2)
    print(f"📄 Saved job metadata to {metadata_file}")


def search_jobs_by_skills(skills: List[str], top_k: int = 5):
    """Search for jobs matching specific skills."""
    handler = PineconeHandler()
    
    # Create search query from skills
    search_query = " ".join(skills)
    
    # Search in Pinecone
    results = handler.search_similar(search_query, top_k=top_k)
    
    return results


if __name__ == "__main__":
    # Run this script to upload all job descriptions to Pinecone
    import sys
    
    if len(sys.argv) > 1:
        directory = sys.argv[1]
    else:
        directory = "job_description"
    
    batch_embed_and_upload(directory)
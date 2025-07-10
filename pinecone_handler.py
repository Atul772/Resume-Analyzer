# pinecone_handler.py - Enhanced Version

"""
Handles Pinecone index setup, upsert, and semantic search for job descriptions and resumes.
Enhanced with better error handling, batch processing, and metadata management.
"""

import os
from pinecone import Pinecone, ServerlessSpec
from typing import List, Dict, Any, Optional, Tuple
from dotenv import load_dotenv
from embedding_utils import embed_text, embed_texts
import time

load_dotenv()

class PineconeHandler:
    """
    Enhanced wrapper for Pinecone index operations with improved functionality.
    """
    def __init__(self):
        self.index_name = os.getenv("PINECONE_INDEX", "resume-analyzer")
        api_key = os.getenv("PINECONE_API_KEY")
        
        if not api_key:
            raise ValueError("PINECONE_API_KEY not found in environment variables")
        
        # Initialize Pinecone
        self.pc = Pinecone(api_key=api_key)
        
        # Check if index exists, create if not
        self._ensure_index_exists()
        
        # Connect to index
        self.index = self.pc.Index(self.index_name)
        
        # Get index stats on initialization
        self._initial_stats = self.get_index_stats()
        print(f"✅ Connected to Pinecone index '{self.index_name}' with {self._initial_stats.get('total_vector_count', 0)} vectors")

    def _ensure_index_exists(self):
        """Ensure the index exists, create if it doesn't."""
        try:
            existing_indexes = [index.name for index in self.pc.list_indexes()]
            
            if self.index_name not in existing_indexes:
                print(f"Creating index '{self.index_name}'...")
                self.pc.create_index(
                    name=self.index_name,
                    dimension=768,  # dimension for all-mpnet-base-v2
                    metric='cosine',
                    spec=ServerlessSpec(
                        cloud='aws',
                        region=os.getenv('PINECONE_ENV', 'us-east-1')
                    )
                )
                print(f"✅ Created index '{self.index_name}'")
                # Wait for index to be ready
                time.sleep(5)
        except Exception as e:
            print(f"Warning: Could not check/create index: {e}")

    def upsert_job_description(self, job_id: str, job_title: str, 
                              job_description: str, company: str = "",
                              requirements: str = "", metadata: Dict = None) -> bool:
        """
        Add a job description to Pinecone index with enhanced metadata.
        """
        try:
            # Create comprehensive text for embedding
            full_text = f"{job_title} at {company}. {job_description} Requirements: {requirements}".strip()
            
            # Generate embedding
            embedding = embed_text(full_text)
            
            # Prepare metadata
            job_metadata = {
                "job_title": job_title,
                "company": company,
                "description": job_description[:1000],  # Truncate for metadata limits
                "requirements": requirements[:1000],
                "full_text": full_text[:2000],
                "char_count": len(full_text),
                "has_requirements": bool(requirements),
                "timestamp": int(time.time())
            }
            
            # Add custom metadata if provided
            if metadata:
                job_metadata.update(metadata)
            
            # Upsert to Pinecone
            self.index.upsert([(job_id, embedding.tolist(), job_metadata)])
            print(f"✅ Job '{job_title}' at '{company}' added to Pinecone index")
            return True
            
        except Exception as e:
            print(f"❌ Error upserting job description: {e}")
            return False

    def upsert_batch(self, items: List[Dict[str, Any]]) -> Tuple[int, int]:
        """
        Upsert a batch of job descriptions or resumes with error handling.
        Returns (success_count, failure_count)
        """
        if not items:
            return 0, 0
        
        vectors = []
        success_count = 0
        failure_count = 0
        
        for item in items:
            try:
                # Validate required fields
                if 'id' not in item or 'text' not in item:
                    print(f"⚠️ Skipping item: missing 'id' or 'text' field")
                    failure_count += 1
                    continue
                
                # Generate embedding
                emb = embed_text(item['text'])
                
                # Prepare metadata
                metadata = item.get('metadata', {})
                metadata['text_preview'] = item['text'][:500]  # Add text preview
                metadata['timestamp'] = int(time.time())
                
                vectors.append((
                    item['id'], 
                    emb.tolist(), 
                    metadata
                ))
                
            except Exception as e:
                print(f"⚠️ Error processing item {item.get('id', 'unknown')}: {e}")
                failure_count += 1
        
        # Upsert in batches of 100 (Pinecone limit)
        batch_size = 100
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i + batch_size]
            try:
                self.index.upsert(batch)
                success_count += len(batch)
            except Exception as e:
                print(f"❌ Error upserting batch: {e}")
                failure_count += len(batch)
        
        print(f"✅ Upserted {success_count} items successfully, {failure_count} failures")
        return success_count, failure_count

    def search_similar(self, query_text: str, top_k: int = 5, 
                      filter_dict: Dict = None, include_metadata: bool = True) -> List[Dict]:
        """
        Enhanced semantic search with filtering options.
        """
        try:
            # Generate query embedding
            query_emb = embed_text(query_text)
            
            # Prepare query parameters
            query_params = {
                "vector": query_emb.tolist(),
                "top_k": top_k,
                "include_metadata": include_metadata
            }
            
            # Add filter if provided
            if filter_dict:
                query_params["filter"] = filter_dict
            
            # Execute query
            results = self.index.query(**query_params)
            
            # Process and return matches
            matches = []
            for match in results.get('matches', []):
                match_data = {
                    "id": match['id'],
                    "score": round(match['score'] * 100, 2),  # Convert to percentage
                    "metadata": match.get('metadata', {})
                }
                
                # Add match quality indicator
                if match_data['score'] >= 80:
                    match_data['match_quality'] = 'Excellent'
                elif match_data['score'] >= 65:
                    match_data['match_quality'] = 'Good'
                elif match_data['score'] >= 50:
                    match_data['match_quality'] = 'Fair'
                else:
                    match_data['match_quality'] = 'Low'
                
                matches.append(match_data)
            
            return matches
            
        except Exception as e:
            print(f"❌ Error searching: {e}")
            return []

    def search_by_skills(self, skills: List[str], top_k: int = 5) -> List[Dict]:
        """
        Search for jobs matching specific skills.
        """
        # Create a weighted query from skills
        skill_query = " ".join(skills)
        # Emphasize the skills by repeating important ones
        if len(skills) > 3:
            # Repeat first 3 skills for emphasis
            skill_query = " ".join(skills[:3]) + " " + skill_query
        
        return self.search_similar(skill_query, top_k=top_k)

    def hybrid_search(self, resume_text: str, skills: List[str], 
                     experience_level: str = None, top_k: int = 5) -> List[Dict]:
        """
        Perform hybrid search using both resume text and extracted skills.
        """
        # Weight different components
        weighted_query_parts = []
        
        # Add resume text (lower weight)
        weighted_query_parts.append(resume_text[:500])  # Use first 500 chars
        
        # Add skills (higher weight - repeat for emphasis)
        if skills:
            skills_text = " ".join(skills)
            weighted_query_parts.extend([skills_text] * 2)  # Repeat twice for emphasis
        
        # Add experience level keywords if provided
        if experience_level:
            level_keywords = {
                "entry": "entry level junior fresh graduate intern",
                "mid": "mid-level experienced professional 2-5 years",
                "senior": "senior lead principal expert 5+ years",
                "executive": "executive director VP head chief"
            }
            if experience_level.lower() in level_keywords:
                weighted_query_parts.append(level_keywords[experience_level.lower()])
        
        # Combine all parts
        final_query = " ".join(weighted_query_parts)
        
        return self.search_similar(final_query, top_k=top_k)

    def get_index_stats(self) -> Dict[str, Any]:
        """Get detailed statistics about the Pinecone index."""
        try:
            stats = self.index.describe_index_stats()
            
            # Process stats for better readability
            processed_stats = {
                "total_vector_count": stats.get('total_vector_count', 0),
                "dimension": stats.get('dimension', 768),
                "index_fullness": stats.get('index_fullness', 0),
                "namespaces": stats.get('namespaces', {})
            }
            
            return processed_stats
            
        except Exception as e:
            print(f"❌ Error getting index stats: {e}")
            return {}

    def update_job_metadata(self, job_id: str, metadata_updates: Dict) -> bool:
        """
        Update metadata for an existing job without changing the embedding.
        """
        try:
            # Fetch existing vector
            fetch_response = self.index.fetch([job_id])
            
            if job_id not in fetch_response['vectors']:
                print(f"Job ID {job_id} not found in index")
                return False
            
            # Get existing data
            existing_vector = fetch_response['vectors'][job_id]
            existing_metadata = existing_vector.get('metadata', {})
            
            # Update metadata
            existing_metadata.update(metadata_updates)
            existing_metadata['last_updated'] = int(time.time())
            
            # Upsert with same vector but updated metadata
            self.index.upsert([(
                job_id,
                existing_vector['values'],
                existing_metadata
            )])
            
            print(f"✅ Updated metadata for job {job_id}")
            return True
            
        except Exception as e:
            print(f"❌ Error updating metadata: {e}")
            return False

    def delete_job(self, job_id: str) -> bool:
        """Delete a specific job from the index."""
        try:
            self.index.delete(ids=[job_id])
            print(f"✅ Deleted job {job_id} from index")
            return True
        except Exception as e:
            print(f"❌ Error deleting job: {e}")
            return False

    def delete_all_vectors(self):
        """Delete all vectors from the index (use with caution)."""
        try:
            self.index.delete(delete_all=True)
            print("🗑️ All vectors deleted from index")
        except Exception as e:
            print(f"❌ Error deleting all vectors: {e}")

    def list_job_ids(self, limit: int = 100) -> List[str]:
        """
        List job IDs in the index (for maintenance/debugging).
        Note: This is a workaround as Pinecone doesn't have a direct list function.
        """
        try:
            # Use a random vector to query and get results
            dummy_vector = [0.0] * 768  # Zero vector
            results = self.index.query(
                vector=dummy_vector,
                top_k=limit,
                include_metadata=False
            )
            job_ids = [match['id'] for match in results.get('matches', [])]
            return job_ids
        except Exception as e:
            print(f"❌ Error listing job IDs: {e}")
            return []

    def find_duplicate_jobs(self, threshold: float = 0.95) -> List[Tuple[str, str, float]]:
        """
        Find potential duplicate job postings based on similarity threshold.
        Returns list of (job_id1, job_id2, similarity_score) tuples.
        """
        duplicates = []
        job_ids = self.list_job_ids(limit=50)  # Check first 50 jobs
        
        for i, job_id in enumerate(job_ids):
            # Fetch the job vector
            fetch_response = self.index.fetch([job_id])
            if job_id not in fetch_response['vectors']:
                continue
                
            job_vector = fetch_response['vectors'][job_id]['values']
            
            # Search for similar jobs
            results = self.index.query(
                vector=job_vector,
                top_k=5,
                include_metadata=True
            )
            
            # Check for high similarity matches (excluding self)
            for match in results['matches']:
                if match['id'] != job_id and match['score'] >= threshold:
                    # Avoid duplicate pairs (A,B) and (B,A)
                    pair = tuple(sorted([job_id, match['id']]))
                    if pair not in [tuple(sorted([d[0], d[1]])) for d in duplicates]:
                        duplicates.append((job_id, match['id'], round(match['score'], 3)))
        
        return duplicates

    def backup_index_metadata(self, output_file: str = "pinecone_backup.json") -> bool:
        """
        Backup all job metadata to a JSON file for disaster recovery.
        """
        import json
        
        try:
            all_metadata = []
            job_ids = self.list_job_ids(limit=1000)
            
            # Fetch in batches
            batch_size = 100
            for i in range(0, len(job_ids), batch_size):
                batch_ids = job_ids[i:i + batch_size]
                fetch_response = self.index.fetch(batch_ids)
                
                for job_id, vector_data in fetch_response['vectors'].items():
                    metadata = vector_data.get('metadata', {})
                    metadata['_id'] = job_id
                    all_metadata.append(metadata)
            
            # Save to file
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(all_metadata, f, indent=2)
            
            print(f"✅ Backed up {len(all_metadata)} job entries to {output_file}")
            return True
            
        except Exception as e:
            print(f"❌ Error backing up metadata: {e}")
            return False

    def get_similar_jobs_by_job_id(self, job_id: str, top_k: int = 5) -> List[Dict]:
        """
        Find jobs similar to a specific job ID.
        """
        try:
            # Fetch the job vector
            fetch_response = self.index.fetch([job_id])
            
            if job_id not in fetch_response['vectors']:
                print(f"Job ID {job_id} not found in index")
                return []
            
            job_vector = fetch_response['vectors'][job_id]['values']
            job_metadata = fetch_response['vectors'][job_id].get('metadata', {})
            
            # Search for similar jobs
            results = self.index.query(
                vector=job_vector,
                top_k=top_k + 1,  # +1 to exclude self
                include_metadata=True
            )
            
            # Process results (exclude self)
            similar_jobs = []
            for match in results['matches']:
                if match['id'] != job_id:
                    similar_jobs.append({
                        "id": match['id'],
                        "score": round(match['score'] * 100, 2),
                        "metadata": match.get('metadata', {}),
                        "match_quality": self._get_match_quality(match['score'])
                    })
            
            return similar_jobs[:top_k]
            
        except Exception as e:
            print(f"❌ Error finding similar jobs: {e}")
            return []

    def _get_match_quality(self, score: float) -> str:
        """Determine match quality based on score."""
        if score >= 0.80:
            return 'Excellent'
        elif score >= 0.65:
            return 'Good'
        elif score >= 0.50:
            return 'Fair'
        else:
            return 'Low'

    def search_with_feedback(self, query_text: str, liked_jobs: List[str] = None, 
                           disliked_jobs: List[str] = None, top_k: int = 5) -> List[Dict]:
        """
        Enhanced search that learns from user feedback (liked/disliked jobs).
        """
        try:
            # Start with base query embedding
            query_emb = embed_text(query_text).tolist()
            
            # Adjust embedding based on feedback
            if liked_jobs:
                # Fetch liked job vectors
                liked_response = self.index.fetch(liked_jobs)
                for job_id in liked_jobs:
                    if job_id in liked_response['vectors']:
                        liked_vector = liked_response['vectors'][job_id]['values']
                        # Add a portion of liked vectors to query (positive reinforcement)
                        query_emb = [q + 0.1 * l for q, l in zip(query_emb, liked_vector)]
            
            if disliked_jobs:
                # Fetch disliked job vectors
                disliked_response = self.index.fetch(disliked_jobs)
                for job_id in disliked_jobs:
                    if job_id in disliked_response['vectors']:
                        disliked_vector = disliked_response['vectors'][job_id]['values']
                        # Subtract a portion of disliked vectors (negative reinforcement)
                        query_emb = [q - 0.05 * d for q, d in zip(query_emb, disliked_vector)]
            
            # Normalize the adjusted embedding
            norm = sum(x**2 for x in query_emb) ** 0.5
            query_emb = [x / norm for x in query_emb]
            
            # Execute search with adjusted query
            results = self.index.query(
                vector=query_emb,
                top_k=top_k,
                include_metadata=True
            )
            
            # Process results
            matches = []
            for match in results.get('matches', []):
                matches.append({
                    "id": match['id'],
                    "score": round(match['score'] * 100, 2),
                    "metadata": match.get('metadata', {}),
                    "match_quality": self._get_match_quality(match['score'])
                })
            
            return matches
            
        except Exception as e:
            print(f"❌ Error in feedback-based search: {e}")
            return self.search_similar(query_text, top_k=top_k)  # Fallback to regular search

    def get_job_recommendations_for_profile(self, resume_data: Dict[str, Any], 
                                          top_k: int = 10) -> Dict[str, List[Dict]]:
        """
        Get comprehensive job recommendations based on complete resume profile.
        Returns categorized recommendations.
        """
        recommendations = {
            "best_matches": [],
            "skill_development": [],
            "career_growth": []
        }
        
        try:
            # Extract key information
            skills = resume_data.get("skills", [])
            experience = resume_data.get("experience", [])
            education = resume_data.get("education", [])
            
            # 1. Best matches based on current profile
            if skills:
                current_profile_query = " ".join(skills)
                if experience:
                    # Add recent job title
                    if isinstance(experience[0], dict):
                        current_profile_query += f" {experience[0].get('job_title', '')}"
                
                best_matches = self.search_similar(current_profile_query, top_k=top_k//2)
                recommendations["best_matches"] = best_matches
            
            # 2. Skill development opportunities (slightly above current level)
            if skills and len(skills) >= 3:
                # Query with enhanced skills
                growth_query = " ".join(skills) + " senior lead advanced"
                skill_dev_matches = self.search_similar(growth_query, top_k=top_k//3)
                recommendations["skill_development"] = skill_dev_matches
            
            # 3. Career growth opportunities
            if experience:
                # Look for next-level positions
                growth_keywords = ["manager", "lead", "senior", "principal", "architect"]
                career_query = " ".join(skills[:5]) + " " + " ".join(growth_keywords)
                career_matches = self.search_similar(career_query, top_k=top_k//3)
                recommendations["career_growth"] = career_matches
            
            return recommendations
            
        except Exception as e:
            print(f"❌ Error getting job recommendations: {e}")
            return recommendations


# Utility functions for index management
def init_pinecone_index():
    """Initialize Pinecone index with job descriptions."""
    try:
        handler = PineconeHandler()
        stats = handler.get_index_stats()
        print(f"📊 Index stats: {stats}")
        return handler
    except Exception as e:
        print(f"❌ Error initializing Pinecone: {e}")
        return None


def cleanup_duplicates(handler: PineconeHandler, threshold: float = 0.95):
    """Find and optionally remove duplicate job postings."""
    duplicates = handler.find_duplicate_jobs(threshold)
    
    if duplicates:
        print(f"Found {len(duplicates)} potential duplicates:")
        for job1, job2, score in duplicates:
            print(f"  - {job1} <-> {job2} (similarity: {score})")
        
        # You can add logic here to remove duplicates if needed
    else:
        print("No duplicates found!")


if __name__ == "__main__":
    # Test the handler
    handler = init_pinecone_index()
    
    if handler:
        # Show index stats
        stats = handler.get_index_stats()
        print(f"Total vectors in index: {stats.get('total_vector_count', 0)}")
        
        # Test search
        test_query = "python developer machine learning"
        print(f"\nSearching for: {test_query}")
        results = handler.search_similar(test_query, top_k=3)
        
        for i, result in enumerate(results, 1):
            print(f"\n{i}. {result['id']} (Score: {result['score']}%)")
            print(f"   Title: {result['metadata'].get('job_title', 'N/A')}")
            print(f"   Company: {result['metadata'].get('company', 'N/A')}")
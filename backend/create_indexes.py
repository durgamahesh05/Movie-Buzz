import os
import pymongo
from pymongo import MongoClient
from dotenv import load_dotenv

def main():
    load_dotenv()
    MONGO_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
    kwargs = {}
    if MONGO_URI.startswith("mongodb+srv://"):
        kwargs["tls"] = True
        kwargs["tlsAllowInvalidCertificates"] = True

    client = MongoClient(MONGO_URI, **kwargs)
    db = client['moviebuzz']

    print("Creating MongoDB Indexes...")
    
    # 1. Movies index
    print("Indexing movies (movieId: unique)...")
    db.movies.create_index('movieId', unique=True)
    
    # 2. Genome Scores index
    print("Indexing genome_scores (movieId, tagId)...")
    db.genome_scores.create_index([('movieId', 1), ('tagId', 1)])
    
    # 3. Ratings index
    print("Indexing ratings (userId, movieId)...")
    db.ratings.create_index([('userId', 1), ('movieId', 1)])
    
    # 4. OMDB Cache index
    print("Indexing omdb_cache (imdbId: unique)...")
    db.omdb_cache.create_index('imdbId', unique=True)
    
    print("Indexes created successfully!")

if __name__ == '__main__':
    main()

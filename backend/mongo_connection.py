"""
🔗 MongoDB Connection Setup & Guide
Complete guide to connect to MongoDB Atlas or local MongoDB
"""

import os
import logging
from typing import Optional, Dict, Any
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════════
#  MONGODB CONNECTION
# ═══════════════════════════════════════════════════════════════════════════════

class MongoDBConnection:
    """
    MongoDB Connection Manager for MovieBuzz
    
    Supports:
    - MongoDB Atlas (cloud)
    - Local MongoDB (development)
    """
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self.client = None
        self.db = None
        self._initialized = True
    
    def connect_atlas(self, uri: str, db_name: str = "moviebuzz") -> bool:
        """
        Connect to MongoDB Atlas (cloud)
        
        Args:
            uri: MongoDB Atlas connection string
                Format: mongodb+srv://user:password@cluster.mongodb.net/?retryWrites=true&w=majority
            db_name: Database name
        
        Returns:
            bool: Connection successful
        """
        try:
            log.info("🔗 Connecting to MongoDB Atlas...")
            kwargs = {"serverSelectionTimeoutMS": 5000}
            if uri.startswith("mongodb+srv://"):
                kwargs["tls"] = True
                kwargs["tlsAllowInvalidCertificates"] = True
            
            self.client = MongoClient(uri, **kwargs)
            
            # Test connection
            self.client.admin.command('ping')
            
            self.db = self.client[db_name]
            log.info("✅ Connected to MongoDB Atlas successfully!")
            return True
            
        except ServerSelectionTimeoutError:
            log.error("❌ Connection timeout - check your URI and network")
            return False
        except ConnectionFailure as e:
            log.error(f"❌ Connection failed: {e}")
            return False
        except Exception as e:
            log.error(f"❌ Unexpected error: {e}")
            return False
    
    def connect_local(self, host: str = "localhost", port: int = 27017, 
                     db_name: str = "moviebuzz") -> bool:
        """
        Connect to local MongoDB
        
        Args:
            host: MongoDB host (default: localhost)
            port: MongoDB port (default: 27017)
            db_name: Database name
        
        Returns:
            bool: Connection successful
        """
        try:
            log.info(f"🔗 Connecting to local MongoDB ({host}:{port})...")
            uri = f"mongodb://{host}:{port}/"
            self.client = MongoClient(uri, serverSelectionTimeoutMS=5000)
            
            # Test connection
            self.client.admin.command('ping')
            
            self.db = self.client[db_name]
            log.info("✅ Connected to local MongoDB successfully!")
            return True
            
        except ServerSelectionTimeoutError:
            log.error(f"❌ Cannot connect to {host}:{port}")
            log.error("   Make sure MongoDB is running: mongod")
            return False
        except Exception as e:
            log.error(f"❌ Connection error: {e}")
            return False
    
    def connect_env(self, db_name: str = "moviebuzz") -> bool:
        """
        Connect using environment variables
        
        Supports:
        - MONGODB_URI (for Atlas)
        - MONGODB_HOST, MONGODB_PORT (for local)
        
        Returns:
            bool: Connection successful
        """
        # Try Atlas first
        uri = os.getenv('MONGODB_URI')
        if uri:
            return self.connect_atlas(uri, db_name)
        
        # Fall back to local
        host = os.getenv('MONGODB_HOST', 'localhost')
        port = int(os.getenv('MONGODB_PORT', 27017))
        return self.connect_local(host, port, db_name)
    
    def is_connected(self) -> bool:
        """Check if connected to MongoDB"""
        try:
            if self.client:
                self.client.admin.command('ping')
                return True
        except:
            pass
        return False
    
    def get_db(self):
        """Get database instance"""
        if not self.is_connected():
            log.error("❌ Not connected to MongoDB")
            return None
        return self.db
    
    def close(self):
        """Close connection"""
        if self.client:
            self.client.close()
            log.info("✅ MongoDB connection closed")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics"""
        if not self.is_connected() or self.db is None:
            return {}
        
        try:
            db_name = self.db.name if self.db is not None else "unknown"
            collections = self.db.list_collection_names() if self.db is not None else []
            stats = self.db.command("dbStats") if self.db is not None else {}
            return {
                "status": "connected",
                "database": db_name,
                "collections": collections,
                "stats": stats
            }
        except Exception as e:
            log.error(f"Error getting stats: {e}")
            return {}


# ═══════════════════════════════════════════════════════════════════════════════
#  USAGE EXAMPLES
# ═══════════════════════════════════════════════════════════════════════════════

def example_connect_atlas():
    """Example: Connect to MongoDB Atlas"""
    print("\n" + "="*70)
    print("EXAMPLE 1: Connect to MongoDB Atlas")
    print("="*70)
    
    # Get connection string from MongoDB Atlas dashboard
    atlas_uri = (
        "mongodb+srv://user:password@moviebuzz-cluster.mongodb.net"
        "/?retryWrites=true&w=majority"
    )
    
    conn = MongoDBConnection()
    if conn.connect_atlas(atlas_uri):
        print("\n✅ Successfully connected!")
        db = conn.get_db()
        if db is not None:
            print(f"Database: {db.name}")
            print(f"Collections: {db.list_collection_names()}")
    else:
        print("\n❌ Connection failed")


def example_connect_local():
    """Example: Connect to local MongoDB"""
    print("\n" + "="*70)
    print("EXAMPLE 2: Connect to Local MongoDB")
    print("="*70)
    
    conn = MongoDBConnection()
    if conn.connect_local("localhost", 27017):
        print("\n✅ Successfully connected!")
        db = conn.get_db()
        if db is not None:
            print(f"Database: {db.name}")
            print(f"Collections: {db.list_collection_names()}")
            
            # Print stats
            stats = conn.get_stats()
            if stats:
                print(f"\nDatabase Stats:")
                print(f"  Collections: {len(stats.get('collections', []))}")
    else:
        print("\n❌ Connection failed")
        print("\nTo start MongoDB locally:")
        print("  Windows:  mongod")
        print("  Mac:      brew services start mongodb-community")
        print("  Linux:    sudo systemctl start mongod")


def example_insert_data():
    """Example: Insert data to MongoDB"""
    print("\n" + "="*70)
    print("EXAMPLE 3: Insert Sample Data")
    print("="*70)
    
    conn = MongoDBConnection()
    if not conn.connect_local():
        return
    
    db = conn.get_db()
    if db is None:
        return
    
    # Create/get collection
    movies_collection = db['movies']
    
    # Sample data
    sample_movies = [
        {
            "movieId": 1,
            "title": "Toy Story (1995)",
            "genres": ["Adventure", "Animation", "Children", "Comedy", "Fantasy"],
            "rating": 4.15,
            "num_ratings": 81491
        },
        {
            "movieId": 2,
            "title": "Jumanji (1995)",
            "genres": ["Adventure", "Children", "Fantasy"],
            "rating": 3.20,
            "num_ratings": 34672
        }
    ]
    
    # Insert
    result = movies_collection.insert_many(sample_movies)
    print(f"\n✅ Inserted {len(result.inserted_ids)} documents")
    print(f"IDs: {result.inserted_ids}")
    
    # Read back
    doc_count = movies_collection.count_documents({})
    print(f"\nDocuments in collection: {doc_count}")
    for doc in movies_collection.find().limit(2):
        print(f"  - {doc['title']}")


def example_query_data():
    """Example: Query data from MongoDB"""
    print("\n" + "="*70)
    print("EXAMPLE 4: Query Data")
    print("="*70)
    
    conn = MongoDBConnection()
    if not conn.connect_local():
        return
    
    db = conn.get_db()
    if db is None:
        return
    
    movies = db['movies']
    
    # Find all
    print(f"\nTotal movies: {movies.count_documents({})}")
    
    # Find by title
    toy_story = movies.find_one({"title": "Toy Story (1995)"})
    if toy_story:
        print(f"\nFound: {toy_story['title']}")
        print(f"  Rating: {toy_story['rating']}")
        print(f"  Genres: {', '.join(toy_story['genres'])}")
    
    # Find by genre
    adventures_count = movies.count_documents({"genres": "Adventure"})
    print(f"\nAdventure movies: {adventures_count}")
    for movie in movies.find({"genres": "Adventure"}).limit(3):
        print(f"  - {movie['title']}")


# ═══════════════════════════════════════════════════════════════════════════════
#  QUICK START
# ═══════════════════════════════════════════════════════════════════════════════

def setup_from_env():
    """
    🔧 AUTO SETUP: Connect using environment variables
    Check .env file first, then try local connection
    """
    print("\n" + "="*70)
    print("🔧 MongoDB Auto-Setup from Environment")
    print("="*70)
    
    conn = MongoDBConnection()
    
    # Try environment variables first
    if conn.connect_env():
        print("\n✅ Connected using environment variables!")
        stats = conn.get_stats()
        if "database" in stats:
            print(f"\n📊 Database: {stats.get('database', 'unknown')}")
            print(f"📁 Collections: {stats.get('collections', [])}")
        conn.close()
        return True
    
    print("\n❌ Environment variables not set")
    print("\n📋 QUICK SETUP OPTIONS:\n")
    print("1️⃣  OPTION 1: Docker (Fastest - 2 minutes) 🐳")
    print("   docker run -d -p 27017:27017 --name moviebuzz-mongo mongo")
    print("   Then run: python mongo_connection.py\n")
    
    print("2️⃣  OPTION 2: MongoDB Atlas (Recommended - 10 minutes) ☁️")
    print("   1. Go to: https://www.mongodb.com/cloud/atlas")
    print("   2. Create free account → Create cluster")
    print("   3. Get connection string")
    print("   4. Add to .env: MONGODB_URI=<your-uri>")
    print("   5. Run: python mongo_connection.py\n")
    
    print("3️⃣  OPTION 3: Local MongoDB (15 minutes) 🖥️")
    print("   Windows: https://www.mongodb.com/try/download/community")
    print("   Mac:     brew install mongodb-community")
    print("   Linux:   sudo apt-get install mongodb")
    print("   Then: mongod (start service)")
    print("   Then: python mongo_connection.py\n")
    
    return False


if __name__ == "__main__":
    print("\n" + "="*70)
    print("🍃 MovieBuzz MongoDB Connection Manager")
    print("="*70)
    
    # Auto-setup from environment
    if setup_from_env():
        print("\n✨ Ready to use MongoDB!")
        print("\nUsage in Python:")
        print("  from mongo_connection import MongoDBConnection")
        print("  conn = MongoDBConnection()")
        print("  conn.connect_env()  # Uses MONGODB_URI or local")
        print("  db = conn.get_db()")
    else:
        print("\n📌 Next Steps:")
        print("   1. Choose setup option above")
        print("   2. Follow the instructions")
        print("   3. Run this script again")
